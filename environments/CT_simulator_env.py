import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
from typing import Optional
import gym
from gym import spaces
import copy
import open3d as o3d
from simulus.utilities import get_data_folder, transparent_cmap, oblique_slice, rotation_matrix_from_vectors, \
    find_surface_contours_of_ct, find_surface_contours_of_segmentation, find_centroid, map_point_to_plane


class CTSimulatorEnv(gym.Env):
    """
    The reader is referred to the project report for a proper introduction
    """

    metadata = {"render.modes": ["human", "rgb_array", "debug"]}

    def __init__(self,
                 *,
                 case_id=None,
                 horizon=200,
                 probe_us_depth=0.15,
                 probe_dev_threshold=0.05,
                 reward_weights: dict = None,
                 seed_v: Optional[int] = None):

        super(CTSimulatorEnv, self).__init__()

        self.seed_v = seed_v if seed_v is not None else None

        # Import necessary data.
        full_case_filename = f"case_{case_id:05d}.npy"
        print(full_case_filename)

        self.volume = np.load(get_data_folder() + "/KiTS19_CT_volumes/" + full_case_filename)
        self.segmentation = np.load(get_data_folder() + "/KiTS19_CT_segmentations/" + full_case_filename)
        self.spacing = np.load(get_data_folder() + "/KiTS19_CT_spacings/" + full_case_filename)

        print(
            f"Volume shape: {self.volume.shape} - Intensity min/max: {np.min(self.volume)} {np.max(self.volume)} - Spacing: {self.spacing}")

        self.window_size = 512

        self.obs_size = self.volume.shape[1]
        self.crop_obs_size = self.cartesian_to_pixel(np.array([0, probe_us_depth, probe_us_depth]), self.spacing)[2]

        # Avoids unpleasant rounding errors down the line
        if (self.crop_obs_size % 2 != 0):
            self.crop_obs_size += 1

        print(f"Size of the cropped image: {self.crop_obs_size}x{self.crop_obs_size}")

        N_CHANNELS = 1 # One channel since we are investigating grayscale images

        # Upper bound is realistically much lower than the following upper limit, but difficult to know in advance
        self.reward_range = (0, float(self.crop_obs_size * self.crop_obs_size))
        self.reward_weights = reward_weights if reward_weights is not None else {"distance": 10, "kidney_area": 1}

        # Extract abdominal surface contour points and map from pixel space to cartesian coordinates in meters.
        self.surface_points = find_surface_contours_of_ct(self.volume)
        self.surface_points = self.surface_points * self.spacing / 1000

        print(f"Contour extents (in meters): {np.min(self.surface_points, axis=0)} "
              f"{np.max(self.surface_points, axis=0)} \n")

        self.surface_centroid = find_centroid(self.surface_points)

        # Create Open3D PointCloud object based on points
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.surface_points)

        # Randomly downsample the point cloud to a fixed number of points (here 200 000 if larger than 200 000) to
        # reduce the computational burden of the point cloud.
        self.pcd = self.pcd.random_down_sample(sampling_ratio=200000 / len(self.pcd.points) if len(self.pcd.points) >= 200000 else 1.0)
        # Calculate the normals on the surface.
        self.pcd.estimate_normals(fast_normal_computation=True)
        # The normal estimation does not ensure a consistent tangent plane, therefore apply a post-processing step to
        # align the normals with a fitted plane of a set number of points.
        self.pcd.orient_normals_consistent_tangent_plane(30)
        # Color the point cloud for improved visualization.
        self.pcd.paint_uniform_color([0.29, 0.25, 0.21])

        # Find the contours of the two kidneys and map them to cartesian space.
        # Also find their centroid, resulting in a point located between the left and right kidney.
        self.kidney_points = find_surface_contours_of_segmentation(self.segmentation)
        self.kidney_points = self.kidney_points * self.spacing / 1000
        self.kidney_centroid_both = find_centroid(self.kidney_points)

        # Using the kidney centroid point we can distinguish between the left and right kidney.
        # Further we create two separate Open3D point clouds.
        left_kidney_points = self.kidney_points[self.kidney_points[:, 2] >= self.kidney_centroid_both[2]]
        right_kidney_points = self.kidney_points[self.kidney_points[:, 2] < self.kidney_centroid_both[2]]

        self.pcd_left_kidney = o3d.geometry.PointCloud()
        self.pcd_left_kidney.points = o3d.utility.Vector3dVector(left_kidney_points)
        self.pcd_left_kidney.estimate_normals(fast_normal_computation=True)
        self.pcd_left_kidney.orient_normals_consistent_tangent_plane(30)

        self.pcd_right_kidney = o3d.geometry.PointCloud()
        self.pcd_right_kidney.points = o3d.utility.Vector3dVector(right_kidney_points)
        self.pcd_right_kidney.estimate_normals(fast_normal_computation=True)
        self.pcd_right_kidney.orient_normals_consistent_tangent_plane(30)

        self.kidney_center_point = np.zeros(3)
        self.kidney_center_point_px = np.zeros(3, dtype=int)
        self.left_kidney_center_point = self.pcd_left_kidney.get_center()
        self.right_kidney_center_point = self.pcd_right_kidney.get_center()
        self.left_kidney_center_point_px = self.cartesian_to_pixel(self.left_kidney_center_point, self.spacing)
        self.right_kidney_center_point_px = self.cartesian_to_pixel(self.right_kidney_center_point, self.spacing)

        # Boundaries and limits
        self.min_obs_bounds = np.zeros(3)
        self.max_obs_bounds = self.pixel_to_cartesian(np.array(self.volume.shape), self.spacing)
        self.surface_x_min, self.surface_y_min, self.surface_z_min = self.pcd.get_min_bound()
        self.surface_x_max, self.surface_y_max, self.surface_z_max = self.pcd.get_max_bound()
        self.min_termination_bounds = np.zeros(3)
        self.max_termination_bounds = copy.deepcopy(self.max_obs_bounds)
        self.max_termination_bounds[1] = self.max_termination_bounds[1] * 0.85
        self.probe_deviation_threshold = probe_dev_threshold

        self.horizon = horizon # Meaning maximum number of allowed steps for each episode

        # The following is not to be confused with Gym's TimeLimit wrapper,
        # which provides better adherence to the SB3 implementation
        self.current_step = 0

        # State variables
        self.pcd_current_pos = np.zeros(3)
        self.probe_position = np.zeros(3)
        self.complete_image = np.zeros((self.obs_size, self.obs_size), np.uint8)
        self.kidney_segmentation = np.zeros((self.obs_size, self.obs_size), int)

        #Image channel is stated first, in compliance with the PyTorch API
        self.cropped_image = np.zeros((N_CHANNELS, self.crop_obs_size, self.crop_obs_size), np.uint8)
        self.cropped_kidney_segmentation = np.zeros((self.crop_obs_size, self.crop_obs_size), int)
        self.grids = (0, 0, 0)

        self.rotated_image = np.zeros((self.obs_size, self.obs_size), np.uint8)
        self.rotated_segm_image = np.zeros((self.obs_size, self.obs_size), int)

        # Relevant points in 3D mapped to 2D in the oblique slice plane.
        self.surface_coord_image = (0, 0)
        self.kidney_coord_image = (0, 0)
        self.kidney_point_image = (0, 0)
        self.rotated_image_point = (0, 0)

        self.action_space = spaces.Box(low=(-1)*np.array([self.surface_x_max/100, self.surface_y_max/100, self.surface_z_max/100]),
                                       high=np.array([self.surface_x_max/100, self.surface_y_max/100, self.surface_z_max/100]),
                                       shape=(3,), dtype=np.float64)

        # Define a 2-D observation space
        # Image space defined by convention C x H x W, since PyTorch uses channel first format.
        self.observation_space = spaces.Dict(
            {"probe": spaces.Box(low=self.min_obs_bounds,
                                 high=self.max_obs_bounds,
                                 shape=(3,), dtype=np.float64),
            "image": spaces.Box(low=0, high=255, shape=(N_CHANNELS, self.crop_obs_size, self.crop_obs_size), dtype=np.uint8),
            }
        )

        self.window = None
        self.clock = None


    def _get_obs(self):
        """
        Since we will need to compute observations both in reset and step, it is often convenient to have a (private)
        method _get_obs that translates the environment’s state into an observation.
        """

        obs_dict = {}
        obs_dict["probe"] = self.probe_position
        obs_dict["image"] = self.cropped_image

        return obs_dict

    def _get_info(self):
        info_dict = {}
        info_dict["kidney_mask"] = self.kidney_segmentation
        info_dict["cropped_kidney_mask"] = self.cropped_kidney_segmentation
        info_dict["kidney_center_point"] = self.kidney_center_point
        info_dict["point_cloud_pos"] = self.pcd_current_pos

        # Simply for information in the terminal during training of the robot
        if self.current_step >= self.horizon:
            print("Info: time_limit_reached")

        return info_dict

    def step(self, action, verbose=True):
        if verbose:
            print(f"Selected action: {action}")

        self.probe_position += action
        self.current_step += 1

        pcd_index_point = np.argmin(np.linalg.norm(np.array(self.pcd.points)[:, ...]-self.probe_position, axis=1))
        self.pcd_current_pos = np.array(self.pcd.points)[pcd_index_point]

        # An episode is done if the agent has reached a terminal state
        episode_done = self._check_termination()

        probe_position_px = self.cartesian_to_pixel(self.probe_position, self.spacing)

        # Plane direction
        if self.probe_position[2] > self.kidney_centroid_both[2]:
            surface_to_kidney_vector = self.left_kidney_center_point - self.probe_position
            self.kidney_center_point = self.left_kidney_center_point
            self.kidney_center_point_px = self.left_kidney_center_point_px
        elif self.probe_position[2] <= self.kidney_centroid_both[2]:
            surface_to_kidney_vector = self.right_kidney_center_point - self.probe_position
            self.kidney_center_point = self.right_kidney_center_point
            self.kidney_center_point_px = self.right_kidney_center_point_px
        else:
            raise ValueError("Error in the random surface point")

        self.rot_matrix_align_z_towards_kidney = rotation_matrix_from_vectors([0, 0, 1], surface_to_kidney_vector)
        print(f"--- Step --- Probe center (pixel): {probe_position_px} "
              f"--- Kidney center (pixel): {self.kidney_center_point_px}")

        # Normal is perpendicular to z direction
        normal_px = self.cartesian_to_pixel(np.dot(self.rot_matrix_align_z_towards_kidney, [0, 1, 0]),
                                            self.spacing)

        # Oblique slices of the CT and segmentation volumes. We also extract the 3D grids of the plane.
        self.complete_image, self.grids = oblique_slice(self.volume, probe_position_px, normal_px,
                                                        return_grids=True)
        self.kidney_segmentation = oblique_slice(self.segmentation > 0, probe_position_px, normal_px)
        # We know the relevant points in 3D, but to use them in the oblique slice plane, we need to map them from
        # 3D to 2D.
        self.surface_coord_image = map_point_to_plane(probe_position_px, plane_grids=self.grids)
        self.kidney_coord_image = map_point_to_plane(self.kidney_center_point_px, plane_grids=self.grids)
        surface_to_kidney_vector_px = self.cartesian_to_pixel(surface_to_kidney_vector, self.spacing)
        kidney_vec_px = probe_position_px + surface_to_kidney_vector_px
        self.kidney_point_image = map_point_to_plane(kidney_vec_px, plane_grids=self.grids)
        displacement_vec = np.array(self.surface_coord_image) - np.array(self.kidney_point_image)
        unit_displacement_vec = displacement_vec / np.linalg.norm(displacement_vec)
        angle = math.degrees(np.arccos(np.dot(unit_displacement_vec, np.array([-1, 0]))))
        if self.surface_coord_image[1] < self.kidney_point_image[1]:
            angle *= -1
        padded_image = np.pad(self.complete_image, ((250, 250), (250, 250)), mode='constant', constant_values=0)
        padded_segm_image = np.pad(self.kidney_segmentation, ((250, 250), (250, 250)), mode='constant',
                                   constant_values=0)
        self.rotated_image = skimage.transform.rotate(padded_image, angle, resize=False,
                                                      center=(self.kidney_point_image[1] + 250,
                                                              self.kidney_point_image[0] + 250),
                                                      preserve_range=True)
        self.rotated_segm_image = skimage.transform.rotate(padded_segm_image, angle, resize=False,
                                                           center=(self.kidney_point_image[1] + 250,
                                                                   self.kidney_point_image[0] + 250),
                                                           preserve_range=True)

        kidney_to_probe_dist = np.around(np.linalg.norm(displacement_vec)).astype(int)
        self.rotated_image_point = (self.kidney_point_image[0] + 250 - kidney_to_probe_dist,
                                    self.kidney_point_image[1] + 250)
        self.cropped_image[0, :, :] = self.rotated_image[
                                      self.rotated_image_point[0]:self.rotated_image_point[0] + self.crop_obs_size,
                                      self.rotated_image_point[1] - int(self.crop_obs_size/2):self.rotated_image_point[
                                                                            1] + int(self.crop_obs_size/2)].round().astype(np.uint8)
        self.cropped_kidney_segmentation = self.rotated_segm_image[
                                           self.rotated_image_point[0]:self.rotated_image_point[0] + self.crop_obs_size,
                                           self.rotated_image_point[1] - int(self.crop_obs_size/2):self.rotated_image_point[
                                                                                 1] + int(self.crop_obs_size/2)]
        self.rotated_image_point = (0, self.cropped_image.shape[2] // 2)
        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward(observation, info)

        return observation, reward, episode_done, info


    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            kidney_view_on_init: bool = False,
            options: Optional[dict] = None,
    ):

        _seed = seed if seed is not None else self.seed_v

        super().reset(seed=_seed, return_info=False)

        self.current_step = 0

        if kidney_view_on_init:
            # Find a suitable kidney view_candidate
            # Points toward the centroid to avoid an "optimal solution" from the start
            pcd_index_kidney_view = np.argmin(
                np.linalg.norm(self.kidney_centroid_both[[0, 1]] - np.array(self.pcd.points)[:, [0, 1]], axis=1))
            self.probe_position = np.array(self.pcd.points)[pcd_index_kidney_view]
        else:
            # Create possible point and normal samples based on location of centroid.
            init_pos_samples = np.array(self.pcd.points)[np.array(self.pcd.points)[:, 1] < self.surface_centroid[1]]
            init_normals_samples = np.array(self.pcd.normals)[np.array(self.pcd.points)[:, 1] < self.surface_centroid[1]]

            # Randomly draw index
            init_index = np.random.choice(init_pos_samples.shape[0], 1, replace=False)

            while(any(init_pos_samples[init_index][0] < self.min_termination_bounds) or any(init_pos_samples[init_index][0] > self.max_termination_bounds)):
                init_index = np.random.choice(init_pos_samples.shape[0], 1, replace=False)

            # Random surface point
            self.probe_position = init_pos_samples[init_index][0]

        probe_position_px = self.cartesian_to_pixel(self.probe_position, self.spacing)

        # Plane direction
        # Vector from the tip of the probe points to the closest kidney, regardless of location
        if self.probe_position[2] > self.kidney_centroid_both[2]:
            surface_to_kidney_vector = self.left_kidney_center_point - self.probe_position
            self.kidney_center_point = self.left_kidney_center_point
            self.kidney_center_point_px = self.left_kidney_center_point_px
        elif self.probe_position[2] <= self.kidney_centroid_both[2]:
            surface_to_kidney_vector = self.right_kidney_center_point - self.probe_position
            self.kidney_center_point = self.right_kidney_center_point
            self.kidney_center_point_px = self.right_kidney_center_point_px
        else:
            raise ValueError("Error in the random surface point")

        self.rot_matrix_align_z_towards_kidney = rotation_matrix_from_vectors([0, 0, 1], surface_to_kidney_vector)

        print(f"--- Reset --- Probe center (pixel): {probe_position_px} "
              f"--- Kidney center (pixel): {self.kidney_center_point_px}")

        # Normal is perpendicular to z direction
        normal_px = self.cartesian_to_pixel(np.dot(self.rot_matrix_align_z_towards_kidney, [0, 1, 0]), self.spacing)

        # Oblique slices of the CT and segmentation volumes. We also extract the 3D grids of the plane.
        self.complete_image, self.grids = oblique_slice(self.volume, probe_position_px, normal_px, return_grids=True)
        self.kidney_segmentation = oblique_slice(self.segmentation > 0, probe_position_px, normal_px)

        # We know the relevant points in 3D, but to use them in the oblique slice plane, we need to map them from 3D to 2D.
        self.surface_coord_image = map_point_to_plane(probe_position_px, plane_grids=self.grids)
        self.kidney_coord_image = map_point_to_plane(self.kidney_center_point_px, plane_grids=self.grids)

        surface_to_kidney_vector_px = self.cartesian_to_pixel(surface_to_kidney_vector, self.spacing)
        kidney_vec_px = probe_position_px + surface_to_kidney_vector_px
        self.kidney_point_image = map_point_to_plane(kidney_vec_px, plane_grids=self.grids)

        # Rotation of the image to ensure consistent slicing
        displacement_vec = np.array(self.surface_coord_image) - np.array(self.kidney_point_image)
        unit_displacement_vec = displacement_vec / np.linalg.norm(displacement_vec)
        angle = math.degrees(np.arccos(np.dot(unit_displacement_vec, np.array([-1, 0]))))
        if self.surface_coord_image[1] < self.kidney_point_image[1]:
            angle *= -1

        padded_image = np.pad(self.complete_image, ((250, 250), (250, 250)), mode='constant', constant_values=0)
        padded_segm_image = np.pad(self.kidney_segmentation, ((250, 250), (250, 250)), mode='constant', constant_values=0)

        self.rotated_image = skimage.transform.rotate(padded_image, angle, resize=False,
                                                      center=(self.kidney_point_image[1]+250, self.kidney_point_image[0]+250),
                                                      preserve_range=True)
        self.rotated_segm_image = skimage.transform.rotate(padded_segm_image, angle, resize=False,
                                                      center=(self.kidney_point_image[1]+250, self.kidney_point_image[0]+250),
                                                      preserve_range=True)

        kidney_to_probe_dist = np.around(np.linalg.norm(displacement_vec)).astype(int)

        self.rotated_image_point = (self.kidney_point_image[0]+250-kidney_to_probe_dist, self.kidney_point_image[1]+250)

        self.cropped_image[0, :, :] = self.rotated_image[self.rotated_image_point[0]:self.rotated_image_point[0]+self.crop_obs_size,
                             self.rotated_image_point[1]-int(self.crop_obs_size/2):self.rotated_image_point[1]+int(self.crop_obs_size/2)].round().astype(np.uint8)
        self.cropped_kidney_segmentation = self.rotated_segm_image[self.rotated_image_point[0]:self.rotated_image_point[0]+self.crop_obs_size,
                             self.rotated_image_point[1]-int(self.crop_obs_size/2):self.rotated_image_point[1]+int(self.crop_obs_size/2)]

        self.rotated_image_point = (0, self.cropped_image.shape[2] // 2)

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation

    def render(self, mode="human", timeout=True):
        """
        Render functionality for three different modes:
        "debug": renders the complete 3D volume and resulting slices, both complete slice and the cropped slice, the
        latter of which is fed to the learning agent.
        "human": renders only the cropped slice that is fed to the learning agent.
        "rgb_array" returns a 2D pixel array.
        """

        if mode == "human":
            observation = self._get_obs()
            fig, ax = plt.subplots(1)
            ax.imshow(observation["image"][0, :, :], cmap='gray')
            plt.tight_layout()
            plt.show()

        elif mode == "debug":
            # pop up a window and render
            observation = self._get_obs()
            probe_position_px = self.cartesian_to_pixel(observation["probe"], self.spacing)

            # For convenience when visualizing, we create two coordinate frames.
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0.0, 0.0, 0.0])

            surface_frame = copy.deepcopy(world_frame).translate(observation["probe"])
            surface_frame.rotate(self.rot_matrix_align_z_towards_kidney)

            # We then draw the point cloud of the surface, kidneys, world frame and surface frame.
            o3d.visualization.draw_geometries([self.pcd, self.pcd_right_kidney, self.pcd_left_kidney, world_frame, surface_frame])

            # Finally, we can visualize the oblique slice with the relevant points.
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(self.complete_image, cmap='gray')
            ax[0].imshow(self.kidney_segmentation, cmap=transparent_cmap(plt.cm.Reds), alpha=0.9)
            ax[0].plot(self.surface_coord_image[1], self.surface_coord_image[0], 'go')
            ax[0].plot(self.kidney_coord_image[1], self.kidney_coord_image[0], 'bo')
            ax[0].plot(self.kidney_point_image[1], self.kidney_point_image[0], 'yx')
            ax[1].imshow(self.rotated_image, cmap='gray')
            ax[2].imshow(observation["image"][0, :, :], cmap='gray')
            ax[2].imshow(self.cropped_kidney_segmentation, cmap=transparent_cmap(plt.cm.Reds), alpha=0.9)
            ax[2].plot(self.rotated_image_point[1], self.rotated_image_point[0], 'go')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.grids[0], self.grids[1], self.grids[2], facecolors=np.array([self.complete_image] * 3).transpose((1, 2, 0)) / 255.0, alpha=0.95)
            ax.scatter(*probe_position_px.T, color='red', s=100)
            plt.show()

        elif mode == "rgb_array":
            # return image frame suitable for video
            observation = self._get_obs()
            return observation["image"][0,:,:]

        else:
            super(CTSimulatorEnv, self).render(mode=mode) # just raise an exception

        return


    def close(self):
        if self.window is not None:
            plt.close()


    def pixel_to_cartesian(self, pixel_value, spacing):
        """
        Map discrete CT pixel values to cartesian space
        """
        return pixel_value * spacing / 1000

    def cartesian_to_pixel(self, cartesian_value, spacing):
        """
        Map from cartesian to pixel space.
        """
        return np.around(1000 * cartesian_value / spacing).astype(int)


    def _get_reward(self, observation, info):
        step_reward = 0

        self.kidney_mask = info["cropped_kidney_mask"]
        self.kidney_center = info["kidney_center_point"]

        #diff function between the probe point and the closest kidney's center point
        inv_distance = 1/np.linalg.norm(observation["probe"]-self.kidney_center)
        kidney_reward = float(np.sum(self.kidney_mask))

        step_reward += self.reward_weights["kidney_area"] * kidney_reward
        step_reward += self.reward_weights["distance"] * inv_distance
        step_reward += (-10000)*self.probe_distance_from_surface

        print(f"--- Reward --- Kidney mask contribution: {self.reward_weights['kidney_area'] * kidney_reward} "
              f"--- Probe to kidney center contribution: {self.reward_weights['distance'] * inv_distance} "
              f"--- Probe to surface contribution: {(-10000)*self.probe_distance_from_surface} "
              f"\n--- Reward --- Total step reward: {step_reward}")

        return step_reward


    def _check_termination(self):
        """
        Checks whether criteria for terminating an episode have been fulfilled, with the following conditions causing termination:
            - Collision with table or any other form of placement near the back of the patient
            - Deviates from the body surface beyond the "self.probe_deviation_threshold" limit
            - Falls outside of the bounding box of the patient on the table

        Returns:
            bool: True if episode is terminated
        """

        termination = False

        if self.probe_distance_from_surface > self.probe_deviation_threshold:
            print("--- Terminate --- Probe is no longer on the surface ---")
            termination = True
        elif any(self.probe_position < self.min_termination_bounds) or any(self.probe_position > self.max_termination_bounds):
            print("--- Terminate --- Probe is outside of the bounds of the human bounding box ---")
            termination = True

        return termination

    @property
    def probe_distance_from_surface(self):
        return np.linalg.norm(self.probe_position-self.pcd_current_pos)

if __name__ == "__main__":
    # Make and instantiate the environment
    base_env = gym.make('CTsim-v0', case_id=2, seed_v=42)
    # Note that we need to seed the action space separately from the environment to ensure reproducible samples.
    base_env.action_space.seed(42)

    # training variables
    n_episodes = 1
    max_n_steps = 100

    for episode in range(n_episodes):
        #reset the environment
        obs = base_env.reset()

        # Select "debug" for complete visualisation, "human" for what the robot agent sees
        base_env.render(mode="debug")

        done = False
        sum_rewards = 0.0

        for i in range(max_n_steps):
            # Sample a random action from the list of available actions
            action = base_env.action_space.sample()

            # Perform this action on the environment
            obs, reward, done, info = base_env.step(action)

            print(f"Step: {i+1}, observation: {obs}, reward: {reward}, done: {done}")

            sum_rewards += reward

            # Render the frame
            base_env.render(mode="human")

            #If true, you may need to end the simulation or reset the environment to restart the episode.
            if done:
                print("Terminal state has been reached, with net reward {}".format(sum_rewards))
                obs = base_env.reset(return_info=False)
                break

        base_env.close()