from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulus.utilities import make_wrapped_env


if __name__ == "__main__":

    model_path = "./PPO_probe_navigation_training/MIP_2022-06-18-17-43/saved_model_wrapped_env"
    stats_path = "./PPO_probe_navigation_training/MIP_2022-06-18-17-43/vec_normalize.pkl"

    num_cpu = 1

    # Load the saved statistics
    loaded_env = VecNormalize(SubprocVecEnv(
        [make_wrapped_env("CTsim-v0", training=False, seed=i) for i in range(num_cpu)]))

    # no need for updating the model or performing reward normalization at test time
    # Turn off updates and reward normalization
    loaded_env.training = False
    loaded_env.norm_reward = False

    # Load the trained agent
    loaded_model = PPO.load(model_path, env=loaded_env, print_system_info=True)

    # Evaluate the agent
    ep_rewards = []
    n_eval_episodes = 10

    for i in range(n_eval_episodes):
        done = False
        cum_ep_reward = 0
        obs = loaded_env.reset()
        loaded_env.env_method("render")

        plots = []
        fig, ax = plt.subplots()
        while not done:
            action, _states = loaded_model.predict(obs)
            print(f"Action: {action}")
            obs, reward, done, info = loaded_env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            cum_ep_reward += reward
            loaded_env.env_method("render", mode="human")

            if done:
                print(f"Episode reward: {cum_ep_reward}")
                ep_rewards.append(cum_ep_reward)

    loaded_env.close()