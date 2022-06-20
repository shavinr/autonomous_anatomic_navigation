import sys
sys.path.append('/home/shahoud/Documents/Act]bbky/Kyboppgaven/ROMO-RL/simulus')

import pydicom
import os
import numpy as np
import collections
import glob
import warnings
from simulus.utilities import get_data_folder

try:
    import nibabel as nib
except ImportError:
    warnings.warn("Module nibabel not possible to import, hence KiTS19Importer will not work. Run 'pip install nibabel'"
                  " if you need it")


class CTVolumeImporter(object):
    def __init__(self, filepath):
        if os.path.isdir(filepath):
            self.files = glob.glob(filepath + "/*.dcm", recursive=False)
        else:
            self.files = [filepath]

        self.dcm_objects = self._load_dcm_objects(self.files)
        self.spacing = [float(self.dcm_objects[0].PixelSpacing[0]),
                        float(self.dcm_objects[0].PixelSpacing[1]),
                        float(self.dcm_objects[0].SliceThickness)]
        self.volume = np.array([dcm_object.pixel_array for dcm_object in self.dcm_objects]).transpose((1, 2, 0))

    @property
    def extent(self):
        Extent = collections.namedtuple("Extent", ["xy", "xz", "yz", "yx"])
        x = 0, self.volume.shape[0]*self.spacing[0]
        y = 0, self.volume.shape[1]*self.spacing[1]
        z = 0, self.volume.shape[2]*self.spacing[2]
        ext = Extent(xy=[*x, *y], xz=[*x, *z], yz=[*y, *z], yx=[*y, *x])
        return ext

    @staticmethod
    def _load_dcm_objects(files):
        dcm_objects = [pydicom.dcmread(fpath) for fpath in files]
        dcm_objects = [dcm_object for dcm_object in dcm_objects if hasattr(dcm_object, "SliceLocation")]
        dcm_objects = sorted(dcm_objects, key=lambda s: s.SliceLocation)
        return dcm_objects

    def get_volume(self, hounsfield_units=True):
        volume = self.volume
        if hounsfield_units:
            intercept = self.dcm_objects[0].RescaleIntercept
            slope = self.dcm_objects[0].RescaleSlope
            if slope != 1:
                volume = slope*volume
            volume = volume + np.int16(intercept)
        return volume


class KiTSImporter(object):
    """Importer for KiTS data.
    """

    def __init__(self, filepath=None, hu_units=False, load_on_init=False):
        self.filepath = self.case_path = filepath
        self.segmentation = None
        self.volume = None
        self.spacing = None

        if load_on_init:
            self.volume, self.segmentation, self.spacing = self.load_data(self.filepath, hu_units=hu_units,
                                                                          return_spacing=True)

    @property
    def case_id(self):
        return os.path.basename(self.case_path)

    @staticmethod
    def load_volume(case_path, as_ndarray=True, hu_units=True, return_spacing=False):
        vol = nib.load(os.path.join(case_path, "imaging.nii.gz"))
        spacing = vol.header.get_zooms()

        if as_ndarray:
            vol = vol.get_fdata().astype(np.int32)
        else:
            vol = vol.get_fdata()

        if not hu_units:
            vol = KiTSImporter.hu_to_grayscale(vol, hu_min=np.int32(-512), hu_max=np.int32(512))

        if return_spacing:
            return vol, spacing
        return vol

    @staticmethod
    def load_segmentation(case_path, as_ndarray=True):
        seg = nib.load(os.path.join(case_path, "segmentation.nii.gz"))
        if as_ndarray:
            seg = seg.get_fdata().astype(np.uint32)  # ! Andreas' update
        else:
            seg = seg.get_fdata()
        return seg

    def load_data(self, case_path, as_ndarray=True, hu_units=True, return_spacing=False):
        volume = nib.load(os.path.join(case_path, "imaging.nii.gz"))
        segmentation = nib.load(os.path.join(case_path, "segmentation.nii.gz"))
        spacing = volume.header.get_zooms()

        if as_ndarray:
            volume = volume.get_fdata().astype(np.int32)
            segmentation = segmentation.get_fdata().astype(np.uint32)

        if not hu_units:
            volume = self.hu_to_grayscale(volume, hu_min=np.int32(-512), hu_max=np.int32(512))

        if return_spacing:
            return volume, segmentation, spacing
        return volume, segmentation

    @staticmethod
    def hu_to_grayscale(volume, hu_min: np.int32, hu_max: np.int32):
        if hu_min is not None or hu_max is not None:
            volume = np.clip(volume, hu_min, hu_max)

        # Scale to values between 0 and 1
        mxval = np.max(volume)
        mnval = np.min(volume)
        im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

        # Return values scaled to 0-255 range, but *not cast to uint8*
        im_volume = 255*im_volume

        return im_volume.astype(np.uint8)

    @property
    def extent(self):
        Extent = collections.namedtuple("Extent", ["xy", "xz", "yz", "yx"])
        x = 0, self.volume.shape[0]*self.spacing[0]
        y = 0, self.volume.shape[1]*self.spacing[1]
        z = 0, self.volume.shape[2]*self.spacing[2]
        ext = Extent(xy=[*x, *y], xz=[*x, *z], yz=[*y, *z], yx=[*y, *x])
        return ext

def import_all_files():
    dataset_dir = get_data_folder() + "/KiTS19/"
    save_path_volumes = get_data_folder() + "/KiTS19_CT_volumes/"
    save_path_segmentations = get_data_folder() + "/KiTS19_CT_segmentations/"
    save_path_spacings = get_data_folder() + "/KiTS19_CT_spacings/"
    if not os.path.exists(save_path_spacings):
        os.makedirs(save_path_volumes, exist_ok=True)
        os.makedirs(save_path_segmentations, exist_ok=True)
        os.makedirs(save_path_spacings, exist_ok=True)

    total_n_cases = len(list(os.walk(dataset_dir))) - 1
    for case_id in range(0, total_n_cases):
        try:
            full_case_id = f"case_{case_id:05d}"
        except ValueError:
            print("failed to convert")


        print(full_case_id)

        if not os.path.isfile(save_path_volumes + full_case_id + ".npy"):
            print("Directory did not exist for case no.", case_id, ", saving the volume and segm numpy arrays...")

            filepath = os.path.join(dataset_dir, full_case_id)
            volume, spacing = KiTSImporter().load_volume(filepath, as_ndarray=True, hu_units=False, return_spacing=True)
            segmentation = KiTSImporter().load_segmentation(filepath, as_ndarray=True)

            print(save_path_volumes + full_case_id + ".npy")

            np.save(os.path.join(save_path_volumes, full_case_id), volume)
            np.save(os.path.join(save_path_segmentations, full_case_id), segmentation)
            np.save(os.path.join(save_path_spacings, full_case_id), spacing)

def retrieve_loaded_files(case_id):
    # Import necessary data. The output volume and segmentation follows the convention (axial, coronal, sagittal).
    full_case_filename = f"case_{case_id:05d}.npy"
    print(full_case_filename)
    volume = np.load(get_data_folder() + "/KiTS19_CT_volumes/" + full_case_filename)
    segmentation = np.load(get_data_folder() + "/KiTS19_CT_segmentations/" + full_case_filename)
    spacing = np.load(get_data_folder() + "/KiTS19_CT_spacings/" + full_case_filename)
    return volume, segmentation, spacing

def old_and_robust_importer_func():
    dataset_dir = get_data_folder() + "/KiTS19/"
    save_path_volumes = get_data_folder() + "/KiTS19_CT_volumes/"
    save_path_segmentations = get_data_folder() + "/KiTS19_CT_segmentations/"

    total_n_cases = 18
    for case_id in range(0, total_n_cases):
        try:
            full_case_id = f"case_{case_id:05d}"
        except ValueError:
            print("failed to convert")

        print(full_case_id)

        if not os.path.isfile(save_path_volumes + full_case_id + ".npy"):
            print("Directory did not exist for case no. ", case_id, ", saving the volume and segm numpy arrays...")

            case_path = os.path.join(dataset_dir, full_case_id)
            volume = KiTSImporter().load_volume(case_path, hu_units=False)
            segmentation = KiTSImporter().load_segmentation(case_path)

            print(os.path.join(save_path_volumes, full_case_id))
            np.save(os.path.join(save_path_volumes, full_case_id), volume)
            np.save(os.path.join(save_path_segmentations, full_case_id), segmentation)
    return

if __name__ == "__main__":
    import_all_files()