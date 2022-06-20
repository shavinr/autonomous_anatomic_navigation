import numpy as np

from skimage import filters
from skimage.measure import regionprops, label, find_contours
from scipy import ndimage as ndi


def find_surface_contours_of_ct(volume):
    points = []
    for idx, axial_slice in enumerate(volume):
        regprops, mask = surface_regionprops_ct_slice(axial_slice, return_mask=True)
        contour = find_contours(mask)

        if len(contour):
            points.extend(np.array(([idx]*len(contour[0]), *contour[0].T)).T)
    return np.array(points)


def find_surface_contours_of_segmentation(segmentation):
    kidney_points = []
    for idx, axial_slice in enumerate(segmentation):
        mask = axial_slice == 1
        mask = ndi.binary_fill_holes(mask)
        label_image = label(mask)

        if np.max(label_image) > 0:
            for midx in range(1, np.max(label_image)+1):
                contour = find_contours(label_image == midx)

                if len(contour):
                    kidney_points.extend(np.array(([idx]*len(contour[0]), *contour[0].T)).T)

    return np.array(kidney_points)


def surface_regionprops_ct_slice(ct_slice, return_mask=False):
    threshold_value = filters.threshold_otsu(ct_slice)
    thresholded_slice = ct_slice > threshold_value
    thresholded_slice = ndi.binary_fill_holes(thresholded_slice)

    reg_props_islands = regionprops(label(thresholded_slice))
    surface_regionprops = reg_props_islands[np.argmax([reg.area for reg in reg_props_islands])]

    if return_mask:
        mask = np.zeros_like(thresholded_slice)
        mask[surface_regionprops.slice] = surface_regionprops.image
        return surface_regionprops, mask
    return surface_regionprops
