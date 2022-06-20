import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from simulus.utilities.math_utilities import robust_linear_regression


def rf_to_iq(rf, sampling_freq, center_frequency, bandwidth=None):
    t = np.arange(rf.shape[0])/sampling_freq

    # Down-mixing
    iq = rf*np.exp(-1j*2*np.pi*center_frequency*t)[..., None]

    # Low pass
    if bandwidth:
        bandwidth_hz = center_frequency * bandwidth / 100.
        wn = bandwidth_hz/sampling_freq
    else:
        wn = np.min([2*center_frequency/sampling_freq, 0.5])

    [b, a] = butter(5, wn)
    iq = 2*filtfilt(b, a, iq, axis=0, padtype="odd", padlen=3*(max(len(b), len(a))-1))
    return iq


def image_polar_grid(image_shape, width, z_max, probe):
    array_width = (probe.n_elements-1)*probe.pitch
    z_0 = -array_width/2*(1+np.cos(width))/np.sin(width)
    radius = np.hypot(array_width/2, z_0)
    theta_grid, radius_grid = np.meshgrid(np.linspace(width/2, -width/2, image_shape[1])+np.pi/2,
                                          np.linspace(radius+2*probe.pitch, -z_0+z_max, image_shape[0]))
    return radius_grid*np.cos(theta_grid), radius_grid*np.sin(theta_grid)+z_0


def iq_to_bmode(iq, dynamic_range=40):
    """Converts IQ signals to 8-bit log-compressed US"""

    image = np.abs(iq)
    image = 20*np.log10(image/np.max(image))+dynamic_range
    return np.clip(255*image/dynamic_range, 0, 255).astype(np.uint8)


def time_gain_compensation(data):
    avg_axis = -1
    if len(data.shape) == 3:
        avg_axis = (0, avg_axis)

    if np.isrealobj(data):
        coefficients = np.mean(np.abs(hilbert(data)), axis=avg_axis)
    else:
        coefficients = np.mean(np.abs(data), axis=avg_axis)

    n1 = np.ceil(len(coefficients)/10).astype(np.uint32)
    n2 = np.floor(len(coefficients)*9/10).astype(np.uint32)
    p = min(200/(n2-n1)*100, 100)
    slope, intercept = robust_linear_regression(range(n1, n2), np.log(coefficients[n1:n2]), p)
    coefficients = np.exp(intercept+slope*range(len(coefficients)))
    coefficients = coefficients[0]/coefficients
    return data*coefficients[..., None]


