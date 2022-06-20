import numpy as np
from simulus.upcoming_us_sim.pressure_field import pressure_field
from scipy.ndimage import correlate
from scipy.signal import hilbert


def mesh_free_rf_simulator(x, z, rc, transmit_delays, probe, c=1540, db_threshold=-100, y=None):
    n_wavelengths = 1
    pitch = probe.pitch
    n_elements = probe.n_elements
    sampling_freq = 4 * probe.center_frequency

    x = x if isinstance(x, np.ndarray) else np.array(x)
    z = z if isinstance(z, np.ndarray) else np.array(z)
    y = y if isinstance(y, np.ndarray) or y is None else np.array(y)

    x_e = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch
    z_e = np.zeros((n_elements,))

    # Maximum distance
    max_dist = np.max(np.linalg.norm([x[..., None]-x_e, z[..., None]-z_e], axis=0))
    max_dist += np.max(transmit_delays)*c + n_wavelengths*c/probe.center_frequency

    # Frequency samples
    df = 1/2/(2*max_dist/c)
    frequency_step = df
    n_freq_samples = int(2 * np.ceil(probe.center_frequency/df) + 1)

    rf_spectrum = np.zeros((n_freq_samples, n_elements), dtype=np.complex64)
    _, spectrum, idx = pressure_field(x, z, transmit_delays, return_rx_spectrum=True, rc=rc, probe=probe,
                                      frequency_step=frequency_step, db_threshold=db_threshold, y_grid=y)
    rf_spectrum[idx] = spectrum

    n_rf = np.floor(sampling_freq/2/probe.center_frequency*(n_freq_samples-1)/2).astype(np.uint32)
    rf = np.fft.ifft(np.conj(rf_spectrum), n=n_rf*2, axis=0)
    rf = rf[:n_rf]*sampling_freq/4/probe.center_frequency

    def low_value_filter(rel_rf, rel_threshold):
        ret_val = 0.5*(1+np.tanh((rel_rf-rel_threshold)/(rel_threshold/10)))
        ret_val = np.round(ret_val/(rel_threshold/10))*(rel_threshold/10)
        return ret_val

    rf = rf*low_value_filter(np.abs(rf)/np.max(np.abs(rf)), rel_threshold=1e-5)
    return rf


def pseudo_iq_simulator(tissue_map, pulse_width=2, beam_width=1.5, speckle_variance=0.01, center_frequency=10e6,
                        speed_of_sound=1540):
    """
    Pseudo US simulator given a tissue map / image.

    Parameters:
    -----------
        tissue_map: np.ndarray
            A 2D image/map of a structure to simulate.

    Returns:
    --------
        Pseudo beamformed IQ image

    """

    tissue_map = tissue_map/255.0

    k = 2*np.pi*center_frequency/speed_of_sound  # wavenumber [rad/m]

    speckle_noise_map = np.random.random(tissue_map.shape)
    speckle_noise_map = (speckle_noise_map-np.mean(speckle_noise_map))*speckle_variance

    template_image = tissue_map*speckle_noise_map

    axial = np.arange(-10*pulse_width, 10*pulse_width+1)
    lateral = np.arange(-10*beam_width, 10*beam_width+1)

    axial_pulse_wave = np.sin(k*axial)*np.exp(-(axial**2)/(2*pulse_width**2))
    lateral_beam_profile = np.exp(-(lateral**2)/(2*beam_width**2))

    filtered_template = correlate(template_image, axial_pulse_wave[..., None], mode="nearest")
    filtered_template = correlate(filtered_template, lateral_beam_profile[None, ...], mode="nearest")

    vcap = hilbert(filtered_template)
    pseudo_iq = filtered_template + (1j*vcap)
    return pseudo_iq