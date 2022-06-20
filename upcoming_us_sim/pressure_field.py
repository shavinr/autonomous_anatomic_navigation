import numpy as np

from simulus.utilities import matlab_mod


def pressure_field(x_grid, z_grid, transmit_delays, probe, c=1540, return_rx_spectrum=False, rc=None,
                   y_grid=None, frequency_step=None, db_threshold=-60):

    apodization = np.ones((probe.n_elements,)).astype(np.float32)
    n_wavelengths = 1
    min_lambda = c/(probe.center_frequency*(1+probe.bandwidth/200))
    element_width = probe.width
    n_elements = probe.n_elements

    n_element_segments = np.ceil(element_width/min_lambda)

    x_grid = x_grid if isinstance(x_grid, np.ndarray) else np.array(x_grid)
    y_grid = y_grid if isinstance(y_grid, np.ndarray) or y_grid is None else np.array(y_grid)
    z_grid = z_grid if isinstance(z_grid, np.ndarray) else np.array(z_grid)

    x_e, z_e = probe.get_tranducer_element_centers()
    th_e = np.zeros((n_elements,))

    segment_length = element_width/n_element_segments
    tmp = -element_width/2 + segment_length/2 + np.arange(0, n_element_segments)*segment_length
    xi = tmp[None, None]*np.cos(th_e)[None, ..., None]
    zi = tmp[None, None]*np.sin(-th_e)[None, ..., None]

    wave_length = c/probe.center_frequency
    d_min = 0  # np.pi * (segment_length/2)**2/wave_length

    is_near = z_grid < d_min
    eps = np.finfo(float).eps
    dxi = x_grid[..., None, None] - xi - x_e[..., None]
    r = np.sqrt(dxi**2 + (z_grid[..., None, None]-zi-z_e[..., None])**2)
    theta = np.arcsin(dxi/(r + eps))-th_e[..., None]
    sin_theta = np.sin(theta)

    temporal_pulse_width = n_wavelengths/probe.center_frequency  # temporal pulse width T
    wc = 2*np.pi*probe.center_frequency
    angular_frequency_bandwidth = probe.bandwidth*wc/100
    p = np.log(126)/np.log(2*wc/angular_frequency_bandwidth)

    if frequency_step:
        df = frequency_step
    else:
        df = 1/(np.max(r)/c + np.max(transmit_delays))

    n_freq_samples = 2 * np.ceil(probe.center_frequency / df) + 1
    freq, df = np.linspace(0, 2*probe.center_frequency, int(n_freq_samples), retstep=True)

    def virtual_transducer_spectrum(w):
        vtx = np.exp(-(np.abs(w-wc)*200*(-np.log(0.5))**(1/p)/probe.bandwidth/wc)**p)
        return np.sqrt(vtx).astype(np.float32)

    def transmit_spectrum(w):
        sinc = lambda x: np.sinc(x/np.pi)
        tx_spectrum = 1j*(sinc(temporal_pulse_width*(w-wc)/2)-sinc(temporal_pulse_width*(w+wc)/2))
        return tx_spectrum.astype(np.complex64)

    tmp_hf = (transmit_spectrum(2*np.pi*freq)*virtual_transducer_spectrum(2*np.pi*freq)).astype(np.complex64)
    max_hf = np.max(np.abs(tmp_hf))
    g_db = 20*np.log10(np.abs(tmp_hf)/max_hf)
    idx = g_db > db_threshold
    freq = freq[idx]
    n_sampling = len(freq)
    alpha_dB = 0

    if return_rx_spectrum:
        rx_delay = np.zeros((1, n_elements))

    tx_spect = transmit_spectrum(2*np.pi*freq)  # transmitted spectrum
    transducer_wf = virtual_transducer_spectrum(2*np.pi*freq)  # transducer window function

    obliq_factor = np.cos(theta)
    obliq_factor[np.abs(theta) >= np.pi/2] = eps

    wave_number = 2*np.pi*freq[0]/c
    attenuation_based_wave_number = alpha_dB/8.7*freq[0]/1e6*1e2
    exp_func = np.exp(-attenuation_based_wave_number*r+1j*matlab_mod(wave_number*r, 2*np.pi))

    if y_grid is not None:
        exp_func = (exp_func*obliq_factor/r).astype(np.complex64)
        rm = np.mean(r, axis=-1)
    else:
        exp_func = (exp_func*obliq_factor/np.sqrt(r+eps)).astype(np.complex64)

    dir = np.pi * probe.center_frequency / c * segment_length * sin_theta
    dir = np.sinc(dir/np.pi).astype(np.float32)
    exp_func = exp_func*dir

    dkw = 2*np.pi*df/c
    dkwa = alpha_dB/8.7*df/1e6*1e2
    exp_func_df = np.exp((-dkwa+1j*dkw)*r).astype(np.complex64)

    spectrum = []

    if y_grid is not None:
        recalculate_mgbm_samples = np.round(np.linspace(0, n_sampling-1, max(3, round(n_sampling/20))))
        coeff_a = np.array([0.187 + 0.275j, 0.288 - 1.954j, *np.conj((0.187 + 0.275j, 0.288 - 1.954j))]).astype(np.complex64)
        coeff_b = np.array([4.558 - 25.59j, 8.598 - 7.924j, *np.conj((4.558 - 25.59j, 8.598 - 7.924j))]).astype(np.complex64)
        alpha = (1j/2*(1/probe.focus-1./rm)).astype(np.complex64)

    rp = np.zeros_like(x_grid)

    for k in range(n_sampling):
        kw = 2*np.pi*freq[k]/c

        if k > 0:
            exp_func *= exp_func_df

        if n_element_segments > 1:
            rp_mono = np.sum(exp_func, axis=-1)/n_element_segments
        else:
            rp_mono = exp_func[..., 0]

        if y_grid is not None:
            if k in recalculate_mgbm_samples:
                mgbm = np.sum(coeff_a*np.sqrt(np.pi/(kw*alpha[..., None] + coeff_b/probe.height**2)), axis=-1)
            rp_mono = rp_mono*mgbm

        tmp = np.exp(1j*kw*c*transmit_delays).astype(np.complex64)
        rpk = np.dot(rp_mono, apodization*tmp)
        rpk = tx_spect[k]*transducer_wf[k]*rpk
        rpk[is_near] = 0

        if return_rx_spectrum:
            spectrum.append(transducer_wf[k]*np.dot(rpk*rc, rp_mono))
        else:
            rp += np.abs(rpk)**2

    correction_factor = df*probe.width
    if y_grid is not None:
        correction_factor = df

    if return_rx_spectrum:
        return rp*correction_factor, np.array(spectrum)*correction_factor, idx
    return np.sqrt(rp*correction_factor)
