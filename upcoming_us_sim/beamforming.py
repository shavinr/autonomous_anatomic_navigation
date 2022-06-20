import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix


def delay_and_sum(data, xi, zi, transmit_delays, probe):
    n_lines, n_channels = data.shape[0], data.shape[1]
    isIQ = np.iscomplex(data).all()
    das_mat = das_matrix(np.dot(~isIQ*1+1j*isIQ, [n_lines, n_channels]), xi, zi, transmit_delays, probe)
    return np.reshape(das_mat*data.ravel(order="F"), xi.shape)


def das_matrix(data, x, z, transmit_delays, probe, c=1540, sampling_frequency=10.8e6):
    x_e, z_e = probe.get_tranducer_element_centers()

    if data.shape == (2,) or data.shape == (3,):
        n_lines = np.abs(data[0]).astype(np.uint32)
        n_channels = np.abs(data[1]).astype(np.uint32)
    else:
        n_lines, n_channels = data.shape[0], data.shape[1]

    idx = ~np.isnan(transmit_delays)
    n_transmitted_elements = np.size(np.nonzero(idx))

    idx_interp = np.linspace(0, n_transmitted_elements-1, 4*n_transmitted_elements)
    x_interp = interp1d(x=np.arange(n_transmitted_elements), y=x_e[idx])(idx_interp)
    z_interp = np.zeros_like(x_interp)

    transmit_delays_interp = interp1d(x=np.arange(n_transmitted_elements), y=transmit_delays[idx])(idx_interp)
    t0 = 0

    d_transmit = np.min((transmit_delays_interp[..., None]*c)[..., None]
                        +np.linalg.norm([x_interp[..., None, None]-x, z_interp[..., None, None]-z], axis=0), axis=0)
    d_x_transmit = x[..., None]-x_e
    d_z_transmit = z[..., None]-z_e
    d_receive = np.linalg.norm([d_x_transmit, d_z_transmit], axis=0)

    travel_time = (d_transmit[..., None]+d_receive)/c
    idx_fast_time = (travel_time-t0)*sampling_frequency
    in_range_indices = np.logical_and(idx_fast_time >= 0, idx_fast_time <= n_lines-1)

    idx = idx_fast_time + np.arange(0, n_channels)*n_lines
    idx = idx[in_range_indices]

    # Case linear
    idx_f = np.floor(idx).astype(np.int64)
    idx = idx_f-idx

    i = np.nonzero(in_range_indices.reshape(-1, in_range_indices.shape[-1]))[0]
    j = np.concatenate([idx_f, idx_f+1])
    s = np.concatenate([idx+1, -idx])

    if not np.isrealobj(data):
        travel_time = travel_time[in_range_indices]
        wc = 2 * np.pi * probe.center_frequency
        s = s*np.exp(1j*wc*np.concatenate([travel_time, travel_time]))

    sparse_matrix = csr_matrix((s, (np.concatenate([i, i]), j)), shape=(x.size, n_lines*n_channels))
    return sparse_matrix
