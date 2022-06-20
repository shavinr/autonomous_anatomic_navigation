import numpy as np

from simulus.utilities import matlab_mod


def transmit_delay(x0, z0, probe, c=1540):
    n_elements = probe.n_elements
    pitch = probe.pitch

    # physical transducer positions
    x = (np.arange(n_elements)-(n_elements-1)/2)*pitch
    z = np.zeros((n_elements, ))

    # delays
    delays = np.linalg.norm([x-x0, z-z0], axis=0)/c
    delays = -delays*np.sign(z0)
    delays = delays-np.min(delays)
    return delays


def transmit_delay_circ(tilt, width, probe, c=1540):
    n_elements = probe.n_elements
    pitch = probe.pitch
    probe_width = (n_elements-1)*pitch

    x = (np.arange(n_elements)-(n_elements-1)/2)*pitch
    z = np.zeros((n_elements, ))

    x0, z0 = angles_to_origin(probe_width, tilt, width)
    delays = np.linalg.norm([x-x0, z-z0], axis=0)/c
    delays = -delays*np.sign(z0)
    delays = delays-np.min(delays)
    return delays


def angles_to_origin(probe_width, tilt, width):
    tilt = matlab_mod(tilt+np.pi/2, 2*np.pi)-np.pi/2
    sign_corr = np.ones(tilt.shape)
    if abs(tilt) > np.pi/2:
        tilt = np.pi-tilt
        sign_corr = -1
    z0 = sign_corr*probe_width/(np.tan(tilt-width/2)-np.tan(tilt+width/2))
    x0 = sign_corr*z0*np.tan(width/2-tilt)+probe_width/2
    return x0, z0
