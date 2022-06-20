import numpy as np


class Probe(object):
    def __init__(self, center_frequency=None, kerf=None, width=None, pitch=None, n_elements=None, bandwidth=None,
                 radius=None, preset=None, height=None, focus=None):
        self.center_frequency = center_frequency
        self.kerf = kerf
        self.width = width
        self.pitch = pitch
        self.n_elements = n_elements
        self.bandwidth = bandwidth
        self.radius = radius
        self.height = height
        self.focus = focus

        if preset == "P4-2v":
            self = self.__init__(center_frequency=2.7e6, kerf=5.0e-5, width=2.5e-4, pitch=3.0e-4, n_elements=64,
                                 bandwidth=74, radius=np.inf, focus=0.060, height=0.0140)
        elif preset == "L11-5v":
            self = self.__init__(center_frequency=7.6e6, kerf=3.0e-5, width=2.7e-4, pitch=3.0e-4, n_elements=128,
                                 bandwidth=77, radius=np.inf)

    def get_tranducer_element_centers(self):
        if self.is_linear():
            x_e = (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.pitch
            z_e = np.zeros((self.n_elements,))
        else:
            raise NotImplemented
        return x_e, z_e

    def is_linear(self):
        return np.isinf(self.radius)

    def __repr__(self):
        return str(self.__dict__)
