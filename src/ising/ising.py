import numpy as np
from numpy.random import randint, rand
import itertools
from operator import mul
from numba import jit


class Hamiltonian(object):
    """
    >>> from ising import Hamiltonian

    H = Hamiltonian()
    """

    # _energy = 0.0
    coupling = 1.0
    ext = 0.0

    def __init__(self, spin, *args, **kwargs):
        self.spin = spin
        self.shape = np.shape(spin)
        self.site = mul(*self.shape)

    @property
    def energy(self):
        self._energy = 0
        x, y = self.shape
        for i, j in itertools.product(range(-1, x - 1), range(-1, y - 1)):
            self._energy += (
                -self.coupling
                * self.spin[i, j]
                * (self.spin[i, j + 1] + self.spin[i + 1, j])
            )  # - external * spin[i,j]

        if self.ext is not None:
            self._energy += -self.ext * np.sum(self.spin)

        return self._energy

    def local_energy(self, pickup):
        i, j = pickup
        N, M = self.shape
        local = (
            self.coupling
            * self.spin[i, j]
            * (
                self.spin[i - 1, j]
                + self.spin[i, j - 1]
                + self.spin[(i + 1) % N, j]
                + self.spin[i, (j + 1) % M]
            )
            + self.ext * self.spin[i, j]
        )
        return 2 * local

    def partition_function(self, temperature):
        pass


    @classmethod
    def boltzmann_dist(cls, hamiltonian, temperature):
        probability = np.exp(-hamiltonian / temperature)
        return probability


class Ising(Hamiltonian):
    """
    >>> from ising import Ising
    >>> init_spin = Ising.initialize()
    >>> spin = Ising(init_spin)
    >>> spin.equibriate(100, method='metropolis')
    """

    is_equil = False

    def equilibriate(self, times, temperature, method):
        fun = METHODS[method]
        for _ in range(times):
            fun(self, temperature)

        self.is_equil = True

    @staticmethod
    @jit
    def metropolis(obj, temperature):
        current_spin = obj.spin
        current_energy = obj.energy

        random_address = []
        for key, value in enumerate(obj.shape):
            x = np.random.randint(0, value)
            random_address.append(x)

        i, j = random_address
        flipped_spin = np.copy(current_spin)
        flipped_spin[i, j] = -flipped_spin[i, j]
        flipped_energy = Hamiltonian(flipped_spin).energy

        r = np.random.rand()
        diff = flipped_energy - current_energy

        if r < Hamiltonian.boltzmann_dist(diff, temperature):
            obj.spin = flipped_spin

    @staticmethod
    def quick(obj, temperature):
        N, M = obj.shape
        for i, j in itertools.product(range(N), range(M)):
            diff = obj.local_energy([i, j])
            if diff <= 0 or (rand() < np.exp(-diff / temperature)):
                obj.energy += diff
                obj.spin[i, j] = -obj.spin[i, j]

    @staticmethod
    def heat_bath(self):
        pass

    @classmethod
    def initialize(cls, shape, lowT=False):
        if not isinstance(shape, (list, tuple, np.ndarray)):
            raise ValueError("{} type should be array_like.".format(shape))

        if not lowT:
            return 2 * randint(0, 2, size=shape) - 1

        else:
            return np.ones(shape)

    def magnetization(self, temperature, mcstep, method):
        if self.is_equil is False:
            raise ValueError('This system is not in equilibrium yet.')

        method = METHODS[method]
        sampling = []
        for _ in range(mcstep):
            method(self, temperature)
            mag = np.sum(self.spin)
            sampling.append(mag)

        return np.mean(sampling) / self.site

    def heat_capacity(self, temperature):
        pass

    @classmethod
    def onsager():
        pass


METHODS = {"metropolis": Ising.metropolis, "heat_bath": Ising.heat_bath}

if __name__ == "__main__":
    init_spin = Ising.initialize(shape=(4, 4))
    spin = Ising(init_spin)
    print(spin.spin)
    spin.equilibriate(10, 100, "metropolis")
    print(spin.spin)
