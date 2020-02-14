import numpy as np
from numpy.random import randint, rand
import itertools

def new_spin(size):
    return 2*( randint(0, 2, size=(size, size)) - 1/2)



def ising2d(spin, size, coupling=1, external=None):
    hamiltonian = 0.0
    x, y = size
    for i,j in itertools.product(range(-1, x-1), range(-1, y-1)):
        hamiltonian += -coupling * spin[i, j] * (spin[i, j+1] + spin[i+1,j ]) # - external * spin[i,j]

    if external == None:
        return hamiltonian
    else:
        hamiltonian += -external * np.sum(spin) # TODO inner product is correct?
        return hamiltonian


def metropolis(spin, size, temperature, coupling=1, external=None):
    energy = ising2d(spin, size, coupling, external)
    fliped = spin_flip(spin, size)
    fliped_energy = ising2d(fliped, size, coupling, external)
    if fliped_energy - energy <= 0:
        spin = fliped
    else:
        spin = fliped if rejection_sampling(fliped_energy - energy, temperature) == 0 else spin

    return spin


def rejection_sampling(energy, temperature):
    r = rand()
    probability = np.exp(-energy/temperature) # Boltzman
    if probability >= r:
        return 0 # accept
    else:
        return 1 # reject

def spin_flip(spin, size):
    random_address = [randint(0, size[i]) for i in range(len(size))]
    i,j = random_address
    spin[i,j] = -spin[i,j]
    return spin


def heat_bath():
    pass


def tensor_group():
    pass

if __name__ == '__main__':
    size = 16
    update = 100
    t_list = np.linspace(1, 4)
    spin = new_spin(size)

    for t in t_list:
        for i in range(update):
            spin = metropolis(spin, [size, size], t,)
        magnetization = np.sum(spin)

        print(t, magnetization)
