import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tools import (
    create_gaussian,
    discretize_continuous_function,
    hamiltionian,
    construct_time_evolve_hamiltonian,
    time_evolve,
)

SIZE = 200
TIME_STEPS = 1000
X_POSITIONS = [float(i)/float(SIZE) for i in range(SIZE)]
PARAMS = {'a': 2.5, 'b': -275, 'c': 65, 're_offset': -0.1, 'im_offset': -0.1}
POTENTIAL = lambda x: 0

initial_gaussian = discretize_continuous_function(create_gaussian(PARAMS), SIZE)
initial_hamiltonian = hamiltionian(size=SIZE, quality=1, potential=POTENTIAL)
evolve_hamiltonian = construct_time_evolve_hamiltonian(initial_hamiltonian)
gaussians = [initial_gaussian]

for time in range(TIME_STEPS):
    gaussians.append(time_evolve(evolve_hamiltonian, gaussians[-1]))

plt.figure()
plt.plot(X_POSITIONS, gaussians[-1])
plt.show()