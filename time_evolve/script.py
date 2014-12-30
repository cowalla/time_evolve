import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tools import (
    create_gaussian,
    discretize_continuous_function,
    hamiltonian,
    construct_time_evolve_hamiltonian,
    time_evolve,
)

SIZE = 200
TIME_STEPS = 100
X_POSITIONS = [float(i)/float(SIZE) for i in range(1, SIZE + 1)]
PARAMS = {'a': 1, 'b': -275, 'c': 60, 're_offset': -0.5, 'im_offset': -0.1}
POTENTIAL = lambda x: 0

initial_gaussian = discretize_continuous_function(create_gaussian(PARAMS), SIZE)
initial_hamiltonian = hamiltonian(size=SIZE, quality=1, potential=POTENTIAL)
evolve_hamiltonian = construct_time_evolve_hamiltonian(initial_hamiltonian)
gaussians = [initial_gaussian]

for time in range(TIME_STEPS):
    gaussians.append(time_evolve(evolve_hamiltonian, gaussians[-1]))


def update_line(num, data, line):
    print num
    line.set_data(X_POSITIONS, numpy.real(data[num]))
    return line

fig1 = plt.figure()

l = plt.plot([], [], 'r-')[0]
l.set_data(X_POSITIONS, gaussians[-1])
plt.xlim(0, 1)
plt.ylim(-5, 5)
plt.xlabel('x')
plt.title('y')

line_ani = animation.FuncAnimation(fig1, update_line, fargs=(gaussians, l),
    interval=50, blit=False)

plt.show()