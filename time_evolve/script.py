import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tools import (
    create_gaussian,
    discretize_continuous_function,
    hamiltonian,
    construct_time_evolve_hamiltonian,
    time_evolve,
    normalize_gaussian as norm,
)

X_STEPS = 100
TIME_STEPS = 300
X_SIZE = 1
X_POSITIONS = [X_SIZE * float(i)/float(X_STEPS) for i in range(1, X_STEPS + 1)]
# a exp( b(re_offset + x)^2 + c(im_offset + x)i ).
PARAMS = {
    'leading_constant': 1.0,
    'real_proportion': -2000.0,
    'complex_proportion': -3000.0,
    're_offset': -0.3,
    'im_offset': -0.3,
}
POTENTIAL_STRENGTH = 30000


def potential(x):
    if x < 0.6:
        return 0
    elif x < 0.61:
        return POTENTIAL_STRENGTH
    else:
        return 0

initial_gaussian = discretize_continuous_function(
    create_gaussian(PARAMS), X_STEPS
)
initial_hamiltonian = hamiltonian(
    size=X_STEPS, quality=1.0, potential=potential
)
evolve_hamiltonian = construct_time_evolve_hamiltonian(initial_hamiltonian)
gaussians = [initial_gaussian]

for time in range(TIME_STEPS):
    gaussians.append(time_evolve(evolve_hamiltonian, gaussians[-1]))

display_gaussians = []
for time in range(TIME_STEPS):
    display_gaussians.append(norm(gaussians[time]))


def update_line(num, data, line):
    if num < TIME_STEPS:
        line.set_data(X_POSITIONS, numpy.real(data[num]))
    return line

fig1 = plt.figure()

l = plt.plot([], [], 'r-')[0]
plt.xlim(0, X_SIZE)
plt.ylim(-5, 5)
plt.xlabel('x')
plt.title('y')

line_ani = animation.FuncAnimation(
    fig1, update_line, fargs=(display_gaussians, l), interval=1, blit=False
)

plt.show()