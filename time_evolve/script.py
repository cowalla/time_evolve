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

LEADING_CONSTANT = 1.0
REAL_CONSTANT = -2000.0
COMPLEX_CONSTANT = -3000.0
RE_OFFSET = -0.3
IM_OFFSET = -0.3

POTENTIAL_LOCATION = 0.6
POTENTIAL_STRENGTH = 30000

gaussian_fn = create_gaussian(
    a=LEADING_CONSTANT,
    b=REAL_CONSTANT,
    c=COMPLEX_CONSTANT,
    re_offset=RE_OFFSET,
    im_offset=IM_OFFSET
)

def wall_potential(x):
    if x < POTENTIAL_LOCATION:
        return 0
    elif x < POTENTIAL_LOCATION + 0.01:
        return POTENTIAL_STRENGTH
    else:
        return 0

initial_gaussian = discretize_continuous_function(gaussian_fn, X_STEPS)
initial_hamiltonian = hamiltonian(size=X_STEPS, quality=1.0, potential=wall_potential)
evolve_hamiltonian = construct_time_evolve_hamiltonian(initial_hamiltonian)
gaussians = [initial_gaussian]

for time in range(TIME_STEPS):
    gaussians.append(time_evolve(evolve_hamiltonian, gaussians[-1]))

display_gaussians = []

for time in range(TIME_STEPS):
    normalized_gaussian = norm(gaussians[time])
    display_gaussians.append(normalized_gaussian)

# make figure

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
    fig1, update_line, fargs=(display_gaussians, l), interval=10, blit=False
)

# TODO: Fix closing the animation
try:
    plt.show()
except KeyboardInterrupt:
    print 'closing!'
    plt.close()
    pass