import numpy
from numpy import exp, zeros, power, linalg, dot


def create_gaussian(kwargs):
    """
    Returns a gaussian wave of the form,
        a exp( b(re_offset + x)^2 + c(im_offset + x)i ).

    Requires parameter dictionary with `a`, `b`, `c`, `re_offset`, `im_offset`.
    """
    a, b, c, re_offset, im_offset = (
        kwargs['leading_constant'],
        kwargs['real_proportion'],
        kwargs['complex_proportion'],
        kwargs['re_offset'],
        kwargs['im_offset'],
    )

    return lambda x: (
        a * exp(b * power(re_offset + x, 2) + c * complex(0, im_offset + x))
    )


def discretize_continuous_function(f, size):
    """
    :param f: function in question to be discretized over [0, 1]
    :param size: number of points in [0, 1]
    :return: [f(0), f(ds), ... , f(size - ds), f(size)]
    """
    ds = 1.0 / float(size)
    return [
        f(i * ds) for i in range(size)
    ]


def _construct_indeces(index, size):
    if index == 0:
        return size - 1, 1
    if index == size - 1:
        return index - 1, 0

    return index - 1, index + 1


def _identity_matrix(size):
    diagonals = [1 for i in range(size)]
    return numpy.diag(diagonals, 0)


def hamiltonian(size, quality, potential):
    """
    :param size: Size of square matrix
    :param quality: Quality; higher q => higher resolution: n * dq = 1/q
    :param potential: Continuous potential function
    :return: Numpy Hamiltonian matrix.
    """
    dn = float(1/(float(size) * float(quality)))
    _matrix = zeros(shape=(size, size))

    for i in range(0, size):
        minus, plus = _construct_indeces(i, size)
        _matrix[i, minus] = (-1/(dn * dn))
        _matrix[i, i] = (2 / (dn * dn)) + potential(i * dn)
        _matrix[i, plus] = (-1/(dn * dn))

    return _matrix


def construct_time_evolve_hamiltonian(
        init_hamiltonian, time_step=(float(1)/float(40000))
):
    """
    H is the time-evolution hamiltonian,
        H = [1 + i dt H]^-1.[1 - i dt H].
    """
    size = len(init_hamiltonian)
    return dot(
        linalg.inv(
            _identity_matrix(size) + 1j*(time_step * init_hamiltonian)
        ),
        _identity_matrix(size) - 1j*(time_step * init_hamiltonian)
    )


def time_evolve(time_hamiltonian, discrete_gaussian):
    """
    We want to evolve a wave form under influence of a Hamiltonian `h` using the
    Crank-Nicholson eq., such that,
        Psi(t+dt) = H . Psi(t).
    :param time_hamiltonian: Hamiltonian influencing gaussian
    :param discrete_gaussian: discrete gaussian with same size as Hamiltonian
    :return: the discrete gaussian time-evolved one step
    """
    return dot(time_hamiltonian, discrete_gaussian)


def normalize_gaussian(gaussian):
    inverse_norm = (1.0 / numpy.linalg.norm(gaussian))
    normalized_gaussian = []
    for component in gaussian:
        normalized_gaussian.append(
            inverse_norm * power(
                dot(component, numpy.conjugate(component)), 0.5
            )
        )

    return numpy.real(normalized_gaussian)







