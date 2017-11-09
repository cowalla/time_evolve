from numpy import exp, zeros, real, power, linalg, dot, conjugate, diag as diagonal_matrix


def _construct_indices(index, size):
    if index == 0:
        return size - 1, 1
    if index == size - 1:
        return index - 1, 0

    return index - 1, index + 1

def _identity_matrix(size):
    """
    Matrix with 1s along the diagonal and 0s everywhere else.
    """
    return diagonal_matrix([1 for i in range(size)], 0)

def create_gaussian(a, b, c, re_offset, im_offset):
    """
    Returns a gaussian wave of the form,
        A exp( B(re_offset + x)^2 + C(im_offset + x)i ).

    Requires parameter dictionary with `a`, `b`, `c`, `re_offset`, `im_offset`.
    """
    return lambda x: (
        a * exp(b * power(re_offset + x, 2) + c * complex(0, im_offset + x))
    )


def discretize_continuous_function(f, size):
    """
    returns [f(0), f(ds), ... , f(size - ds), f(size)]
    """
    ds = 1.0 / float(size)

    return [f(i * ds) for i in range(size)]


def hamiltonian(size, quality, potential):
    """
    returns the Hamiltonian matrix for the system

    :param size: Size of square matrix
    :param quality: Quality; higher q => higher resolution: n * dq = 1/q
    :param potential: Continuous potential function
    """
    dn = float(1/(float(size) * float(quality)))
    _matrix = zeros(shape=(size, size))

    for i in range(0, size):
        minus, plus = _construct_indices(i, size)
        _matrix[i, minus] = (-1/(dn * dn))
        _matrix[i, i] = (2 / (dn * dn)) + potential(i * dn)
        _matrix[i, plus] = (-1/(dn * dn))

    return _matrix


def construct_time_evolve_hamiltonian(init_hamiltonian, time_step=None):
    """
    returns the time-evolution hamiltonian,
        H = [1 + i dt H]^-1.[1 - i dt H].
    """
    if time_step is None:
        time_step = float(1) / float(40000)

    size = len(init_hamiltonian)

    return dot(
        linalg.inv(_identity_matrix(size) + 1j*(time_step * init_hamiltonian)),
        _identity_matrix(size) - 1j*(time_step * init_hamiltonian)
    )


def time_evolve(time_hamiltonian, discrete_gaussian):
    """
    returns the discrete gaussian time-evolved one step

    We want to evolve a wave form under influence of a Hamiltonian `h` using the
    Crank-Nicholson eq., such that,
        Psi(t+dt) = H . Psi(t).
    """
    return dot(time_hamiltonian, discrete_gaussian)


def normalize_gaussian(gaussian):
    inverse_norm = (1.0 / linalg.norm(gaussian))
    normalized_gaussian = [
        inverse_norm * power(dot(component, conjugate(component)), 0.5)
        for component in gaussian
    ]

    return real(normalized_gaussian)
