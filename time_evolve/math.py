import numpy
from numpy import exp, zeros, pi, cos, imag


def create_gaussian(**kwargs):
    """
    Returns a gaussian wave of the form,
        a exp( b(re_offset + x)^2 + c(im_offset + x)i ).

    Requires parameter dictionary with `a`, `b`, `c`, `re_offset`, `im_offset`.
    """
    required_keys = ['a', 'b', 'c', 're_offset', 'im_offset']

    if not kwargs.keys() == required_keys:
        message = 'Please provide all required constants: {}'.format(
            str(required_keys)
        )
        raise Exception(message)

    a, b, c, re_offset, im_offset = (
        kwargs['a'],
        kwargs['b'],
        kwargs['c'],
        kwargs['re_offset'],
        kwargs['im_offset'],
    )

    return lambda x: (
        a * exp( b * (re_offset + x)^2 + c * complex(0, im_offset + x) )
    )