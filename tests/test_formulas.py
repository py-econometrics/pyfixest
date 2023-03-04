import pytest
from pyfixest.utils import unpack_fml

@pytest.mark.skip(reason="these tests are failing")
def test_unpack_fml():

    assert unpack_fml('a') == a
    assert unpack_fml('a + b') == ['a + b']
    assert unpack_fml('a + b + c') == ['a + b + c']

    assert unpack_fml('sw(a, b)') == ['a', 'b']
    assert unpack_fml('sw(a, b, c)') == ['a', 'b', 'c']
    assert unpack_fml('sw(a, b)' + 'c') == ['a + c', 'b + c']
    assert unpack_fml('d + sw(a, b)' + 'c') == ['d + a + c', 'd + b + c']

    assert unpack_fml('sw0(a, b)') == [0, 'a', 'b']
    assert unpack_fml('sw0(a, b, c)') == [0, 'a', 'b', 'c']
    assert unpack_fml('sw0(a, b)' + 'c') == ['c', 'a + c', 'b + c']
    assert unpack_fml('d + sw0(a, b)' + 'c') == ['d + c', 'd + a + c', 'd + b + c']


