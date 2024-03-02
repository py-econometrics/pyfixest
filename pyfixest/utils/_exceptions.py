import inspect
import os


def find_stack_level() -> int:
    """Find the first place in the stack that is not inside pyfixest."""
    import pyfixest as pf

    pkg_dir = os.path.dirname(pf.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and not fname.startswith(test_dir):
            frame = frame.f_back
            n += 1
        else:
            break
    return n
