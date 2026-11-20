"""Wall-clock budgets for regression tests, separate from timeout-contract tests."""
import platform


def regression_timeout(seconds, *, riscv_seconds=None):
    # The SG2044 Python interpreter can take several minutes on the same
    # full-size IR fixture that completes in seconds on a development host.
    if platform.machine() != 'riscv64':
        return seconds
    return riscv_seconds if riscv_seconds is not None else seconds * 8
