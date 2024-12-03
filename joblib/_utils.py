import ast
from dataclasses import dataclass
import operator as op
from ._multiprocessing_helpers import mp
if mp is not None:
    from .externals.loky.process_executor import _ExceptionWithTraceback
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg}

def eval_expr(expr):
    """
    >>> eval_expr('2*6')
    12
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4) / (6 + -7)')
    -161.0
    """
    return eval(expr, {"__builtins__": None}, operators)

@dataclass(frozen=True)
class _Sentinel:
    """A sentinel to mark a parameter as not explicitly set"""
    default_value: object

    def __repr__(self):
        return f'default({self.default_value!r})'

class _TracebackCapturingWrapper:
    """Protect function call and return error with traceback."""

    def __init__(self, func):
        self.func = func

    def __call__(self, **kwargs):
        try:
            return self.func(**kwargs)
        except BaseException as e:
            return _ExceptionWithTraceback(e)
