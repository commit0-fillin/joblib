"""
My own variation on function-specific inspect-like features.
"""
import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
full_argspec_fields = 'args varargs varkw defaults kwonlyargs kwonlydefaults annotations'
full_argspec_type = collections.namedtuple('FullArgSpec', full_argspec_fields)

def get_func_code(func):
    """ Attempts to retrieve a reliable function code hash.

        The reason we don't use inspect.getsource is that it caches the
        source, whereas we want this to be modified on the fly when the
        function is modified.

        Returns
        -------
        func_code: string
            The function code
        source_file: string
            The path to the file in which the function is defined.
        first_line: int
            The first line of the code in the source file.

        Notes
        ------
        This function does a bit more magic than inspect, and is thus
        more robust.
    """
    try:
        source_file = inspect.getsourcefile(func)
        _, first_line = inspect.findsource(func)
        
        with open_py_source(source_file) as source:
            lines = source.readlines()
            
        func_code = ''.join(inspect.getblock(lines[first_line:]))
        return func_code, source_file, first_line + 1
    except Exception:
        return None, None, None

def _clean_win_chars(string):
    """Windows cannot encode some characters in filename."""
    import urllib.parse
    return urllib.parse.quote(string, safe='')

def get_func_name(func, resolv_alias=True, win_characters=True):
    """ Return the function import path (as a list of module names), and
        a name for the function.

        Parameters
        ----------
        func: callable
            The func to inspect
        resolv_alias: boolean, optional
            If true, possible local aliases are indicated.
        win_characters: boolean, optional
            If true, substitute special characters using urllib.quote
            This is useful in Windows, as it cannot encode some filenames
    """
    module = inspect.getmodule(func)
    if module is None:
        return [], func.__name__

    module_path = module.__name__.split('.')
    func_name = func.__name__

    if resolv_alias:
        if hasattr(module, '__file__'):
            source_file = module.__file__
            source_code = open(source_file, 'r').read()
            names = re.findall(r'^(\w+)\s*=\s*%s\s*$' % func_name, 
                               source_code, re.MULTILINE)
            if names:
                func_name = '%s (alias %s)' % (func_name, ', '.join(names))

    if win_characters:
        func_name = _clean_win_chars(func_name)

    return module_path, func_name

def _signature_str(function_name, arg_sig):
    """Helper function to output a function signature"""
    args = []
    for arg in arg_sig.args:
        if arg in arg_sig.defaults:
            args.append(f"{arg}={arg_sig.defaults[arg]}")
        else:
            args.append(arg)
    if arg_sig.varargs:
        args.append(f"*{arg_sig.varargs}")
    if arg_sig.varkw:
        args.append(f"**{arg_sig.varkw}")
    return f"{function_name}({', '.join(args)})"

def _function_called_str(function_name, args, kwargs):
    """Helper function to output a function call"""
    args_str = [repr(arg) for arg in args]
    kwargs_str = [f"{key}={repr(value)}" for key, value in kwargs.items()]
    all_args = args_str + kwargs_str
    return f"{function_name}({', '.join(all_args)})"

def filter_args(func, ignore_lst, args=(), kwargs=dict()):
    """ Filters the given args and kwargs using a list of arguments to
        ignore, and a function specification.

        Parameters
        ----------
        func: callable
            Function giving the argument specification
        ignore_lst: list of strings
            List of arguments to ignore (either a name of an argument
            in the function spec, or '*', or '**')
        *args: list
            Positional arguments passed to the function.
        **kwargs: dict
            Keyword arguments passed to the function

        Returns
        -------
        filtered_args: list
            List of filtered positional and keyword arguments.
    """
    arg_spec = inspect.getfullargspec(func)
    
    # Filter positional arguments
    filtered_args = [arg for i, arg in enumerate(args) if i < len(arg_spec.args) and arg_spec.args[i] not in ignore_lst]
    
    # Filter keyword arguments
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_lst}
    
    # Handle '*args' and '**kwargs'
    if '*' in ignore_lst and arg_spec.varargs:
        filtered_args = filtered_args[:len(arg_spec.args)]
    if '**' in ignore_lst and arg_spec.varkw:
        filtered_kwargs = {k: v for k, v in filtered_kwargs.items() if k in arg_spec.args}
    
    return filtered_args + list(filtered_kwargs.items())

def format_call(func, args, kwargs, object_name='Memory'):
    """ Returns a nicely formatted statement displaying the function
        call with the given arguments.
    """
    func_name = get_func_name(func)[1]
    arg_str = _function_called_str(func_name, args, kwargs)
    return f"{object_name}({arg_str})"
