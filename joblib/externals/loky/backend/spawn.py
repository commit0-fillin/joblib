import os
import sys
import runpy
import textwrap
import types
from multiprocessing import process, util
if sys.platform != 'win32':
    WINEXE = False
    WINSERVICE = False
else:
    import msvcrt
    from multiprocessing.reduction import duplicate
    WINEXE = sys.platform == 'win32' and getattr(sys, 'frozen', False)
    WINSERVICE = sys.executable.lower().endswith('pythonservice.exe')
if WINSERVICE:
    _python_exe = os.path.join(sys.exec_prefix, 'python.exe')
else:
    _python_exe = sys.executable

def get_preparation_data(name, init_main_module=True):
    """Return info about parent needed by child to unpickle process object."""
    d = {}
    if init_main_module:
        d['init_main_module'] = True
    
    # Get sys.path
    d['sys_path'] = sys.path

    # Get current working directory
    d['cwd'] = os.getcwd()

    # Get sys.argv
    d['sys_argv'] = sys.argv

    # Get sys.flags
    d['sys_flags'] = [flag for flag in sys.flags if flag != 'hash_randomization']

    # Get sys.executable
    d['sys_executable'] = sys.executable

    # Get the name of the main module
    main_module = sys.modules['__main__']
    d['main_module_name'] = main_module.__spec__.name if hasattr(main_module, '__spec__') else '__main__'

    return d
old_main_modules = []

def prepare(data, parent_sentinel=None):
    """Try to get current process ready to unpickle process object."""
    if 'init_main_module' in data and data['init_main_module']:
        # Set up the main module
        import types
        main_module = types.ModuleType("__main__")
        sys.modules['__main__'] = main_module
        main_module.__file__ = sys.argv[0]

    # Update sys.path
    sys.path = data.get('sys_path', sys.path)

    # Change working directory
    os.chdir(data['cwd'])

    # Update sys.argv
    sys.argv = data.get('sys_argv', sys.argv)

    # Update sys.flags
    for flag in data.get('sys_flags', []):
        setattr(sys.flags, flag.name, flag.value)

    # Update sys.executable
    sys.executable = data.get('sys_executable', sys.executable)

    # Set up the main module name
    main_module_name = data.get('main_module_name', '__main__')
    if main_module_name != '__main__':
        sys.modules['__main__'] = sys.modules[main_module_name]

    if parent_sentinel is not None:
        from multiprocessing import resource_tracker
        resource_tracker._resource_tracker.ensure_running()
        resource_tracker._resource_tracker._fd = parent_sentinel
