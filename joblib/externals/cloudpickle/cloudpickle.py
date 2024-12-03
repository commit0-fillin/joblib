"""Pickler class to extend the standard pickle.Pickler functionality

The main objective is to make it natural to perform distributed computing on
clusters (such as PySpark, Dask, Ray...) with interactively defined code
(functions, classes, ...) written in notebooks or console.

In particular this pickler adds the following features:
- serialize interactively-defined or locally-defined functions, classes,
  enums, typevars, lambdas and nested functions to compiled byte code;
- deal with some other non-serializable objects in an ad-hoc manner where
  applicable.

This pickler is therefore meant to be used for the communication between short
lived Python processes running the same version of Python and libraries. In
particular, it is not meant to be used for long term storage of Python objects.

It does not include an unpickler, as standard Python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012-now, CloudPickle developers and contributors.
Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import _collections_abc
from collections import ChainMap, OrderedDict
import abc
import builtins
import copyreg
import dataclasses
import dis
from enum import Enum
import io
import itertools
import logging
import opcode
import pickle
from pickle import _getattribute
import platform
import struct
import sys
import threading
import types
import typing
import uuid
import warnings
import weakref
from types import CellType
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
_PICKLE_BY_VALUE_MODULES = set()
_DYNAMIC_CLASS_TRACKER_BY_CLASS = weakref.WeakKeyDictionary()
_DYNAMIC_CLASS_TRACKER_BY_ID = weakref.WeakValueDictionary()
_DYNAMIC_CLASS_TRACKER_LOCK = threading.Lock()
PYPY = platform.python_implementation() == 'PyPy'
builtin_code_type = None
if PYPY:
    builtin_code_type = type(float.__new__.__code__)
_extract_code_globals_cache = weakref.WeakKeyDictionary()

def register_pickle_by_value(module):
    """Register a module to make it functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
    global _PICKLE_BY_VALUE_MODULES
    _PICKLE_BY_VALUE_MODULES.add(module)

def unregister_pickle_by_value(module):
    """Unregister that the input module should be pickled by value."""
    global _PICKLE_BY_VALUE_MODULES
    _PICKLE_BY_VALUE_MODULES.discard(module)

def _whichmodule(obj, name):
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    module_name = getattr(obj, '__module__', None)
    if module_name is not None:
        return module_name
    for module_name, module in list(sys.modules.items()):
        if module_name == '__main__':
            continue
        try:
            if getattr(module, name, None) is obj:
                return module_name
        except Exception:
            pass
    return None

def _should_pickle_by_reference(obj, name=None):
    """Test whether an function or a class should be pickled by reference

    Pickling by reference means by that the object (typically a function or a
    class) is an attribute of a module that is assumed to be importable in the
    target Python environment. Loading will therefore rely on importing the
    module and then calling `getattr` on it to access the function or class.

    Pickling by reference is the only option to pickle functions and classes
    in the standard library. In cloudpickle the alternative option is to
    pickle by value (for instance for interactively or locally defined
    functions and classes or for attributes of modules that have been
    explicitly registered to be pickled by value.
    """
    if name is None:
        name = getattr(obj, '__qualname__', None)
    if name is None:
        name = getattr(obj, '__name__', None)
    
    module_name = _whichmodule(obj, name)
    if module_name is None:
        return False
    
    if module_name == "__main__":
        return False
    
    module = sys.modules.get(module_name, None)
    if module is None:
        return False
    
    if module in _PICKLE_BY_VALUE_MODULES:
        return False
    
    if name is None:
        return False
    
    attribute = getattr(module, name, None)
    return attribute is obj

def _extract_code_globals(co):
    """Find all globals names read or written to by codeblock co."""
    pass

def _find_imported_submodules(code, top_level_dependencies):
    """Find currently imported submodules used by a function.

    Submodules used by a function need to be detected and referenced for the
    function to work correctly at depickling time. Because submodules can be
    referenced as attribute of their parent package (``package.submodule``), we
    need a special introspection technique that does not rely on GLOBAL-related
    opcodes to find references of them in a code object.

    Example:
    ```
    import concurrent.futures
    import cloudpickle
    def func():
        x = concurrent.futures.ThreadPoolExecutor
    if __name__ == '__main__':
        cloudpickle.dumps(func)
    ```
    The globals extracted by cloudpickle in the function's state include the
    concurrent package, but not its submodule (here, concurrent.futures), which
    is the module used by func. Find_imported_submodules will detect the usage
    of concurrent.futures. Saving this module alongside with func will ensure
    that calling func once depickled does not fail due to concurrent.futures
    not being imported
    """
    pass
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG
_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k

def _walk_global_ops(code):
    """Yield referenced name for global-referencing instructions in code."""
    pass

def _extract_class_dict(cls):
    """Retrieve a copy of the dict of a class without the inherited method."""
    pass

def is_tornado_coroutine(func):
    """Return whether `func` is a Tornado coroutine function.

    Running coroutines are not supported.
    """
    pass

def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """
    pass

@instance
class _empty_cell_value:
    """Sentinel for empty closures."""

    @classmethod
    def __reduce__(cls):
        return cls.__name__

def _make_skeleton_class(type_constructor, name, bases, type_kwargs, class_tracker_id, extra):
    """Build dynamic class with an empty __dict__ to be filled once memoized

    If class_tracker_id is not None, try to lookup an existing class definition
    matching that id. If none is found, track a newly reconstructed class
    definition under that id so that other instances stemming from the same
    class id will also reuse this class definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    pass

def _make_skeleton_enum(bases, name, qualname, members, module, class_tracker_id, extra):
    """Build dynamic enum with an empty __dict__ to be filled once memoized

    The creation of the enum class is inspired by the code of
    EnumMeta._create_.

    If class_tracker_id is not None, try to lookup an existing enum definition
    matching that id. If none is found, track a newly reconstructed enum
    definition under that id so that other instances stemming from the same
    class id will also reuse this enum definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    pass

def _code_reduce(obj):
    """code object reducer."""
    pass

def _cell_reduce(obj):
    """Cell (containing values of a function's free variables) reducer."""
    pass

def _file_reduce(obj):
    """Save a file."""
    pass

def _dynamic_class_reduce(obj):
    """Save a class that can't be referenced as a module attribute.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from importable modules.
    """
    pass

def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj."""
    pass

def _function_setstate(obj, state):
    """Update the state of a dynamic function.

    As __closure__ and __globals__ are readonly attributes of a function, we
    cannot rely on the native setstate routine of pickle.load_build, that calls
    setattr on items of the slotstate. Instead, we have to modify them inplace.
    """
    pass
_DATACLASSE_FIELD_TYPE_SENTINELS = {dataclasses._FIELD.name: dataclasses._FIELD, dataclasses._FIELD_CLASSVAR.name: dataclasses._FIELD_CLASSVAR, dataclasses._FIELD_INITVAR.name: dataclasses._FIELD_INITVAR}

class Pickler(pickle.Pickler):
    _dispatch_table = {}
    _dispatch_table[classmethod] = _classmethod_reduce
    _dispatch_table[io.TextIOWrapper] = _file_reduce
    _dispatch_table[logging.Logger] = _logger_reduce
    _dispatch_table[logging.RootLogger] = _root_logger_reduce
    _dispatch_table[memoryview] = _memoryview_reduce
    _dispatch_table[property] = _property_reduce
    _dispatch_table[staticmethod] = _classmethod_reduce
    _dispatch_table[CellType] = _cell_reduce
    _dispatch_table[types.CodeType] = _code_reduce
    _dispatch_table[types.GetSetDescriptorType] = _getset_descriptor_reduce
    _dispatch_table[types.ModuleType] = _module_reduce
    _dispatch_table[types.MethodType] = _method_reduce
    _dispatch_table[types.MappingProxyType] = _mappingproxy_reduce
    _dispatch_table[weakref.WeakSet] = _weakset_reduce
    _dispatch_table[typing.TypeVar] = _typevar_reduce
    _dispatch_table[_collections_abc.dict_keys] = _dict_keys_reduce
    _dispatch_table[_collections_abc.dict_values] = _dict_values_reduce
    _dispatch_table[_collections_abc.dict_items] = _dict_items_reduce
    _dispatch_table[type(OrderedDict().keys())] = _odict_keys_reduce
    _dispatch_table[type(OrderedDict().values())] = _odict_values_reduce
    _dispatch_table[type(OrderedDict().items())] = _odict_items_reduce
    _dispatch_table[abc.abstractmethod] = _classmethod_reduce
    _dispatch_table[abc.abstractclassmethod] = _classmethod_reduce
    _dispatch_table[abc.abstractstaticmethod] = _classmethod_reduce
    _dispatch_table[abc.abstractproperty] = _property_reduce
    _dispatch_table[dataclasses._FIELD_BASE] = _dataclass_field_base_reduce
    dispatch_table = ChainMap(_dispatch_table, copyreg.dispatch_table)

    def _dynamic_function_reduce(self, func):
        """Reduce a function that is not pickleable via attribute lookup."""
        pass

    def _function_reduce(self, obj):
        """Reducer for function objects.

        If obj is a top-level attribute of a file-backed module, this reducer
        returns NotImplemented, making the cloudpickle.Pickler fall back to
        traditional pickle.Pickler routines to save obj. Otherwise, it reduces
        obj using a custom cloudpickle reducer designed specifically to handle
        dynamic functions.
        """
        pass

    def __init__(self, file, protocol=None, buffer_callback=None):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        super().__init__(file, protocol=protocol, buffer_callback=buffer_callback)
        self.globals_ref = {}
        self.proto = int(protocol)
    if not PYPY:
        dispatch = dispatch_table

        def reducer_override(self, obj):
            """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C `pickle.Pickler` class
            cannot register custom reducers for functions and classes in the
            dispatch_table attribute. Reducers for such types must instead
            implemented via the special `reducer_override` method.

            Note that this method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.

            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
            pass
    else:
        dispatch = pickle.Pickler.dispatch.copy()

        def save_global(self, obj, name=None, pack=struct.pack):
            """Main dispatch method.

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
            pass
        dispatch[type] = save_global

        def save_function(self, obj, name=None):
            """Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
            pass

        def save_pypy_builtin_func(self, obj):
            """Save pypy equivalent of builtin functions.

            PyPy does not have the concept of builtin-functions. Instead,
            builtin-functions are simple function instances, but with a
            builtin-code attribute.
            Most of the time, builtin functions should be pickled by attribute.
            But PyPy has flaky support for __qualname__, so some builtin
            functions such as float.__new__ will be classified as dynamic. For
            this reason only, we created this special routine. Because
            builtin-functions are not expected to have closure or globals,
            there is no additional hack (compared the one already implemented
            in pickle) to protect ourselves from reference cycles. A simple
            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note
            also that PyPy improved their support for __qualname__ in v3.6, so
            this routing should be removed when cloudpickle supports only PyPy
            3.6 and later.
            """
            pass
        dispatch[types.FunctionType] = save_function

def dump(obj, file, protocol=None, buffer_callback=None):
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
    speed between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python (although this is not always
    guaranteed to work because cloudpickle relies on some internal
    implementation details that can change from one Python version to the
    next).
    """
    pass

def dumps(obj, protocol=None, buffer_callback=None):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
    speed between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python (although this is not always
    guaranteed to work because cloudpickle relies on some internal
    implementation details that can change from one Python version to the
    next).
    """
    pass
load, loads = (pickle.load, pickle.loads)
CloudPickler = Pickler
