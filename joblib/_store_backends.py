"""Storage providers backends for Memory caching."""
from pickle import PicklingError
import re
import os
import os.path
import datetime
import json
import shutil
import time
import warnings
import collections
import operator
import threading
from abc import ABCMeta, abstractmethod
from .backports import concurrency_safe_rename
from .disk import mkdirp, memstr_to_bytes, rm_subdirs
from .logger import format_time
from . import numpy_pickle
CacheItemInfo = collections.namedtuple('CacheItemInfo', 'path size last_access')

class CacheWarning(Warning):
    """Warning to capture dump failures except for PicklingError."""
    pass

def concurrency_safe_write(object_to_write, filename, write_func):
    """Writes an object into a unique file in a concurrency-safe way."""
    temporary_filename = filename + '.tmp'
    try:
        with open(temporary_filename, 'wb') as f:
            write_func(object_to_write, f)
        concurrency_safe_rename(temporary_filename, filename)
    except:
        os.remove(temporary_filename)
        raise

class StoreBackendBase(metaclass=ABCMeta):
    """Helper Abstract Base Class which defines all methods that
       a StorageBackend must implement."""
    location = None

    @abstractmethod
    def _open_item(self, f, mode):
        """Opens an item on the store and return a file-like object.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        f: a file-like object
            The file-like object where an item is stored and retrieved
        mode: string, optional
            the mode in which the file-like object is opened allowed valued are
            'rb', 'wb'

        Returns
        -------
        a file-like object
        """
        return open(f, mode)

    @abstractmethod
    def _item_exists(self, location):
        """Checks if an item location exists in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        location: string
            The location of an item. On a filesystem, this corresponds to the
            absolute path, including the filename, of a file.

        Returns
        -------
        True if the item exists, False otherwise
        """
        return os.path.exists(location)

    @abstractmethod
    def _move_item(self, src, dst):
        """Moves an item from src to dst in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        src: string
            The source location of an item
        dst: string
            The destination location of an item
        """
        concurrency_safe_rename(src, dst)

    @abstractmethod
    def create_location(self, location):
        """Creates a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory.
        """
        os.makedirs(location, exist_ok=True)

    @abstractmethod
    def clear_location(self, location):
        """Clears a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory or a filename absolute path
        """
        if os.path.isdir(location):
            shutil.rmtree(location)
        elif os.path.isfile(location):
            os.remove(location)

    @abstractmethod
    def get_items(self):
        """Returns the whole list of items available in the store.

        Returns
        -------
        The list of items identified by their ids (e.g filename in a
        filesystem).
        """
        items = []
        for root, dirs, files in os.walk(self.location):
            for file in files:
                items.append(os.path.join(root, file))
        return items

    @abstractmethod
    def configure(self, location, verbose=0, backend_options=dict()):
        """Configures the store.

        Parameters
        ----------
        location: string
            The base location used by the store. On a filesystem, this
            corresponds to a directory.
        verbose: int
            The level of verbosity of the store
        backend_options: dict
            Contains a dictionary of named parameters used to configure the
            store backend.
        """
        self.location = location
        self.verbose = verbose
        self.backend_options = backend_options
        self.create_location(location)

class StoreBackendMixin(object):
    """Class providing all logic for managing the store in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    a private _open_item, _item_exists and _move_item methods. The _open_item
    method has to have the same signature as the builtin open and return a
    file-like object.
    """

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        """Load an item from the store given its id as a list of str."""
        pass

    def dump_item(self, call_id, item, verbose=1):
        """Dump an item in the store at the id given as a list of str."""
        pass

    def clear_item(self, call_id):
        """Clear the item at the id, given as a list of str."""
        pass

    def contains_item(self, call_id):
        """Check if there is an item at the id, given as a list of str."""
        pass

    def get_item_info(self, call_id):
        """Return information about item."""
        pass

    def get_metadata(self, call_id):
        """Return actual metadata of an item."""
        pass

    def store_metadata(self, call_id, metadata):
        """Store metadata of a computation."""
        pass

    def contains_path(self, call_id):
        """Check cached function is available in store."""
        pass

    def clear_path(self, call_id):
        """Clear all items with a common path in the store."""
        pass

    def store_cached_func_code(self, call_id, func_code=None):
        """Store the code of the cached function."""
        pass

    def get_cached_func_code(self, call_id):
        """Store the code of the cached function."""
        pass

    def get_cached_func_info(self, call_id):
        """Return information related to the cached function if it exists."""
        pass

    def clear(self):
        """Clear the whole store content."""
        pass

    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Remove the store's oldest files to enforce item, byte, and age limits.
        """
        pass

    def _get_items_to_delete(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Get items to delete to keep the store under size, file, & age limits.
        """
        pass

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        pass

    def __repr__(self):
        """Printable representation of the store location."""
        return '{class_name}(location="{location}")'.format(class_name=self.__class__.__name__, location=self.location)

class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""
    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    def clear_location(self, location):
        """Delete location on store."""
        pass

    def create_location(self, location):
        """Create object location on store"""
        pass

    def get_items(self):
        """Returns the whole list of items available in the store."""
        pass

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'
        """
        pass
