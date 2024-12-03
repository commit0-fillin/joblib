"""Utilities for fast persistence of big data, with optional compression."""
import pickle
import os
import warnings
import io
from pathlib import Path
from .compressor import lz4, LZ4_NOT_INSTALLED_ERROR
from .compressor import _COMPRESSORS, register_compressor, BinaryZlibFile
from .compressor import ZlibCompressorWrapper, GzipCompressorWrapper, BZ2CompressorWrapper, LZMACompressorWrapper, XZCompressorWrapper, LZ4CompressorWrapper
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_utils import _ensure_native_byte_order
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
from .numpy_pickle_compat import ZNDArrayWrapper
from .backports import make_memmap
register_compressor('zlib', ZlibCompressorWrapper())
register_compressor('gzip', GzipCompressorWrapper())
register_compressor('bz2', BZ2CompressorWrapper())
register_compressor('lzma', LZMACompressorWrapper())
register_compressor('xz', XZCompressorWrapper())
register_compressor('lz4', LZ4CompressorWrapper())
NUMPY_ARRAY_ALIGNMENT_BYTES = 16

class NumpyArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    This object is used to hack into the pickle machinery and read numpy
    array data from our custom persistence format.
    More precisely, this object is used for:
    * carrying the information of the persisted array: subclass, shape, order,
    dtype. Those ndarray metadata are used to correctly reconstruct the array
    with low level numpy functions.
    * determining if memmap is allowed on the array.
    * reading the array bytes from a file.
    * reading the array using memorymap from a file.
    * writing the array bytes to a file.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    shape: numpy.ndarray shape
        Determine the shape of the wrapped array.
    order: {'C', 'F'}
        Determine the order of wrapped array data. 'C' is for C order, 'F' is
        for fortran order.
    dtype: numpy.ndarray dtype
        Determine the data type of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
        Default: False.
    """

    def __init__(self, subclass, shape, order, dtype, allow_mmap=False, numpy_array_alignment_bytes=NUMPY_ARRAY_ALIGNMENT_BYTES):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.shape = shape
        self.order = order
        self.dtype = dtype
        self.allow_mmap = allow_mmap
        self.numpy_array_alignment_bytes = numpy_array_alignment_bytes

    def write_array(self, array, pickler):
        """Write array bytes to pickler file handle.

        This function is an adaptation of the numpy write_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        if array.dtype.hasobject:
            # We contain Python objects so we cannot represent
            # the data as bytes.
            pickler.save(array)
            return

        # Ensure contiguous data
        array = _ensure_native_byte_order(array)

        # Write the shape and dtype
        pickler.write(array.shape)
        pickler.write(array.dtype.str.encode('ascii'))

        # Write the data
        if isinstance(pickler.file_handle, io.BufferedIOBase):
            pickler.file_handle.write(array.tobytes())
        else:
            pickler.file_handle.write(array.tobytes('C'))

    def read_array(self, unpickler):
        """Read array from unpickler file handle.

        This function is an adaptation of the numpy read_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        shape = unpickler.read()
        dtype = np.dtype(unpickler.read().decode('ascii'))
        count = np.prod(shape)
        
        array = np.empty(count, dtype=dtype)
        data = _read_bytes(unpickler.file_handle, array.nbytes, "array data")
        array.data[:] = data

        array.shape = shape
        return array

    def read_mmap(self, unpickler):
        """Read an array using numpy memmap."""
        shape = unpickler.read()
        dtype = np.dtype(unpickler.read().decode('ascii'))
        count = np.prod(shape)

        # Get the file offset
        offset = unpickler.file_handle.tell()

        # Create a memmap array
        array = np.memmap(unpickler.filename, dtype=dtype, shape=shape,
                          mode=unpickler.mmap_mode, offset=offset)

        # Advance the file pointer
        unpickler.file_handle.seek(offset + array.nbytes)

        return array

    def read(self, unpickler):
        """Read the array corresponding to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler

        Returns
        -------
        array: numpy.ndarray

        """
        if unpickler.mmap_mode is not None and self.allow_mmap:
            array = self.read_mmap(unpickler)
        else:
            array = self.read_array(unpickler)

        if self.subclass is not np.ndarray:
            # We need to wrap the array in the specified subclass
            array = array.view(self.subclass)

        return array

class NumpyPickler(Pickler):
    """A pickler to persist big data efficiently.

    The main features of this object are:
    * persistence of numpy arrays in a single file.
    * optional compression with a special care on avoiding memory copies.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    protocol: int, optional
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL.
    """
    dispatch = Pickler.dispatch.copy()

    def __init__(self, fp, protocol=None):
        self.file_handle = fp
        self.buffered = isinstance(self.file_handle, BinaryZlibFile)
        if protocol is None:
            protocol = pickle.DEFAULT_PROTOCOL
        Pickler.__init__(self, self.file_handle, protocol=protocol)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array."""
        if (not hasattr(array, 'dtype') or
                array.dtype.hasobject or
                self.np is None or
                not isinstance(array, self.np.ndarray)):
            # This is not a numpy array or it cannot be handled by this
            # wrapper.
            return array
        if array.dtype.kind == 'V':
            # Avoid errors with saving or memmapping arrays with
            # dtype.kind == 'V' and dtype.metadata
            array = array.view(np.dtype((array.dtype.type, array.dtype.shape)))
        return NumpyArrayWrapper(
            subclass=array.__class__,
            shape=array.shape,
            dtype=array.dtype,
            order='F' if array.flags.f_contiguous else 'C',
            allow_mmap=True,
            numpy_array_alignment_bytes=self.numpy_array_alignment_bytes
        )

    def save(self, obj):
        """Subclass the Pickler `save` method.

        This is a total abuse of the Pickler class in order to use the numpy
        persistence function `save` instead of the default pickle
        implementation. The numpy array is replaced by a custom wrapper in the
        pickle persistence stack and the serialized array is written right
        after in the file. Warning: the file produced does not follow the
        pickle format. As such it can not be read with `pickle.load`.
        """
        if self.np is not None:
            if isinstance(obj, self.np.ndarray):
                wrapper = self._create_array_wrapper(obj)
                if wrapper is not obj:
                    # This is a numpy array that we can handle
                    Pickler.save(self, wrapper)
                    self.write_array(obj, self)
                    return
        Pickler.save(self, obj)

class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.

    Attributes
    ----------
    mmap_mode: str
        The memorymap mode to use for reading numpy arrays.
    file_handle: file_like
        File object to unpickle from.
    filename: str
        Name of the file to unpickle from. It should correspond to file_handle.
        This parameter is required when using mmap_mode.
    np: module
        Reference to numpy module if numpy is installed else None.

    """
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        self.filename = filename
        self.compat_mode = False
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects, NDArrayWrapper or
        NumpyArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        NDArrayWrapper is used for backward compatibility with joblib <= 0.9.
        """
        wrapper = self.stack[-1]
        if isinstance(wrapper, (NDArrayWrapper, NumpyArrayWrapper)):
            array = wrapper.read(self)
            self.stack[-1] = array
        else:
            Unpickler.load_build(self)
    dispatch[pickle.BUILD[0]] = load_build

def dump(value, filename, compress=0, protocol=None, cache_size=None):
    """Persist an arbitrary Python object into one file.

    Read more in the :ref:`User Guide <persistence>`.

    Parameters
    ----------
    value: any Python object
        The object to store to disk.
    filename: str, pathlib.Path, or file object.
        The file object or path of the file in which it is to be stored.
        The compression method corresponding to one of the supported filename
        extensions ('.z', '.gz', '.bz2', '.xz' or '.lzma') will be used
        automatically.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
        If compress is a 2-tuple, the first element must correspond to a string
        between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
        'xz'), the second element must be an integer from 0 to 9, corresponding
        to the compression level.
    protocol: int, optional
        Pickle protocol, see pickle.dump documentation for more details.
    cache_size: positive int, optional
        This option is deprecated in 0.10 and has no effect.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    joblib.load : corresponding loader

    Notes
    -----
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take up extra memory during
    dump and load.

    """
    if cache_size is not None:
        warnings.warn("The 'cache_size' parameter has been deprecated in "
                      "joblib 0.12 and will be ignored. It has no effect.",
                      DeprecationWarning, stacklevel=2)

    if compress is True:
        compress = ('zlib', 3)
    elif compress is False:
        compress = None
    elif isinstance(compress, int):
        compress = ('zlib', compress)
    elif isinstance(compress, tuple) and len(compress) != 2:
        raise ValueError(
            'Compress argument tuple should have exactly two elements: '
            'compress method as a string, and compress level as an integer.'
        )

    if not isinstance(filename, (str, Path)):
        file_handle = filename
        filename = getattr(file_handle, 'name', '')
    else:
        file_handle = None

    try:
        if compress is None:
            file_handle = _write_fileobject(filename, compress=compress)
        elif isinstance(compress, tuple):
            file_handle = _write_fileobject(filename, compress=compress)
        else:
            raise ValueError("Compress argument should be either None, "
                             "a boolean, an integer, or a tuple.")

        NumpyPickler(file_handle, protocol=protocol).dump(value)

    finally:
        if file_handle is not None:
            file_handle.close()

    return [filename]

def _unpickle(fobj, filename='', mmap_mode=None):
    """Internal unpickling function."""
    try:
        with _read_fileobject(fobj, filename, mmap_mode) as fobj:
            unpickler = NumpyUnpickler(filename, fobj, mmap_mode=mmap_mode)
            obj = unpickler.load()
    except UnicodeDecodeError as e:
        # More user-friendly error message
        new_exc = ValueError(
            'You may be trying to read with '
            'python 3 a joblib pickle generated with python 2. '
            'Try to use joblib.load(\'filename\', \'latin1\') '
            'instead of joblib.load(\'filename\').'
        )
        raise new_exc from e
    return obj

def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Read more in the :ref:`User Guide <persistence>`.

    WARNING: joblib.load relies on the pickle module and can therefore
    execute arbitrary Python code. It should therefore never be used
    to load files from untrusted sources.

    Parameters
    ----------
    filename: str, pathlib.Path, or file object.
        The file object or path of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
        case the reconstructed object might no longer match exactly
        the originally pickled object.

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmapped.
    """
    if isinstance(filename, (str, Path)):
        with open(filename, 'rb') as f:
            return _unpickle(f, filename, mmap_mode)
    elif hasattr(filename, 'read'):
        return _unpickle(filename, getattr(filename, 'name', ''), mmap_mode)
    else:
        raise TypeError("Expected a string, pathlib.Path or file object. "
                        "Got {0} instead.".format(type(filename)))
