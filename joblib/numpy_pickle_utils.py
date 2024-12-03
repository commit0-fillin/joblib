"""Utilities for fast persistence of big data, with optional compression."""
import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
try:
    import numpy as np
except ImportError:
    np = None
Unpickler = pickle._Unpickler
Pickler = pickle._Pickler
xrange = range
try:
    import bz2
except ImportError:
    bz2 = None
_IO_BUFFER_SIZE = 1024 ** 2

def _is_raw_file(fileobj):
    """Check if fileobj is a raw file object, e.g created with open."""
    return isinstance(fileobj, (io.FileIO, io.BufferedReader, io.BufferedWriter))

def _is_numpy_array_byte_order_mismatch(array):
    """Check if numpy array is having byte order mismatch"""
    if np is None:
        return False
    return (array.dtype.byteorder == '>' and sys.byteorder == 'little') or \
           (array.dtype.byteorder == '<' and sys.byteorder == 'big')

def _ensure_native_byte_order(array):
    """Use the byte order of the host while preserving values

    Does nothing if array already uses the system byte order.
    """
    if _is_numpy_array_byte_order_mismatch(array):
        return array.byteswap().newbyteorder()
    return array

def _detect_compressor(fileobj):
    """Return the compressor matching fileobj.

    Parameters
    ----------
    fileobj: file object

    Returns
    -------
    str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat', 'not-compressed'}
    """
    if isinstance(fileobj, io.BytesIO):
        # BytesIO object: we need to look at the first bytes
        first_bytes = fileobj.getvalue()[:4]
    else:
        # Regular file object: we need to read the first 4 bytes
        first_bytes = fileobj.read(4)
        fileobj.seek(0)

    if first_bytes.startswith(_ZLIB_PREFIX):
        return 'zlib'
    elif first_bytes.startswith(_GZIP_PREFIX):
        return 'gzip'
    elif first_bytes.startswith(_BZ2_PREFIX):
        return 'bz2'
    elif first_bytes.startswith(_LZMA_PREFIX):
        return 'lzma'
    elif first_bytes.startswith(_XZ_PREFIX):
        return 'xz'
    elif first_bytes.startswith(_ZFILE_PREFIX):
        return 'compat'
    else:
        return 'not-compressed'

def _buffered_read_file(fobj):
    """Return a buffered version of a read file object."""
    if isinstance(fobj, io.BufferedReader):
        return fobj
    return io.BufferedReader(io.FileIO(fobj.fileno(), 'rb'))

def _buffered_write_file(fobj):
    """Return a buffered version of a write file object."""
    if isinstance(fobj, io.BufferedWriter):
        return fobj
    return io.BufferedWriter(io.FileIO(fobj.fileno(), 'wb'))

@contextlib.contextmanager
def _read_fileobject(fileobj, filename, mmap_mode=None):
    """Utility function opening the right fileobject from a filename.

    The magic number is used to choose between the type of file object to open:
    * regular file object (default)
    * zlib file object
    * gzip file object
    * bz2 file object
    * lzma file object (for xz and lzma compressor)

    Parameters
    ----------
    fileobj: file object
    compressor: str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat',
                        'not-compressed'}
    filename: str
        filename path corresponding to the fileobj parameter.
    mmap_mode: str
        memory map mode that should be used to open the pickle file. This
        parameter is useful to verify that the user is not trying to one with
        compression. Default: None.

    Returns
    -------
        a file like object

    """
    compressor = _detect_compressor(fileobj)
    
    if compressor == 'not-compressed':
        if mmap_mode is not None:
            fileobj = _buffered_read_file(fileobj)
            yield np.memmap(fileobj, mode=mmap_mode)
        else:
            yield fileobj
    else:
        if mmap_mode is not None:
            warnings.warn('File "%(filename)s" is compressed using '
                          '"%(compressor)s" which is not compatible with '
                          'mmap_mode "%(mmap_mode)s" flag passed.'
                          % locals(), Warning)
        
        if compressor == 'zlib':
            yield _COMPRESSORS['zlib'].decompressor_file(fileobj)
        elif compressor == 'gzip':
            yield _COMPRESSORS['gzip'].decompressor_file(fileobj)
        elif compressor == 'bz2':
            if bz2 is None:
                raise ValueError('Trying to read a bz2 compressed file but '
                                 'bz2 module is not available.')
            yield _COMPRESSORS['bz2'].decompressor_file(fileobj)
        elif compressor in ('lzma', 'xz'):
            if lzma is None:
                raise ValueError('Trying to read a lzma compressed file but '
                                 'lzma module is not available.')
            yield _COMPRESSORS['lzma'].decompressor_file(fileobj)
        elif compressor == 'compat':
            # Compatibility with old versions of joblib
            fileobj.seek(0)
            length = int(fileobj.readline())
            yield io.BytesIO(zlib.decompress(fileobj.read(length)))

def _write_fileobject(filename, compress=('zlib', 3)):
    """Return the right compressor file object in write mode."""
    if compress is None or compress == 'not-compressed':
        return open(filename, 'wb')
    
    compressor, compresslevel = compress
    if compressor == 'gzip':
        return _COMPRESSORS['gzip'].compressor_file(filename, compresslevel)
    elif compressor == 'bz2':
        if bz2 is None:
            raise ValueError('Trying to compress using bz2 but '
                             'bz2 module is not available.')
        return _COMPRESSORS['bz2'].compressor_file(filename, compresslevel)
    elif compressor in ('lzma', 'xz'):
        if lzma is None:
            raise ValueError('Trying to compress using lzma but '
                             'lzma module is not available.')
        return _COMPRESSORS['lzma'].compressor_file(filename, compresslevel)
    elif compressor == 'zlib':
        return _COMPRESSORS['zlib'].compressor_file(filename, compresslevel)
    else:
        raise ValueError("Compression method not supported: %s" % compressor)
BUFFER_SIZE = 2 ** 18

def _read_bytes(fp, size, error_template='ran out of data'):
    """Read from file-like object until size bytes are read.

    TODO python2_drop: is it still needed? The docstring mentions python 2.6
    and it looks like this can be at least simplified ...

    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.

    This function was taken from numpy/lib/format.py in version 1.10.2.

    Parameters
    ----------
    fp: file-like object
    size: int
    error_template: str

    Returns
    -------
    a bytes object
        The data read in bytes.

    """
    data = bytes()
    while len(data) < size:
        chunk = fp.read(size - len(data))
        if not chunk:
            raise ValueError(error_template)
        data += chunk
    return data
