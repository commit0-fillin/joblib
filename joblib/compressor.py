"""Classes and functions for managing compressors."""
import io
import zlib
from joblib.backports import LooseVersion
try:
    from threading import RLock
except ImportError:
    from dummy_threading import RLock
try:
    import bz2
except ImportError:
    bz2 = None
try:
    import lz4
    from lz4.frame import LZ4FrameFile
except ImportError:
    lz4 = None
try:
    import lzma
except ImportError:
    lzma = None
LZ4_NOT_INSTALLED_ERROR = 'LZ4 is not installed. Install it with pip: https://python-lz4.readthedocs.io/'
_COMPRESSORS = {}
_ZFILE_PREFIX = b'ZF'
_ZLIB_PREFIX = b'x'
_GZIP_PREFIX = b'\x1f\x8b'
_BZ2_PREFIX = b'BZ'
_XZ_PREFIX = b'\xfd7zXZ'
_LZMA_PREFIX = b']\x00'
_LZ4_PREFIX = b'\x04"M\x18'

def register_compressor(compressor_name, compressor, force=False):
    """Register a new compressor.

    Parameters
    ----------
    compressor_name: str.
        The name of the compressor.
    compressor: CompressorWrapper
        An instance of a 'CompressorWrapper'.
    """
    global _COMPRESSORS
    if compressor_name in _COMPRESSORS and not force:
        raise ValueError(f"Compressor '{compressor_name}' already registered. "
                         "Use force=True to override.")
    _COMPRESSORS[compressor_name] = compressor

class CompressorWrapper:
    """A wrapper around a compressor file object.

    Attributes
    ----------
    obj: a file-like object
        The object must implement the buffer interface and will be used
        internally to compress/decompress the data.
    prefix: bytestring
        A bytestring corresponding to the magic number that identifies the
        file format associated to the compressor.
    extension: str
        The file extension used to automatically select this compressor during
        a dump to a file.
    """

    def __init__(self, obj, prefix=b'', extension=''):
        self.fileobj_factory = obj
        self.prefix = prefix
        self.extension = extension

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        if compresslevel is not None:
            return self.fileobj_factory(fileobj, 'wb', compresslevel=compresslevel)
        return self.fileobj_factory(fileobj, 'wb')

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        return self.fileobj_factory(fileobj, 'rb')

class BZ2CompressorWrapper(CompressorWrapper):
    prefix = _BZ2_PREFIX
    extension = '.bz2'

    def __init__(self):
        if bz2 is not None:
            self.fileobj_factory = bz2.BZ2File
        else:
            self.fileobj_factory = None

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        if self.fileobj_factory is None:
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)
        if compresslevel is not None:
            return self.fileobj_factory(fileobj, 'wb', compresslevel=compresslevel)
        return self.fileobj_factory(fileobj, 'wb')

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        if self.fileobj_factory is None:
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)
        return self.fileobj_factory(fileobj, 'rb')

class LZMACompressorWrapper(CompressorWrapper):
    prefix = _LZMA_PREFIX
    extension = '.lzma'
    _lzma_format_name = 'FORMAT_ALONE'

    def __init__(self):
        if lzma is not None:
            self.fileobj_factory = lzma.LZMAFile
            self._lzma_format = getattr(lzma, self._lzma_format_name)
        else:
            self.fileobj_factory = None

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        if self.fileobj_factory is None:
            raise ValueError("LZMA is not installed.")
        if compresslevel is not None:
            return self.fileobj_factory(fileobj, 'wb', format=self._lzma_format, preset=compresslevel)
        return self.fileobj_factory(fileobj, 'wb', format=self._lzma_format)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        if self.fileobj_factory is None:
            raise ValueError("LZMA is not installed.")
        return self.fileobj_factory(fileobj, 'rb', format=self._lzma_format)

class XZCompressorWrapper(LZMACompressorWrapper):
    prefix = _XZ_PREFIX
    extension = '.xz'
    _lzma_format_name = 'FORMAT_XZ'

class LZ4CompressorWrapper(CompressorWrapper):
    prefix = _LZ4_PREFIX
    extension = '.lz4'

    def __init__(self):
        if lz4 is not None:
            self.fileobj_factory = LZ4FrameFile
        else:
            self.fileobj_factory = None

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        if self.fileobj_factory is None:
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)
        if compresslevel is not None:
            return self.fileobj_factory(fileobj, 'wb', compression_level=compresslevel)
        return self.fileobj_factory(fileobj, 'wb')

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        if self.fileobj_factory is None:
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)
        return self.fileobj_factory(fileobj, 'rb')
_MODE_CLOSED = 0
_MODE_READ = 1
_MODE_READ_EOF = 2
_MODE_WRITE = 3
_BUFFER_SIZE = 8192

class BinaryZlibFile(io.BufferedIOBase):
    """A file object providing transparent zlib (de)compression.

    TODO python2_drop: is it still needed since we dropped Python 2 support A
    BinaryZlibFile can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that BinaryZlibFile provides only a *binary* file interface: data read
    is returned as bytes, and data to be written should be given as bytes.

    This object is an adaptation of the BZ2File object and is compatible with
    versions of python >= 2.7.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """
    wbits = zlib.MAX_WBITS

    def __init__(self, filename, mode='rb', compresslevel=3):
        self._lock = RLock()
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED
        self._pos = 0
        self._size = -1
        self.compresslevel = compresslevel
        if not isinstance(compresslevel, int) or not 1 <= compresslevel <= 9:
            raise ValueError("'compresslevel' must be an integer between 1 and 9. You provided 'compresslevel={}'".format(compresslevel))
        if mode == 'rb':
            self._mode = _MODE_READ
            self._decompressor = zlib.decompressobj(self.wbits)
            self._buffer = b''
            self._buffer_offset = 0
        elif mode == 'wb':
            self._mode = _MODE_WRITE
            self._compressor = zlib.compressobj(self.compresslevel, zlib.DEFLATED, self.wbits, zlib.DEF_MEM_LEVEL, 0)
        else:
            raise ValueError('Invalid mode: %r' % (mode,))
        if isinstance(filename, str):
            self._fp = io.open(filename, mode)
            self._closefp = True
        elif hasattr(filename, 'read') or hasattr(filename, 'write'):
            self._fp = filename
        else:
            raise TypeError('filename must be a str or bytes object, or a file')

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        with self._lock:
            if self._mode == _MODE_CLOSED:
                return
            try:
                if self._mode in (_MODE_READ, _MODE_READ_EOF):
                    self._decompressor = None
                elif self._mode == _MODE_WRITE:
                    self._fp.write(self._compressor.flush())
                    self._compressor = None
            finally:
                try:
                    if self._closefp:
                        self._fp.close()
                finally:
                    self._fp = None
                    self._closefp = False
                    self._mode = _MODE_CLOSED

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode in (_MODE_READ, _MODE_READ_EOF)

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b'' if the file is already at EOF.
        """
        with self._lock:
            self._check_can_read()
            if size == 0:
                return b""
            
            if self._mode == _MODE_READ_EOF or size < 0:
                return self._read_all()
            
            return self._read_limited(size)

    def readinto(self, b):
        """Read up to len(b) bytes into b.

        Returns the number of bytes read (0 for EOF).
        """
        with self._lock:
            self._check_can_read()
            data = self.read(len(b))
            n = len(data)
            b[:n] = data
            return n

    def write(self, data):
        """Write a byte string to the file.

        Returns the number of uncompressed bytes written, which is
        always len(data). Note that due to buffering, the file on disk
        may not reflect the data written until close() is called.
        """
        with self._lock:
            self._check_can_write()
            compressed = self._compressor.compress(data)
            self._fp.write(compressed)
            self._pos += len(data)
            return len(data)

    def seek(self, offset, whence=0):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Values for whence are:

            0: start of stream (default); offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        with self._lock:
            self._check_can_seek()
            
            if whence == 0:
                if offset < 0:
                    raise ValueError("Negative seek position {}".format(offset))
                return self._seek_forward(offset)
            elif whence == 1:
                return self._seek_forward(self._pos + offset)
            elif whence == 2:
                if offset > 0:
                    raise ValueError("Positive seek position {}".format(offset))
                return self._seek_backward(offset)
            else:
                raise ValueError("Invalid whence value")

    def tell(self):
        """Return the current file position."""
        self._check_not_closed()
        return self._pos

    def _check_not_closed(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def _check_can_read(self):
        if self._mode not in (_MODE_READ, _MODE_READ_EOF):
            raise io.UnsupportedOperation("File not open for reading")

    def _check_can_write(self):
        if self._mode != _MODE_WRITE:
            raise io.UnsupportedOperation("File not open for writing")

    def _check_can_seek(self):
        if self._mode not in (_MODE_READ, _MODE_READ_EOF):
            raise io.UnsupportedOperation("Seeking is only supported on files open for reading")

    def _read_all(self):
        chunks = []
        while True:
            chunk = self._fp.read(_BUFFER_SIZE)
            if not chunk:
                break
            decompressed = self._decompressor.decompress(chunk)
            if decompressed:
                chunks.append(decompressed)
        if self._decompressor.unused_data:
            self._fp.seek(-len(self._decompressor.unused_data), 1)
        self._mode = _MODE_READ_EOF
        return b"".join(chunks)

    def _read_limited(self, size):
        chunks = []
        while size > 0:
            chunk = self._fp.read(min(_BUFFER_SIZE, size))
            if not chunk:
                break
            decompressed = self._decompressor.decompress(chunk)
            if decompressed:
                chunks.append(decompressed)
                size -= len(decompressed)
        if self._decompressor.unused_data:
            self._fp.seek(-len(self._decompressor.unused_data), 1)
        return b"".join(chunks)

    def _seek_forward(self, offset):
        if offset < self._pos:
            raise ValueError("Negative seek in forward direction")
        self._pos = offset
        return self._pos

    def _seek_backward(self, offset):
        if offset > 0:
            raise ValueError("Positive seek in backward direction")
        self._fp.seek(0)
        self._decompressor = zlib.decompressobj(self.wbits)
        self._pos = 0
        while self._pos < offset:
            chunk = self._fp.read(min(_BUFFER_SIZE, offset - self._pos))
            if not chunk:
                break
            decompressed = self._decompressor.decompress(chunk)
            self._pos += len(decompressed)
        return self._pos

class ZlibCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryZlibFile, prefix=_ZLIB_PREFIX, extension='.z')

class BinaryGzipFile(BinaryZlibFile):
    """A file object providing transparent gzip (de)compression.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """
    wbits = 31

class GzipCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryGzipFile, prefix=_GZIP_PREFIX, extension='.gz')
