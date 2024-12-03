import os
import socket
import _socket
from multiprocessing.connection import Connection
from multiprocessing.context import get_spawning_popen
from .reduction import register
HAVE_SEND_HANDLE = hasattr(socket, 'CMSG_LEN') and hasattr(socket, 'SCM_RIGHTS') and hasattr(socket.socket, 'sendmsg')

def DupFd(fd):
    """Return a wrapper for an fd."""
    def rebuild():
        return os.dup(fd)
    return rebuild
register(socket.socket, _reduce_socket)
register(_socket.socket, _reduce_socket)
register(Connection, reduce_connection)
