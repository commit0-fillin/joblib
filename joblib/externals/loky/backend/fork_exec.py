import os
import sys

def close_fds(keep_fds):
    """Close all the file descriptors except those in keep_fds."""
    import resource
    maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if maxfd == resource.RLIM_INFINITY:
        maxfd = 1024  # Use a reasonable default if RLIMIT_NOFILE is infinite

    for fd in range(3, maxfd):  # Skip 0, 1, 2 (stdin, stdout, stderr)
        if fd not in keep_fds:
            try:
                os.close(fd)
            except OSError:
                pass  # Ignore errors for file descriptors that are not open
