import os
import sys
import errno
import weakref
import threading
from multiprocessing import util
from multiprocessing.queues import Full, Queue as mp_Queue, SimpleQueue as mp_SimpleQueue, _sentinel
from multiprocessing.context import assert_spawning
from .reduction import dumps
__all__ = ['Queue', 'SimpleQueue', 'Full']

class Queue(mp_Queue):

    def __init__(self, maxsize=0, reducers=None, ctx=None):
        super().__init__(maxsize=maxsize, ctx=ctx)
        self._reducers = reducers

    def __getstate__(self):
        assert_spawning(self)
        return (self._ignore_epipe, self._maxsize, self._reader, self._writer, self._reducers, self._rlock, self._wlock, self._sem, self._opid)

    def __setstate__(self, state):
        self._ignore_epipe, self._maxsize, self._reader, self._writer, self._reducers, self._rlock, self._wlock, self._sem, self._opid = state
        if sys.version_info >= (3, 9):
            self._reset()
        else:
            self._after_fork()

    def _on_queue_feeder_error(self, e, obj):
        """
        Private API hook called when feeding data in the background thread
        raises an exception.  For overriding by concurrent.futures.
        """
        import traceback
        traceback.print_exc()
        print(f"Error in queue feeder: {e}")
        print(f"Failed to feed object: {obj}")

class SimpleQueue(mp_SimpleQueue):

    def __init__(self, reducers=None, ctx=None):
        super().__init__(ctx=ctx)
        self._reducers = reducers

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._reducers, self._rlock, self._wlock)

    def __setstate__(self, state):
        self._reader, self._writer, self._reducers, self._rlock, self._wlock = state
