from concurrent.futures import Future as _BaseFuture
from concurrent.futures._base import LOGGER
import threading

class Future(_BaseFuture):
    def __init__(self):
        super().__init__()
        self._condition = threading.Condition()
        self._state = 'PENDING'
        self._result = None
        self._exception = None
        self._waiters = []

    def set_result(self, result):
        with self._condition:
            if self._state != 'PENDING':
                raise RuntimeError('Future is not pending')
            self._result = result
            self._state = 'FINISHED'
            for waiter in self._waiters:
                waiter.add_result(self)
            self._condition.notify_all()

    def set_exception(self, exception):
        with self._condition:
            if self._state != 'PENDING':
                raise RuntimeError('Future is not pending')
            self._exception = exception
            self._state = 'FINISHED'
            for waiter in self._waiters:
                waiter.add_exception(self)
            self._condition.notify_all()

    def result(self, timeout=None):
        with self._condition:
            if self._state == 'FINISHED':
                if self._exception:
                    raise self._exception
                return self._result
            self._condition.wait(timeout)
            if self._state == 'FINISHED':
                if self._exception:
                    raise self._exception
                return self._result
            raise TimeoutError()

    def exception(self, timeout=None):
        with self._condition:
            if self._state == 'FINISHED':
                return self._exception
            self._condition.wait(timeout)
            if self._state == 'FINISHED':
                return self._exception
            raise TimeoutError()

    def add_done_callback(self, fn):
        with self._condition:
            if self._state == 'FINISHED':
                fn(self)
            else:
                self._waiters.append(fn)

    def cancel(self):
        with self._condition:
            if self._state == 'PENDING':
                self._state = 'CANCELLED'
                for waiter in self._waiters:
                    waiter.add_cancelled(self)
                self._condition.notify_all()
                return True
            return False

    def cancelled(self):
        with self._condition:
            return self._state == 'CANCELLED'

    def running(self):
        with self._condition:
            return self._state == 'RUNNING'

    def done(self):
        with self._condition:
            return self._state in ('FINISHED', 'CANCELLED')
