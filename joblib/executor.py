"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""
from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from .externals.loky.reusable_executor import _ReusablePoolExecutor
_executor_args = None

class MemmappingExecutor(_ReusablePoolExecutor):

    @classmethod
    def get_memmapping_executor(cls, n_jobs, timeout=300, initializer=None, initargs=(), env=None, temp_folder=None, context_id=None, **backend_args):
        """Factory for ReusableExecutor with automatic memmapping for large
        numpy arrays.
        """
        memmapping_reducers = get_memmapping_reducers(
            temp_folder_resolver=TemporaryResourcesManager(temp_folder, context_id).resolve_temp_folder_name,
            **backend_args
        )
        job_reducers, result_reducers = memmapping_reducers

        return get_reusable_executor(
            max_workers=n_jobs,
            timeout=timeout,
            job_reducers=job_reducers,
            result_reducers=result_reducers,
            initializer=initializer,
            initargs=initargs,
            env=env,
            **backend_args
        )

class _TestingMemmappingExecutor(MemmappingExecutor):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.

    """

    def apply_async(self, func, args=(), kwds=None, callback=None, error_callback=None):
        """Schedule a func to be run"""
        if kwds is None:
            kwds = {}
        future = self.submit(func, *args, **kwds)
        
        if callback or error_callback:
            def _callback_wrapper(future):
                try:
                    result = future.result()
                    if callback:
                        callback(result)
                except Exception as exc:
                    if error_callback:
                        error_callback(exc)
                    else:
                        raise
            future.add_done_callback(_callback_wrapper)
        
        return future
