from functools import wraps


class TaskCancelled(InterruptedError):pass


def checkpoint(cancelled):
    if cancelled and cancelled():raise TaskCancelled('任务已取消')


def cancellable_result(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        try:return function(*args,**kwargs)
        except TaskCancelled:return None
    return wrapped


def cancellable_false(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        try:return function(*args,**kwargs)
        except TaskCancelled:return False
    return wrapped
