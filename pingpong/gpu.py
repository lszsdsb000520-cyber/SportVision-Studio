"""Match DirectML's DXGI adapter order and prefer dedicated video memory."""
from functools import lru_cache
import sys


@lru_cache(maxsize=1)
def preferred_adapter():
    if sys.platform != 'win32':
        return 0
    import ctypes as c
    from ctypes import wintypes as w
    from uuid import UUID

    class AdapterDescription(c.Structure):
        _fields_ = [('description', w.WCHAR * 128), ('vendor', w.UINT),
                    ('device', w.UINT), ('subsystem', w.UINT), ('revision', w.UINT),
                    ('video_memory', c.c_size_t), ('system_memory', c.c_size_t),
                    ('shared_memory', c.c_size_t), ('luid', w.DWORD * 2)]

    def method(pointer, index, result, *args):
        table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(result, c.c_void_p, *args)(table[index])

    factory = c.c_void_p()
    try:
        create = c.WinDLL('dxgi').CreateDXGIFactory
        create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
        iid = (c.c_ubyte * 16).from_buffer_copy(UUID('7b7166ec-21c7-44ae-b21a-c9ae321ae369').bytes_le)
        if create(iid, c.byref(factory)) != 0:
            return 0
        enumerate_adapter = method(factory, 7, c.c_long, w.UINT, c.POINTER(c.c_void_p))
        best_index, best_memory = 0, 0
        for index in range(16):
            adapter = c.c_void_p()
            if enumerate_adapter(factory, index, c.byref(adapter)) != 0:
                break
            try:
                desc = AdapterDescription()
                status = method(adapter, 8, c.c_long, c.POINTER(AdapterDescription))(adapter, c.byref(desc))
                if status == 0 and desc.video_memory > best_memory:
                    best_index, best_memory = index, desc.video_memory
            finally:
                method(adapter, 2, w.ULONG)(adapter)
        return best_index
    except (OSError, AttributeError):
        return 0
    finally:
        if factory:
            method(factory, 2, w.ULONG)(factory)
