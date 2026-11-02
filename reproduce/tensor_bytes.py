"""Copy owned contiguous CPU tensor bytes in one operation, without NumPy."""
import ctypes


def raw_tensor_bytes(value):
    value=value.detach().cpu().contiguous()
    # data_ptr addresses this live tensor; the copy is bounded by its logical size.
    return ctypes.string_at(value.data_ptr(),value.numel()*value.element_size())
