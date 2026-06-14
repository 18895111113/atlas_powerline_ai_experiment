import numpy as np
from threading import Lock

try:
    import acl
except ImportError as exc:
    raise ImportError(
        "Failed to import acl. Please make sure CANN/pyACL has been installed on the Atlas device."
    ) from exc


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_REPEAT_INITIALIZE = 100002

_acl_init_lock = Lock()
_acl_init_ref_count = 0


def check_ret(name, ret):
    if ret != 0:
        raise RuntimeError(f"{name} failed, ret={ret}")


def acl_init_once():
    global _acl_init_ref_count
    with _acl_init_lock:
        if _acl_init_ref_count == 0:
            ret = acl.init()
            if ret not in (0, ACL_ERROR_REPEAT_INITIALIZE):
                check_ret("acl.init", ret)
        _acl_init_ref_count += 1


def acl_finalize_once():
    global _acl_init_ref_count
    with _acl_init_lock:
        if _acl_init_ref_count <= 0:
            return
        _acl_init_ref_count -= 1
        if _acl_init_ref_count == 0:
            check_ret("acl.finalize", acl.finalize())


class AclModelRunner:
    def __init__(self, model_path, device_id=0):
        self.model_path = model_path
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []
        self.output_shapes = []
        self.output_dtypes = []

        acl_init_once()
        check_ret("acl.rt.set_device", acl.rt.set_device(self.device_id))
        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)
        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        check_ret("acl.mdl.get_desc", acl.mdl.get_desc(self.model_desc, self.model_id))

        self._prepare_datasets()

    def _prepare_datasets(self):
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()

        input_num = acl.mdl.get_num_inputs(self.model_desc)
        output_num = acl.mdl.get_num_outputs(self.model_desc)

        for index in range(input_num):
            size = acl.mdl.get_input_size_by_index(self.model_desc, index)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc(input)", ret)
            data_buffer = acl.create_data_buffer(ptr, size)
            self.input_dataset, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            check_ret("acl.mdl.add_dataset_buffer(input)", ret)
            self.input_buffers.append({"ptr": ptr, "size": size, "buffer": data_buffer})

        for index in range(output_num):
            size = acl.mdl.get_output_size_by_index(self.model_desc, index)
            ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc(output)", ret)
            data_buffer = acl.create_data_buffer(ptr, size)
            self.output_dataset, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
            check_ret("acl.mdl.add_dataset_buffer(output)", ret)
            self.output_buffers.append({"ptr": ptr, "size": size, "buffer": data_buffer})

            try:
                dims, ret = acl.mdl.get_output_dims(self.model_desc, index)
                check_ret("acl.mdl.get_output_dims", ret)
                shape = tuple(dims["dims"][: dims["dimCount"]])
            except Exception:
                shape = (size // 4,)
            self.output_shapes.append(shape)
            self.output_dtypes.append(np.float32)

    def infer(self, input_tensor):
        # The ACL context is thread-local. Rebind it on each inference call
        # because Flask may dispatch requests on a different worker thread.
        check_ret("acl.rt.set_context", acl.rt.set_context(self.context))
        input_tensor = np.ascontiguousarray(input_tensor.astype(np.float32))
        input_bytes = input_tensor.tobytes()
        if len(input_bytes) > self.input_buffers[0]["size"]:
            raise ValueError("input tensor larger than model input buffer")

        input_ptr = acl.util.bytes_to_ptr(input_bytes)
        ret = acl.rt.memcpy(
            self.input_buffers[0]["ptr"],
            self.input_buffers[0]["size"],
            input_ptr,
            len(input_bytes),
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        check_ret("acl.rt.memcpy(H2D)", ret)

        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret("acl.mdl.execute", ret)

        outputs = []
        for index, item in enumerate(self.output_buffers):
            host_ptr, ret = acl.rt.malloc_host(item["size"])
            check_ret("acl.rt.malloc_host", ret)
            ret = acl.rt.memcpy(
                host_ptr,
                item["size"],
                item["ptr"],
                item["size"],
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            check_ret("acl.rt.memcpy(D2H)", ret)
            raw = acl.util.ptr_to_bytes(host_ptr, item["size"])
            array = np.frombuffer(raw, dtype=self.output_dtypes[index]).copy()
            try:
                array = array.reshape(self.output_shapes[index])
            except ValueError:
                pass
            outputs.append(array)
            check_ret("acl.rt.free_host", acl.rt.free_host(host_ptr))
        return outputs

    def release(self):
        if self.input_dataset is not None:
            for item in self.input_buffers:
                acl.destroy_data_buffer(item["buffer"])
                acl.rt.free(item["ptr"])
            acl.mdl.destroy_dataset(self.input_dataset)
            self.input_dataset = None

        if self.output_dataset is not None:
            for item in self.output_buffers:
                acl.destroy_data_buffer(item["buffer"])
                acl.rt.free(item["ptr"])
            acl.mdl.destroy_dataset(self.output_dataset)
            self.output_dataset = None

        if self.model_desc is not None:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        if self.model_id is not None:
            acl.mdl.unload(self.model_id)
            self.model_id = None

        if self.stream is not None:
            acl.rt.destroy_stream(self.stream)
            self.stream = None

        if self.context is not None:
            acl.rt.destroy_context(self.context)
            self.context = None

        acl.rt.reset_device(self.device_id)
        acl_finalize_once()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
