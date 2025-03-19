
#include <torch/extension.h>
#include "selective_scan/selective_scan_wrapper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // selective scan: mamba
    m.def("selective_scan_cuda_core_forward", &selective_scan_cuda_core_fwd, "Selective scan forward");
    m.def("selective_scan_cuda_core_backward", &selective_scan_cuda_core_bwd, "Selective scan backward");
}

