
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "selective_scan.h"

std::vector<at::Tensor>
selective_scan_cuda_core_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  bool delta_softplus,
                  int nrows);

std::vector<at::Tensor>
selective_scan_cuda_core_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  bool delta_softplus,
                  int nrows);

// std::vector<at::Tensor>
// selective_scan_cuda_ndstate_fwd(const at::Tensor &u, const at::Tensor &delta,
//                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
//                   const c10::optional<at::Tensor> &D_,
//                   const c10::optional<at::Tensor> &delta_bias_,
//                   bool delta_softplus,
//                   int nrows);
                  
// std::vector<at::Tensor>
// selective_scan_cuda_ndstate_bwd(const at::Tensor &u, const at::Tensor &delta,
//                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
//                   const c10::optional<at::Tensor> &D_,
//                   const c10::optional<at::Tensor> &delta_bias_,
//                   const at::Tensor &dout,
//                   const c10::optional<at::Tensor> &x_,
//                   bool delta_softplus,
//                   int nrows);

// std::vector<at::Tensor>
// selective_scan_cuda_nrow_fwd(const at::Tensor &u, const at::Tensor &delta,
//                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
//                   const c10::optional<at::Tensor> &D_,
//                   const c10::optional<at::Tensor> &delta_bias_,
//                   bool delta_softplus,
//                   int nrows);

// std::vector<at::Tensor>
// selective_scan_cuda_nrow_bwd(const at::Tensor &u, const at::Tensor &delta,
//                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
//                   const c10::optional<at::Tensor> &D_,
//                   const c10::optional<at::Tensor> &delta_bias_,
//                   const at::Tensor &dout,
//                   const c10::optional<at::Tensor> &x_,
//                   bool delta_softplus,
//                   int nrows);

// std::vector<at::Tensor>
// selective_scan_cuda_oflex_fwd(const at::Tensor &u, const at::Tensor &delta,
//                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
//                   const c10::optional<at::Tensor> &D_,
//                   const c10::optional<at::Tensor> &delta_bias_,
//                   bool delta_softplus,
//                   int nrows,
//                   bool out_float);

// std::vector<at::Tensor>
// selective_scan_cuda_oflex_bwd(const at::Tensor &u, const at::Tensor &delta,
//                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
//                   const c10::optional<at::Tensor> &D_,
//                   const c10::optional<at::Tensor> &delta_bias_,
//                   const at::Tensor &dout,
//                   const c10::optional<at::Tensor> &x_,
//                   bool delta_softplus,
//                   int nrows);                  