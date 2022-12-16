#pragma once
#ifndef _TORCH_GATHER_H
#define _TORCH_GATHER_H

//#include <torch/serialize/tensor.h>
//#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>

template <typename T>
void torch_gather(T* output, const T* input, const int* index, int dim,
                  int* input_dims, int* index_dims, int nb_dims,
                  cudaStream_t stream);

#endif
