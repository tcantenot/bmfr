#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define K_SUPPORT_HALF16_ARITHMETIC (__CUDA_ARCH__ >= 530)
