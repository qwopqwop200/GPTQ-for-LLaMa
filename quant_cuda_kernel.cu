#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void VecQuant4MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);


// https://github.com/iwalton3/GPTQ-for-LLaMa/commit/209d16b0187f149bf13318360925cc4f679cb2ea
template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_G(
    const     half2* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
);

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 = 32;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);
  
  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernel<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<int>(),
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void VecQuant4MatMulKernel(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8; 
  int z_mod = (w % 8) * 4;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);
    half2 zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)));
	
    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
	i += width;
    k += 4;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant4matmul_g_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_SWITCH(vec.type(), "vecquant4matmul_g_cuda",
    AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
      VecQuant4MatMulKernel_G<<<blocks, threads>>>(
        (half2*) vec.data_ptr<scalar_t>(),
        mat.data_ptr<int>(),
        mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),
        zeros.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  ));
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_G(
    const     half2* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	    int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  scalar_t res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    res2 = {};
    tmp = as_unsigned(mat[i]);

    int tmp_k = 0;
    half2 scales_tmp[4];
    half2 zeros_tmp[4];
    while (tmp_k < 4) {
      int g = as_int(g_idx[g_h + (k + tmp_k) * 2]);
      int g2 = as_int(g_idx[g_h + (k + tmp_k) * 2 + 1]);
      scalar_t scale_f = scales[g * width + w];
      scalar_t scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
        __hmul(-scale_f, __int2half_rn(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)),
        __hmul(-scale_f2, __int2half_rn(((as_unsigned(zeros[g2 * zero_width + z_w]) >> z_mod) & 0xF) + 1))
      );
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
      tmp_k += 1;
    }

    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scales_tmp[0], zeros_tmp[0]), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scales_tmp[1], zeros_tmp[1]), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]), blockvec[k + 3], res2);
	i += width;
    k += 4;
    res = __hadd(res, __hadd(res2.x, res2.y));;
  }

  __half* mul2 = (__half*)mul;
  atomicAdd(&mul2[b * width + w], res);
}