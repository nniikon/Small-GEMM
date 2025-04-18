#ifndef MATMUL_H_
#define MATMUL_H_

#include <cstddef>

enum MatmulStatus
{
    MATMUL_STATUS_OK = 0,
    MATMUL_STATUS_BAD_INPUT = 1,
    MATMUL_STATUS_BAD_ALIGNMENT = 2,
    MATMUL_STATUS_BAD_SIZE = 3,
    MATMUL_STATUS_BIG_SIZE = 4,
    MATMUL_STATUS_BAD_ALLOC = 5,
};

MatmulStatus matmul_naive(size_t m, size_t n, size_t k,
                          const float* __restrict __attribute((aligned(32))) a,
                          const float* __restrict __attribute((aligned(32))) b,
                          float*       __restrict __attribute((aligned(32))) c);

MatmulStatus matmul_vec_unroll(size_t m, size_t n, size_t k,
                        const float* __restrict __attribute((aligned(32))) a,
                        const float* __restrict __attribute((aligned(32))) b,
                        float*       __restrict __attribute((aligned(32))) c);

MatmulStatus matmul_vec_transp(size_t m, size_t n, size_t k,
                               const float* __restrict __attribute((aligned(32))) a,
                               const float* __restrict __attribute((aligned(32))) b,
                               float*       __restrict __attribute((aligned(32))) c);

MatmulStatus matmul_vec_kernel(size_t m, size_t n, size_t k,
                               const float* __restrict __attribute((aligned(32))) a,
                               const float* __restrict __attribute((aligned(32))) b,
                               float*       __restrict __attribute((aligned(32))) c);

MatmulStatus matmul_kernel8x8(size_t M, size_t N, size_t K,
                              const float* __restrict __attribute__((aligned(32))) a,
                              const float* __restrict __attribute__((aligned(32))) b,
                              float*       __restrict __attribute__((aligned(32))) c);

#endif // MATMUL_H_
