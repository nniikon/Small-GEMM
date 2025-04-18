#include "matmul.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

const size_t VECTOR_SIZE = 8;
#define FOR_VECTOR(iter) for(int iter = 0; iter < VECTOR_SIZE; iter++)

static bool IsPowerOfTwo(size_t x)
{
    // Трюк =)
    return (x & (x - 1)) == 0;
}

__always_inline MatmulStatus matmul_check(
        size_t M, size_t N, size_t K,
        const float* __restrict __attribute((aligned(32))) a,
        const float* __restrict __attribute((aligned(32))) b,
        float*       __restrict __attribute((aligned(32))) c
    )
{
    if (!IsPowerOfTwo(M) || !IsPowerOfTwo(N) || !IsPowerOfTwo(K)) {
        fprintf(stderr, "matmul_check failed: M, N, or K not power of two (M=%zu, N=%zu, K=%zu)\n", M, N, K);
        return MATMUL_STATUS_BAD_SIZE;
    }

#if 0
    if ((M * K + K * N + M * N) * sizeof(float) >= 32 * 1024) {
        fprintf(stderr, "matmul_check failed: Size too big (Total kilobytes=%zu)\n",
                (M * K + K * N + M * N) * sizeof(float) / 1024);
        return MATMUL_STATUS_BIG_SIZE;
    }
#endif

    if ((size_t)a % 32 != 0 || (size_t)b % 32 != 0 || (size_t)c % 32 != 0) {
        fprintf(stderr, "matmul_check failed: Bad alignment (a=%p, b=%p, c=%p)\n", (const void*)a, (const void*)b, (void*)c);
        return MATMUL_STATUS_BAD_ALIGNMENT;
    }

    if (a == nullptr || b == nullptr || c == nullptr) {
        fprintf(stderr, "matmul_check failed: Nullptr input (a=%p, b=%p, c=%p)\n", (const void*)a, (const void*)b, (void*)c);
        return MATMUL_STATUS_BAD_INPUT;
    }

    return MATMUL_STATUS_OK;
}

MatmulStatus matmul_naive(
        size_t M, size_t N, size_t K,
        const float* __restrict __attribute((aligned(32))) a,
        const float* __restrict __attribute((aligned(32))) b,
        float*       __restrict __attribute((aligned(32))) c
    )
{
    MatmulStatus status = matmul_check(M, N, K, a, b, c);
    if (status != MATMUL_STATUS_OK)
        return status;

    for (size_t i = 0; i < M * N; i++)
        c[i] = 0;

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < K; k++)
                c[i * N + j] += a[i * K + k] * b[k * N + j];

    return MATMUL_STATUS_OK;
}

const int UNROLL_FACTOR = 8;
MatmulStatus matmul_vec_unroll(
        size_t M, size_t N, size_t K,
        const float* __restrict __attribute((aligned(32))) a,
        const float* __restrict __attribute((aligned(32))) b,
        float*       __restrict __attribute((aligned(32))) c
    )
{
    MatmulStatus status = matmul_check(M, N, K, a, b, c);
    if (status != MATMUL_STATUS_OK)
        return status;

    for (size_t i = 0; i < M * N; i++)
        c[i] = 0;

    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < K; k += UNROLL_FACTOR)
        {
            float a_ik[UNROLL_FACTOR] = {}; // a[i*K + k];
            for (size_t u = 0; u < UNROLL_FACTOR; u++)
            {
                a_ik[u] = a[i * K + k + u];
            }

            for (size_t j = 0; j < N; j += VECTOR_SIZE)
            {
                for (size_t u = 0; u < UNROLL_FACTOR; u++)
                {
                    FOR_VECTOR(v)
                        c[i*N + j + v] += a_ik[u] * b[(k + u)*N + j + v];
                }
            }
        }
    }

    return MATMUL_STATUS_OK;
}

#include <stdlib.h>

MatmulStatus matmul_vec_transp(
        size_t M, size_t N, size_t K,
        const float* __restrict __attribute__((aligned(32))) a,
        const float* __restrict __attribute__((aligned(32))) b,
        float*       __restrict __attribute__((aligned(32))) c
    )
{
    MatmulStatus status = matmul_check(M, N, K, a, b, c);
    if (status != MATMUL_STATUS_OK)
        return status;

    for (size_t i = 0; i < M * N; i++) {
        c[i] = 0;
    }

    float* b_t = (float*) aligned_alloc(32, N * K * sizeof(float));
    if (b_t == NULL) {
        return MATMUL_STATUS_BAD_ALLOC;
    }

    for (size_t k = 0; k < K; k++) {
        for (size_t j = 0; j < N; j++) {
            b_t[j * K + k] = b[k * N + j];
        }
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b_t[j * K + k];
            }
            c[i * N + j] = sum;
        }
    }

    free(b_t);
    return MATMUL_STATUS_OK;
}

#include <immintrin.h>

MatmulStatus matmul_kernel8x8(
    size_t M, size_t N, size_t K,
    const float* __restrict __attribute__((aligned(32))) a,
    const float* __restrict __attribute__((aligned(32))) b,
    float*       __restrict __attribute__((aligned(32))) c
) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m256 b_vec = _mm256_load_ps(&b[k * N]);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[0 * K + k]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[1 * K + k]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(a[2 * K + k]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(a[3 * K + k]), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(a[4 * K + k]), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(a[5 * K + k]), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(a[6 * K + k]), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(a[7 * K + k]), b_vec, c7);
    }

    _mm256_store_ps(&c[0 * N], c0);
    _mm256_store_ps(&c[1 * N], c1);
    _mm256_store_ps(&c[2 * N], c2);
    _mm256_store_ps(&c[3 * N], c3);
    _mm256_store_ps(&c[4 * N], c4);
    _mm256_store_ps(&c[5 * N], c5);
    _mm256_store_ps(&c[6 * N], c6);
    _mm256_store_ps(&c[7 * N], c7);

    return MATMUL_STATUS_OK;
}

MatmulStatus matmul_vec_kernel(
        size_t M, size_t N, size_t K,
        const float* __restrict __attribute__((aligned(32))) a,
        const float* __restrict __attribute__((aligned(32))) b,
        float*       __restrict __attribute__((aligned(32))) c
    )
{
    MatmulStatus status = matmul_check(M, N, K, a, b, c);
    if (status != MATMUL_STATUS_OK)
        return status;

    for (int i = 0; i < M; i += 8)
    {
        for (int j = 0; j < N; j += 8)
        {
            matmul_kernel8x8(M, N, K, &a[i * K], &b[j], &c[i * N + j]);
        }
    }

    return MATMUL_STATUS_OK;
}
