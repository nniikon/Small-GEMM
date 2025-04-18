#include "matmul.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <immintrin.h>

#include <blis.h>
#include <cblas.h>

const size_t ALIGNEMENT = 32; // AVX2

typedef MatmulStatus (*matmul_func)(size_t m, size_t n, size_t k,
                                    const float *a,
                                    const float *b,
                                    float *c);

MatmulStatus matmul_openblas(size_t m, size_t n, size_t k,
                             const float* a,
                             const float* b,
                             float*       c)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
    return MATMUL_STATUS_OK;
}

const float eps = 0.01f;
bool compare_matrices(const float* a, const float* b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (std::abs((a[i] - b[i]) / a[i]) > eps) {
            return false;
        }
    }
    return true;
}

void print_matrix(const float* a, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%f ", a[i * cols + j]);
        }
        printf("\n");
    }
}

template <
    MatmulStatus (*func)(size_t, size_t, size_t,
                         const float*, const float*, float*)
>
void test_algo(size_t m, size_t n, size_t k, size_t n_tests,
               const float* a, const float* b, float* c, float* c_ref,
               const char* name)
{

    MatmulStatus status = MATMUL_STATUS_OK;

    for (size_t i = 0; i < m * n; i++) c[i] = 0.f;

    for (size_t i = 0; i < 100; i++) // Прогрев
    {
        status = func(m, n, k, a, b, c);
        if (status != MATMUL_STATUS_OK)
        {
            fprintf(stderr, "%s failed: %d\n", name, status);
            return;
        }
    }

    size_t time_start = __rdtsc();

    for (size_t i = 0; i < n_tests; i++)
        status = func(m, n, k, a, b, c);

    size_t time_end = __rdtsc();
    fprintf(stdout, "%zu\n", (time_end - time_start) / (n_tests));

    if (!compare_matrices(c, c_ref, m * n))
    {
        fprintf(stderr, "%s compare failed", name);
        fprintf(stderr, "--------------------------" "\n");
        fprintf(stderr, "%s" "\n", name);
        print_matrix(c, m, n);
        fprintf(stderr, "--------------------------" "\n");
        fprintf(stderr, "Matrix Reference" "\n");
        print_matrix(c_ref, m, n);
        fprintf(stderr, "--------------------------" "\n");
    }
}

int main(int argc, const char* argv[])
{
    size_t M = 0,
           K = 0,
           N = 0;

    size_t n_tests = 0;

    char name[100] = {};

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <name> <M> <K> <N> <number of tests>\n", argv[0]);
        return 1;
    }

    sscanf(argv[1], "%99s", name);

    sscanf(argv[2], "%zu", &M);
    sscanf(argv[3], "%zu", &K);
    sscanf(argv[4], "%zu", &N);

    sscanf(argv[5], "%zu", &n_tests);

    size_t MATRIX_A_SIZE = M * K;
    size_t MATRIX_B_SIZE = K * N;
    size_t MATRIX_C_SIZE = M * N;

    float* a     = (float*) aligned_alloc(ALIGNEMENT, MATRIX_A_SIZE * sizeof(float));
    float* b     = (float*) aligned_alloc(ALIGNEMENT, MATRIX_B_SIZE * sizeof(float));
    float* c     = (float*) aligned_alloc(ALIGNEMENT, MATRIX_C_SIZE * sizeof(float));
    float* c_ref = (float*) aligned_alloc(ALIGNEMENT, MATRIX_C_SIZE * sizeof(float));

    for (size_t i = 0; i < MATRIX_A_SIZE; i++) a[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < MATRIX_B_SIZE; i++) b[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < MATRIX_C_SIZE; i++) c[i] = 1.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a, K, b, N, 0.0f, c_ref, N);

    if      (strcmp(name, "openblas") == 0) test_algo<matmul_openblas  >(M, N, K, n_tests, a, b, c, c_ref, "openblas");
    else if (strcmp(name, "naive")    == 0) test_algo<matmul_naive     >(M, N, K, n_tests, a, b, c, c_ref, "naive");
    else if (strcmp(name, "unroll")   == 0) test_algo<matmul_vec_unroll>(M, N, K, n_tests, a, b, c, c_ref, "vec_sum");
    else if (strcmp(name, "transp")   == 0) test_algo<matmul_vec_transp>(M, N, K, n_tests, a, b, c, c_ref, "sum_transp");
    else if (strcmp(name, "kernel")   == 0) test_algo<matmul_vec_kernel>(M, N, K, n_tests, a, b, c, c_ref, "kernel");
    else
    {
        fprintf(stderr, "Unknown algorithm: %s\n", name);
        return 1;
    }

    free(a);
    free(b);
    free(c);
    free(c_ref);

    return 0;
}
