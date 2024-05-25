#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include<arm_neon.h>
using namespace std;
#define N 1000
#define NUM_THREADS 7
float A[N][N];
float32x4_t test1;
struct timespec sts, ets;
time_t dsec;
long dnsec;
void A_init() {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            A[i][j] = 0;
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            A[i][j] = rand() % 1000;
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] += A[k][j];
}


void LU() {
    int i, j, k;
    for (k = 0; k < N; k++) {
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void LU_omp() {    //
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        A[k][k] = 1.0;
        }

#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void LU_omp_guided() {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void LU_omp_dynamic() {
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}



void neon_omp_static() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        int j;
        int off = ((unsigned long int)(&A[k][k + 1])) % 16 / 4;
        for (j = k + 1; j < k + 1 + off && j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (; j <= N - 4; j += 4)
        {
            test1 = vld1q_f32(&(A[k][j]));
            float32x4_t temp = vmovq_n_f32(A[k][k]);
            test1 = vdivq_f32(test1, temp);
            vst1q_f32(&(A[k][j]), test1);

        }
        for (; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
#pragma omp for schedule(static)
        for (i = k + 1; i < N; i++) {

            int j;
            int off = ((unsigned long int)(&A[i][k + 1])) % 16 / 4;
            for (j = k + 1; j < k + 1 + off && j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            for (; j <= N - 4; j += 4)
            {
                test1 = vld1q_f32((&A[i][j]));
                float32x4_t temp1 = vmovq_n_f32(A[i][k]);
                float32x4_t temp2 = vld1q_f32(&(A[k][j]));
                temp1 = vmulq_f32(temp1, temp2);
                test1 = vsubq_f32(test1, temp1);
            }
            for (; j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void neon_omp_dynamic() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            int j;
            int off = ((unsigned long int)(&A[k][k + 1])) % 16 / 4;
            for (j = k + 1; j < k + 1 + off && j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            for (; j <= N - 4; j += 4)
            {
                test1 = vld1q_f32(&(A[k][j]));
                float32x4_t temp = vmovq_n_f32(A[k][k]);
                test1 = vdivq_f32(test1, temp);
                vst1q_f32(&(A[k][j]), test1);

            }
            for (; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            int j;
            int off = ((unsigned long int)(&A[i][k + 1])) % 16 / 4;
            for (j = k + 1; j < k + 1 + off && j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            for (; j <= N - 4; j += 4)
            {
                test1 = vld1q_f32((&A[i][j]));
                float32x4_t temp1 = vmovq_n_f32(A[i][k]);
                float32x4_t temp2 = vld1q_f32(&(A[k][j]));
                temp1 = vmulq_f32(temp1, temp2);
                test1 = vsubq_f32(test1, temp1);
            }
            for (; j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void neon_optimized() {
    int k;
    for (k = 0; k < N; k++)
    {
        int j;
        int off = ((unsigned long int)(&A[k][k + 1])) % 16 / 4;
        for (j = k + 1; j < k + 1 + off && j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (; j <= N - 4; j += 4)
        {
            test1 = vld1q_f32(&(A[k][j]));
            float32x4_t temp = vmovq_n_f32(A[k][k]);
            test1 = vdivq_f32(test1, temp);
            vst1q_f32(&(A[k][j]), test1);

        }
        for (; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 0;
        for (int i = k + 1; i < N; i++)
        {
            int j;
            int off = ((unsigned long int)(&A[i][k + 1])) % 16 / 4;
            for (j = k + 1; j < k + 1 + off && j < N; j++)
            {
               A[i][j] -= A[i][k] * A[k][j];
            }
            for (; j <= N - 4; j += 4)
            {
                test1 = vld1q_f32((&A[i][j]));
                float32x4_t temp1 = vmovq_n_f32(A[i][k]);
                float32x4_t temp2 = vld1q_f32(&(A[k][j]));
                temp1 = vmulq_f32(temp1, temp2);
                test1 = vsubq_f32(test1, temp1);


            }
            for (; j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }

    }
}

void neon_omp_guided() {
    int i = 0, j = 0, k = 0;

#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            int j;
            int off = ((unsigned long int)(&A[k][k + 1])) % 16 / 4;
            for (j = k + 1; j < k + 1 + off && j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            for (; j <= N - 4; j += 4)
            {
                test1 = vld1q_f32(&(A[k][j]));
                float32x4_t temp = vmovq_n_f32(A[k][k]);
                test1 = vdivq_f32(test1, temp);
                vst1q_f32(&(A[k][j]), test1);
            }
            for (; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            int j;
            int off = ((unsigned long int)(&A[i][k + 1])) % 16 / 4;
            for (j = k + 1; j < k + 1 + off && j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            for (; j <= N - 4; j += 4)
            {
                test1 = vld1q_f32((&A[i][j]));
                float32x4_t temp1 = vmovq_n_f32(A[i][k]);
                float32x4_t temp2 = vld1q_f32(&(A[k][j]));
                temp1 = vmulq_f32(temp1, temp2);
                test1 = vsubq_f32(test1, temp1);
            }
            for (; j < N; j++)
            {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void test(void(*func)()) {
    A_init();
    timespec_get(&sts, TIME_UTC);
    func();
    timespec_get(&ets, TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if (dnsec < 0) {
        dsec--;
        dnsec += 1000000000ll;
    }

}


int main() {
    test(LU);
    printf("平凡串行算法耗时： %ld.%09lds\n", dsec, dnsec);

    test(LU_omp);
    printf("平凡算法static耗时： %ld.%09lds\n", dsec, dnsec);

    test(LU_omp_dynamic);
    printf("平凡算法dynamic耗时： %ld.%09lds\n", dsec, dnsec);

    test(LU_omp_guided);
    printf("平凡算法guided耗时： %ld.%09lds\n", dsec, dnsec);

    test(neon_optimized);
    printf("neon优化算法算法耗时： %ld.%09lds\n", dsec, dnsec);

    test(neon_omp_static);
    printf("neon优化算法static耗时： %ld.%09lds\n", dsec, dnsec);

    test(neon_omp_dynamic);
    printf("neon优化算法dynamic耗时： %ld.%09lds\n", dsec, dnsec);

    test(neon_omp_guided);
    printf("neon优化算法guided耗时： %ld.%09lds\n", dsec, dnsec);
}

