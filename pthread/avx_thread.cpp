#include<iostream>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<semaphore.h>
#include<pthread.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<windows.h>
using namespace std;
#define N 1000

#define NUM_THREADS 7
float A[N][N];


long long head, tail, freq;

sem_t sem_main;  //信号量
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

struct threadParam_t {    //参数数据结构
    int k;
    int t_id;
};

void A_init() {     //未对齐的数组的初始化
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
void* avx_threadFunc(void* param) {  //avx对齐优化与动态线程结合
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务

    __m256 vaik = _mm256_set1_ps(A[i][k]);
    for (int j = k + 1; j < k + 8 - k % 8; j++) {
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
    int j;
    for (j = k + 1; j + 8 <= N; j += 8) {
        __m256 vkj = _mm256_load_ps(&A[k][j]);
        __m256 vij = _mm256_load_ps(&A[i][j]);
        __m256 vx = _mm256_mul_ps(vaik, vkj);
        vij = _mm256_sub_ps(vij, vx);
        _mm256_store_ps(&A[i][j], vij);
    }
    for (; j < N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}

void avx_dynamic() {            //avx对齐优化算法,动态分配
    for (int k = 0; k < N; k++) {
        __m256 vt = _mm256_set1_ps(A[k][k]);
        for (int j = k + 1; j < k + 8 - k % 8; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }//串行计算至对齐
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 va = _mm256_load_ps(&A[k][j]);
            va = _mm256_div_ps(va, vt);//A[k][j] = A[k][j]/A[k][k];
            _mm256_store_ps(&A[k][j], va);//储存到内存
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        int thread_cnt = N - 1 - k;
        pthread_t* handle = (pthread_t*)malloc(thread_cnt * sizeof(pthread_t));
        threadParam_t* param = (threadParam_t*)malloc(thread_cnt * sizeof(threadParam_t));

        for (int t_id = 0; t_id < thread_cnt; t_id++) {//分配任务
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++) {
            pthread_create(&handle[t_id], NULL, avx_threadFunc, &param[t_id]);
        }

        for (int t_id = 0; t_id < thread_cnt; t_id++) {
            pthread_join(handle[t_id], NULL);
        }
        free(handle);
        free(param);
    }


}


void test(void(*func)()) {
    A_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

}
int main() {

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    test(avx_dynamic);
    cout << "avx优化pthread耗时：" << (tail - head) * 1000 / freq << "ms" << endl;



}
