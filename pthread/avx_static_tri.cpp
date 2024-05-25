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
void* sem_triplecircle_thread(void* param) { //静态线程+信号量+三重循环，avx对齐优化
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) { //0号线程做除法，其余等待

        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(A[k][k]);//将四个单精度浮点数从内存加载到向量寄存器
            for (int j = k + 1; j < k + 8 - k % 8; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }//串行计算至对齐
            for (int j = k + 8 - k % 8; j + 8 <= N; j += 8) {
                __m256 va = _mm256_load_ps(&A[k][j]);
                va = _mm256_div_ps(va, vt);//A[k][j] = A[k][j]/A[k][k];
                _mm256_store_ps(&A[k][j], va);//储存到内存
            }
            for (int j = N - N % 8; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }//结尾几个元素串行计算
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Division[t_id - 1]);

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {   //主线程唤醒其余线程
                sem_post(&sem_Division[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            __m256 vik = _mm256_set1_ps(A[i][k]);//将四个单精度浮点数从内存加载到向量寄存器
            for (int j = k + 1; j < k + 8 - k % 8; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            for (int j = k + 8 - k % 8; j + 7 < N; j += 8) {
                __m256 vkj = _mm256_load_ps(&A[k][j]);
                __m256 vij = _mm256_load_ps(&A[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_store_ps(&A[i][j], vij);
            }
            for (int j = N - N % 8; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }//结尾几个元素串行计算
            A[i][k] = 0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }

    }

    pthread_exit(NULL);
    return NULL;
}

void sem_triplecircle() {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS; i++) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, sem_triplecircle_thread, &param[t_id]);

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handle[t_id], NULL);
    }
    sem_destroy(&sem_main);    //销毁线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workstart[t_id]);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        sem_destroy(&sem_workend[t_id]);

    free(handle);
    free(param);
}

void test(void(*func)()) {
    A_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

}
int main() {

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    test(sem_triplecircle);
    cout << "avx静态8线程+信号量三重循环： " << (tail - head) * 1000 / freq << "ms" << endl;

}