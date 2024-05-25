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

void LU() {    //普通消元算法
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void* LU_threadFunc(void* param) {//平凡算法，动态线程版本
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //消去的轮次
    int t_id = p->t_id;     //线程
    int i = k + t_id + 1;   //获取任务

    for (int j = k + 1; j < N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}

void LU_pthread_dynamic() {    //平凡算法，动态线程版本
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        //创建工作线程，进行消去工作
        int worker_cnt = N - 1 - k;//工作线程数量
        pthread_t* handles = (pthread_t*)malloc(worker_cnt * sizeof(pthread_t));//创建对应的Handle
        threadParam_t* param = (threadParam_t*)malloc(worker_cnt * sizeof(threadParam_t));//创建对应的线程数据结构
        //分配任务
        for (int t_id = 0; t_id < worker_cnt; t_id++) {//分配任务
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < worker_cnt; t_id++) {
            pthread_create(&handles[t_id], NULL, LU_threadFunc, &param[t_id]);
        }
        //主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_cnt; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        free(handles);
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
    test(LU_pthread_dynamic);
    cout << "平凡pthread耗时：" << (tail - head) * 1000 / freq << "ms" << endl;


}
