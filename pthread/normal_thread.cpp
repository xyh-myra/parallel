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

sem_t sem_main;  //�ź���
sem_t sem_workstart[NUM_THREADS];
sem_t sem_workend[NUM_THREADS];

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS];
sem_t sem_Elimination[NUM_THREADS];

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

struct threadParam_t {    //�������ݽṹ
    int k;
    int t_id;
};

void A_init() {     //δ���������ĳ�ʼ��
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

void LU() {    //��ͨ��Ԫ�㷨
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

void* LU_threadFunc(void* param) {//ƽ���㷨����̬�̰߳汾
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           //��ȥ���ִ�
    int t_id = p->t_id;     //�߳�
    int i = k + t_id + 1;   //��ȡ����

    for (int j = k + 1; j < N; j++) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(NULL);
    return NULL;
}

void LU_pthread_dynamic() {    //ƽ���㷨����̬�̰߳汾
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        //���������̣߳�������ȥ����
        int worker_cnt = N - 1 - k;//�����߳�����
        pthread_t* handles = (pthread_t*)malloc(worker_cnt * sizeof(pthread_t));//������Ӧ��Handle
        threadParam_t* param = (threadParam_t*)malloc(worker_cnt * sizeof(threadParam_t));//������Ӧ���߳����ݽṹ
        //��������
        for (int t_id = 0; t_id < worker_cnt; t_id++) {//��������
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //�����߳�
        for (int t_id = 0; t_id < worker_cnt; t_id++) {
            pthread_create(&handles[t_id], NULL, LU_threadFunc, &param[t_id]);
        }
        //���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
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
    cout << "ƽ��pthread��ʱ��" << (tail - head) * 1000 / freq << "ms" << endl;


}
