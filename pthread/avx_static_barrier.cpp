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

#define NUM_THREADS 5
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
void* barrier_threadFunc(void* param) {//avx�Ż�����̬�߳�+barrierͬ��
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) { //0���߳�������
        if (t_id == 0) {
            __m256 vt = _mm256_set1_ps(A[k][k]);//���ĸ������ȸ��������ڴ���ص������Ĵ���
            for (int j = k + 1; j < k + 8 - k % 8; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }//���м���������
            for (int j = k + 8 - k % 8; j + 8 <= N; j += 8) {
                __m256 va = _mm256_load_ps(&A[k][j]);
                va = _mm256_div_ps(va, vt);//A[k][j] = A[k][j]/A[k][k];
                _mm256_store_ps(&A[k][j], va);//���浽�ڴ�
            }
            for (int j = N - N % 8; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }//��β����Ԫ�ش��м���
            A[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Division);//��һ��ͬ����

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            __m256 vik = _mm256_set1_ps(A[i][k]);//���ĸ������ȸ��������ڴ���ص������Ĵ���
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
            }//��β����Ԫ�ش��м���
            A[i][k] = 0;
        }

        pthread_barrier_wait(&barrier_Elimination);//�ڶ���ͬ����


    }
    pthread_exit(NULL);
    return NULL;
}

void barrier_static()
{
    pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    pthread_t* handle = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
    threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS * sizeof(threadParam_t));

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(&handle[t_id], NULL, barrier_threadFunc, &param[t_id]);

    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handle[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);

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

    test(barrier_static);
    cout << "avx��̬barrier��" << (tail - head) * 1000 / freq << "ms" << endl;

}