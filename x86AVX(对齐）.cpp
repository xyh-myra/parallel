#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <immintrin.h> //AVX、AVX2、AVX-512
using namespace std;
long long head, tail, freq;        // timers
const int n = 2048;
int T = 1;
int x=2000;
float m[n][n];
void m_reset(int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}
void AVX(int N)
{
     for(int k = 0; k < N; k++) {
        __m256 vt = _mm256_set1_ps(m[k][k]);//将四个单精度浮点数从内存加载到向量寄存器
        for(int j = k + 1; j < k + 8 - k%8; j++) {
            m[k][j] = m[k][j]/m[k][k];
        }//串行计算至对齐
        for(int j = k + 8 - k%8; j + 8 <= N; j += 8) {
            __m256 va = _mm256_load_ps(&m[k][j]);
            va = _mm256_div_ps(va, vt);//A[k][j] = A[k][j]/A[k][k];
            _mm256_store_ps(&m[k][j], va);//储存到内存
        }
        for(int j = N - N%8; j < N; j++) {
            m[k][j] = m[k][j]/m[k][k];
        }//结尾几个元素串行计算
        m[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(m[i][k]);//将四个单精度浮点数从内存加载到向量寄存器
            for(int j = k + 1; j < k + 8 - k%8; j++) {
                m[i][j] = m[i][j] - m[i][k]*m[k][j];
            }
            for(int j = k + 8 - k%8; j + 7 < N; j += 8) {
               __m256 vkj = _mm256_load_ps(&m[k][j]);
               __m256 vij = _mm256_load_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_store_ps(&m[i][j], vij);
            }
            for(int j = N - N%8; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k]*m[k][j];
            }//结尾几个元素串行计算
            m[i][k] = 0;
        }
    }
}
int main()
{

    double result=0;

    for (int j = 0; j < T; j++) {
            m_reset(x);
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            AVX(x);
               QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
    cout << ((tail - head) * 1000.0 / freq) / 1000 << endl;

    }
}
