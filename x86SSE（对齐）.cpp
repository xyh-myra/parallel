#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <xmmintrin.h>  //SSE
#include <emmintrin.h>  //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h>  //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h>  //SSSE4.2

using namespace std;
__m128 test1, temp, temp2;
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
void SSE2(int N)
{
    for(int k = 0; k < N; k++) {
        __m128 vt = _mm_set1_ps(m[k][k]);//将四个单精度浮点数从内存加载到向量寄存器
        for(int j = k + 1; j < k + 4 - k%4; j++) {
            m[k][j] = m[k][j]/m[k][k];
        }//串行计算至对齐
        for(int j = k + 4 - k%4; j + 4 <= N; j += 4) {
            __m128 va = _mm_load_ps(&m[k][j]);
            va = _mm_div_ps(va, vt);//A[k][j] = A[k][j]/A[k][k];
            _mm_store_ps(&m[k][j], va);//储存到内存
        }
        for(int j = N - N%4; j < N; j++) {
            m[k][j] = m[k][j]/m[k][k];
        }//结尾几个元素串行计算
        m[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(m[i][k]);//将四个单精度浮点数从内存加载到向量寄存器
            for(int j = k + 1; j < k + 4 - k%4; j++) {
                m[i][j] = m[i][j] - m[i][k]*m[k][j];
            }
            for(int j = k + 4 - k%4; j + 3 < N; j += 4) {
               __m128 vkj = _mm_load_ps(&m[k][j]);
               __m128 vij = _mm_load_ps(&m[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_store_ps(&m[i][j], vij);
            }
            for(int j = N - N%4; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k]*m[k][j];
            }//结尾几个元素串行计算
            m[i][k] = 0;
        }
    }
}
int main()
{
double result=0;
for(int i=0;i<T;i++){
            m_reset(x);
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            SSE2(x);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
            cout << ((tail - head) * 1000.0 / freq) / 1000 << endl;
}

}
