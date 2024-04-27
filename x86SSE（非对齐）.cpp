#include <iostream>
#include <windows.h>
#include <stdio.h>
#include <typeinfo>
#include <stdlib.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

using namespace std;
int x=2000;
__m128 test1, temp, temp2;
long long head, tail, freq;        // timers
const int n = 2048;
int t = 1;
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
void SSE(int N)
{
    for (int k = 0; k < N; k++) {
        __m128 t1 = _mm_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= N; j += 4) {
            __m128 t2 = _mm_loadu_ps(&m[k][j]);   //未对齐，用loadu和storeu指令
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&m[k][j], t2);
        }
        for (; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vik = _mm_set1_ps(m[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                __m128 vkj = _mm_loadu_ps(&m[k][j]);
                __m128 vij = _mm_loadu_ps(&m[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&m[i][j], vij);
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}
int main()
{
    double result=0;
    for (int j = 0; j <t ; j++) {
            m_reset(x);
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            SSE(x);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
            cout << ((tail - head) * 1000.0 / freq) / 1000 << endl;

    }

}
