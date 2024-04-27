#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <immintrin.h>
using namespace std;
long long head, tail, freq;        // timers
const int n = 2048;
int T = 1;
float m[n][n];
float result[n][n]{};
int x=2000;
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
void normal(int N)
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];

        }
        m[k][k] = 0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] -= m[i][k] * m[k][j];
            }
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
            normal(x);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
            cout << ((tail - head) * 1000.0 / freq)/1000 << endl;
            result+=((tail - head) * 1000.0 / freq)/1000;

    }
}
