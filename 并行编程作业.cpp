#include <iostream>
#include<Windows.h>
using namespace std;
void matrix(int* a, int b[][1000],int n)
{
    float* sum;
    sum = new float[n];

    for (int i = 0; i < n; i++)
        sum[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum[i] += b[j][i] * a[j];
    delete sum;
}
void matrix2(int* a, int b[][1000], int n)
{
    float* sum;
    sum = new float[n];
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
         for (int j = 0; j < n; j++)
             sum[i] += b[j][i] * a[j];    
    }
    delete sum;
}
int main()
{
    int a[1000], b[1000][1000],step = 10;
    double time;
    for (int i = 0; i < 1000; i++)
    {
        a[i] = i;
        for (int j = 0; j < 1000; j++)
            b[i][j] = j;
    }
    for (int i = 0; i <= 1000; i += step)
    {
        LARGE_INTEGER t1, t2, tc;
        QueryPerformanceFrequency(&tc);
        QueryPerformanceCounter(&t1);
        for(int j=0;j<100;j++)
            matrix2(a, b, i);
        QueryPerformanceCounter(&t2);
        time = (double)(t2.QuadPart - t1.QuadPart)*10  / (double)tc.QuadPart;
        cout << "规模：" << i;
        cout << "平凡算法time = " << time << "ms" << endl;  //输出时间（单位：mｓ）
        LARGE_INTEGER t11, t21, tc1;
        QueryPerformanceFrequency(&tc1);
        QueryPerformanceCounter(&t11);
        for(int j=0;j<100;j++)
             matrix(a, b, i);
        QueryPerformanceCounter(&t21);
        time = (double)(t21.QuadPart - t11.QuadPart)*10 / (double)tc1.QuadPart;
        cout << "规模：" << i;
        cout << "优化算法time = " << time <<"ms"<< endl;  //输出时间（单位：mｓ）
        if (i == 100)
            step = 100;
        if (i == 1000)
            step = 1000;

    
        }
}