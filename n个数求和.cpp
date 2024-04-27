#include <iostream>
#include<Windows.h>
using namespace std;
void add(int* a,  int n)
{
    int sum=0;
    for (int i = 0; i < n; i++)
        sum += a[i];

}
void cache(int* a,  int n)
{
    int sum, sum1, sum2;
    sum = sum1 = sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
        
    }
     sum = sum1 + sum2;
}
int main()
{
    int a[65536], step = 2;
    double time,time1;
    for (int i = 0; i < 65536; i++)
        a[i] = i;
    for (int i = 0; i <= 65536; i *= step)
    {
        // get time for size n
        //start = clock();
        LARGE_INTEGER t1, t2, tc;
        QueryPerformanceFrequency(&tc);
        QueryPerformanceCounter(&t1);
        for(int j=0;j<1000;j++)
            add(a, i);
        QueryPerformanceCounter(&t2);
        time = (double)(t2.QuadPart - t1.QuadPart)  / (double)tc.QuadPart;
        cout << "规模："<<i<<' '<<"平凡算法："<<time << "ms" << endl;  //输出时间（单位：mｓ）
        if (i == 0)
            i = 1;
        LARGE_INTEGER t11, t21, tc1;
        QueryPerformanceFrequency(&tc1);
        QueryPerformanceCounter(&t11);
        for(int j=0;j<1000;j++)
            cache(a, i);
        QueryPerformanceCounter(&t21);
        time1 = (double)(t21.QuadPart - t11.QuadPart)  / (double)tc1.QuadPart;
        cout <<"规模："<<i<<"优化算法：" << time1 <<"ms"<< endl;  //输出时间（单位：mｓ）
        if (i == 0)
            i = 1;
    }
}