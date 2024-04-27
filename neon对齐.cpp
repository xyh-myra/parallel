#include <stdio.h>
#include <time.h>
#include <iostream>
#include<ctime>
#include<arm_neon.h>
using namespace std;
float32x4_t test1;
struct timespec sts,ets;
const int n=2048;
float m[n][n];
int T=10;
void m_reset()
{
         for (int i = 0; i < n; i++)
                 {  for (int j = 0; j < i; j++)
                 m[i][j] = 0;
         m[i][i] = 1.0;
         for (int j = i + 1; j < n; j++)
                 m[i][j] = rand();
         }
         for (int k = 0; k < n; k++)
                 for (int i = k + 1; i < n; i++)
                 for (int j = 0; j < n; j++)
                m[i][j] += m[k][j];
         }
void Neon2(int N)
{               int k;
        for ( k = 0; k < N; k++)
        {int j;
                int off=((unsigned long int)(&m[k][k+1]))%16/4;
                for(j=k+1;j<k+1+off&&j<N;j++)
                        {
                        m[k][j]=m[k][j]/m[k][k];
                        }
                for (; j <= N-4; j+=4)
                {
                        test1=vld1q_f32(&(m[k][j]));
                        float32x4_t temp=vmovq_n_f32(m[k][k]);
                        test1=vdivq_f32(test1,temp);
                        vst1q_f32(&(m[k][j]),test1);

                }
                for(;j<N;j++)
                        {
                        m[k][j]=m[k][j]/m[k][k];
                        }
                m[k][k] = 0;
                for (int i = k + 1; i < N; i++)
                {int j;
                        int off=((unsigned long int)(&m[i][k+1]))%16/4;
                        for(j=k+1;j<k+1+off&&j<N;j++)
                        {
                        m[i][j]-=m[i][k]*m[k][j];
                        }
                        for (; j <= N-4; j+=4)
                        {       test1=vld1q_f32((&m[i][j]));
                                float32x4_t temp1=vmovq_n_f32(m[i][k]);
                                float32x4_t temp2=vld1q_f32(&(m[k][j]));
                                temp1=vmulq_f32(temp1,temp2);
                                test1=vsubq_f32(test1,temp1);


                        }
                        for(;j<N;j++)
                                {
                                m[i][j]-=m[i][k]*m[k][j];
                                }
                        m[i][k] = 0;
                }

        }
}
int main()
{for(int j=0;j<T;j++){

        m_reset();
        timespec_get(&sts, TIME_UTC);
        Neon2(2000);
       timespec_get(&ets, TIME_UTC);
time_t dsec=ets.tv_sec-sts.tv_sec;
long dnsec=ets.tv_nsec-sts.tv_nsec;
if(dnsec<0)
{
dsec--;
dnsec+=1000000000ll;
}
 printf ("%llu.%09llu \n",dsec,dnsec);
cout<<endl;
}
}

