#include<iostream>
#include<time.h>
#include<arm_neon.h>
using namespace std;
const int n = 2048;
int T = 10;
float m[n][n];
float a[n][n];
void m_reset(int N){
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
}
void LU(int N){      //平凡算法
    for(int k = 0; k<N; k++){
        for(int j = k+1; j<N; j++ ){
            m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++){
                m[i][j] = m[i][j]-m[i][k]*m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void neon_optimized(int N){            //neon优化算法
    for(int k = 0; k < N; k++){
        float32x4_t vt = vdupq_n_f32(m[k][k]);
        int j = 0;
        for(j = k+1; j+4 <= N; j+=4){
                float32x4_t va = vld1q_f32(&m[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&m[k][j], va);
        }
        for( ;j < N; j++){
                m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1; i < N; i++){
                float32x4_t vaik = vdupq_n_f32(m[i][k]);
                for(j = k+1; j+4 <= N; j+=4){
                        float32x4_t vakj = vld1q_f32(&m[k][j]);
                        float32x4_t vaij = vld1q_f32(&m[i][j]);
                        float32x4_t vx = vmulq_f32(vakj, vaik);
                        vaij = vsubq_f32(vaij , vx);
                        vst1q_f32(&m[i][j] , vaij);
        }
        for( ; j < N; j++){
                m[i][j] = m[i][j]-m[i][k]*m[k][j];
        }
        m[i][k] = 0;

    }
}
}

int main(){
    for(int i=0;i<T;i++){
    m_reset(2000);
    struct timespec sts,ets;
    timespec_get(&sts, TIME_UTC);
    LU(2000);
    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec-sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec<0){
        dsec--;
        dnsec+=1000000000ll;
   }
    printf("平凡算法耗时： %ld.%09lds\n",dsec,dnsec);
    }
    for(int i=0;i<T;i++){
    struct timespec sts1,ets1;
    timespec_get(&sts1,TIME_UTC);
    neon_optimized(2000);
    timespec_get(&ets1,TIME_UTC);
    time_t dsec1 = ets1.tv_sec - sts1.tv_sec;
    long dnsec1 = ets1.tv_nsec - sts1.tv_nsec;
    if(dnsec1 < 0){
        dsec1--;
        dnsec1 += 1000000000ll;
    }
    printf("NEON优化后耗时： %ld.%09lds\n",dsec1,dnsec1);
    }
    
}
