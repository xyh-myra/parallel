#include<iostream>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>
using namespace std;
#define N 1000
#define NUM_THREADS 7
float** A = NULL;

struct timespec sts, ets;
time_t dsec;
long dnsec;


struct threadParam_t {    
    int k;
    int t_id;
};

void A_init() {     
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 1000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}
void A_initAsEmpty() {
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
        memset(A[i], 0, N * sizeof(float));
    }

}

void deleteA() {
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete A;
}

void LU() {   
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}


void LU_mpi(int argc, char* argv[]) {  //�黮��
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << rank << " of " << total << " created" << endl;
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    cout << "rank " << rank << " from " << begin << " to " << end << endl;
    if (rank == 0) {  
        A_init();
        cout << "initialize success:" << endl;
        print(A);
        cout << endl << endl;
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD); 
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) { 
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int src;
            if (k < N / total * total)
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI���黮�ֺ�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        print(A);
    }
    MPI_Finalize();
}

void LU_mpi_plus(int argc, char* argv[]) {  //�Ż��黮��
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << rank << " of " << total << " created" << endl;
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    cout << "rank " << rank << " from " << begin << " to " << end << endl;
    if (rank == 0) {  
        A_init();
        cout << "initialize success:" << endl;
        print(A);
        cout << endl << endl;
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }
    if (rank == 2) {
        cout << rank << " : " << endl;
        print(A);
        cout << endl << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = rank + 1; j < total; j++) { 

                MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
            if (k == end - 1)
                break;
        }
        else {
            int src = k / (N / total);
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI���黮���Ż���ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        print(A);
    }
    MPI_Finalize();
}


void LU_mpi_circle(int argc, char* argv[]) {  //ѭ������
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << rank << " of " << total << " created" << endl;
    if (rank == 0) {  
        A_init();
        cout << "initialize success:" << endl;
        print(A);
        cout << endl << endl;
        for (j = 1; j < total; j++) {
            for (i = j; i < N; i += total) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = rank; i < N; i += total) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if (k % total == rank) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) { //
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int src = k % total;

            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % total != rank)
            begin++;
        for (i = begin; i < N; i += total) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI,ѭ�����ֺ�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        print(A);
    }
    MPI_Finalize();
}

void LU_mpi_withExtra(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int extra = -1;
    bool ifExtraDone = true;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << rank << " of " << total << " created" << endl;
    int begin = N / total * rank;
    int end = N / total * (rank + 1);
    if (rank < N % total) {
        extra = N / total * total + rank;
        ifExtraDone = false;
    }
    if (rank == 0) {  
        A_init();
        cout << "initialize success:" << endl;
        print(A);
        cout << endl << endl;
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
        if (extra != -1) {
            for (i = 1; i < N % total; i++) {
                MPI_Send(&A[N / total * total + i][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);  
            }
        }
    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
        if (extra != -1) {
            MPI_Recv(&A[extra][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end) || (k == extra)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) {
                if (j != rank) {
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            int src;
            if (k < N / total * total)
                src = k / (N / total);
            else
                src = k - (N / total * total);

            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);

        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
        if (!ifExtraDone) {          
            for (j = k + 1; j < N; j++) {
                A[extra][j] = A[extra][j] - A[extra][k] * A[k][j];
            }
            A[extra][k] = 0;
            if (extra == k + 1) {
                ifExtraDone = true;
            }
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);	
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI��β�����ֺ�ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        print(A);
    }
    MPI_Finalize();
}

double LU_mpi_async(int argc, char* argv[]) {  //������ͨ��
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); 

    }
    else {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank]; 
            for (j = rank + 1; j < total; j++) { 

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; 
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);       
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI���黮��+��������ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}

void test(void(*func)()) {
    A_init();
    timespec_get(&sts, TIME_UTC);
    func();
    timespec_get(&ets, TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if (dnsec < 0) {
        dsec--;
        dnsec += 1000000000ll;
    }
}
double LU_mpi_pipeline(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {
        A_init();

        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];
            for (j = rank + 1; j < total; j++) {

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break;
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("ƽ��MPI����ˮ���Ż���ʱ��%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}
int main(int argc, char* argv[]) {

    LU_mpi(argc, argv);
     test(LU);
     printf("ƽ���㷨���к�ʱ�� %ld.%09lds\n", dsec, dnsec);
     deleteA();


}
