// 2维数据

#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <math.h>
#include "config.h"
#include "utils.h"


#define N 1048576
#define k 3
#define exp 1e-3

void random_data(float *arr, float *ct){

    // 生成N组随机数对
   srand((unsigned)time(NULL));
    for(int i=0;i<2*N;i++){
        arr[i] = 1.0*(rand()%RAND_MAX)/RAND_MAX *(20.0-0.0);
    }
    
    ct[0] = arr[0];
    ct[1] = arr[1];
    ct[2] = arr[N];
    ct[3] = arr[N+1];
    ct[4] = arr[2*N-2];
    ct[5] = arr[2*N-1];
}

__global__ void kmeans(float *data, float *kcen, int *symbols){
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x; //col
    const int tid_y = blockIdx.y * blockDim.y + threadIdx.y; //row
    const int threadId = tid_x+tid_y*blockDim.x*gridDim.x;
    float x = data[2*(threadId)];
    float y = data[2*(threadId)+1];

    float min = powf(x-kcen[0],2)+powf(y-kcen[1],2);
    int c = 0;
    
    for(int i=2;i<2*k;i+=2){
        float ds = powf(x-kcen[i],2)+powf(y-kcen[i+1],2);
        if(min > ds){
            min = ds;
            c = i/2;
        }
    }
    symbols[threadId] = c;
    // printf("threadID: %d\n", threadId);
}

__global__ void updateCen(float *data, int *symbols, float *kcen, float *new_kcen, int n, int kk){
    int tid = threadIdx.x; 
    // printf("tid: %d \n", tid);
    float sumx = 0.0;
    float sumy = 0.0;
    int cnt = 0;
    for(int i=0;i<n;i++){
        if(tid==symbols[i]){
            cnt += 1;
            sumx += data[2*i];
            sumy += data[2*i+1];
        }
    }

    new_kcen[2*tid] = sumx/cnt;
    new_kcen[2*tid+1] = sumy/cnt;
   
    // if(pow(new_kcen[2*tid]-kcen[2*tid], 2)+pow(new_kcen[2*tid+1]-kcen[2*tid+1], 2) > exp){
    //     kcen[2*tid] = new_kcen[2*tid];
    //     kcen[2*tid+1] = new_kcen[2*tid+1];
    // }

}

int main() {

    float *data_host = (float*)malloc(sizeof(float) * 2 * N);    
    float *kcen_host = (float*)malloc(sizeof(float) * 2 * k);
    float *new_kcen_host = (float*)malloc(sizeof(float) * 2 * k);
    float *sum_host = (float*)malloc(sizeof(float) * 2 * k);
    int *cnt_host = (int*)malloc(sizeof(int) * k);
    int *symbols_host = (int*)malloc(sizeof(int) * N);

    random_data(data_host, kcen_host);

    // printf("-----main:-----\n");

    // for(int i=0;i<2*N;i+=2){
    //     printf("%f %f\n", data_host[i], data_host[i+1]);
    // }

    // printf("\n");

    for(int i=0;i<2*k;i+=2){
        printf("%f %f\n", kcen_host[i], kcen_host[i+1]);
    }

    for(int i=0;i<3*k;i++){
        new_kcen_host[i] = 0;
    }

    float *data_device;
    float *kcen_device;
    float *new_kcen_device;
    int *symbols_device;


    CHECK(cudaMalloc((void **)&data_device, sizeof(float)*2*N));
    CHECK(cudaMemcpy(data_device, data_host, sizeof(float)*2*N, cudaMemcpyHostToDevice)); 

    CHECK(cudaMalloc((void **)&kcen_device, sizeof(float)*2*k));
    CHECK(cudaMemcpy(kcen_device, kcen_host, sizeof(float)*2*k, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void **)&new_kcen_device, sizeof(float)*2*k));
    CHECK(cudaMemcpy(new_kcen_device, new_kcen_host, sizeof(float)*2*k, cudaMemcpyHostToDevice));
    
    CHECK(cudaMalloc((void **)&symbols_device, sizeof(int)*N));
    CHECK(cudaMemcpy(symbols_device, symbols_host, sizeof(int)*N, cudaMemcpyHostToDevice));

    char flag = 1;
    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 32;
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim(divup(sqrt(N), BLOCK_DIM_X), divup(sqrt(N), BLOCK_DIM_Y));

    dim3 blockDim1(k, 1);
    dim3 gridDim1(1, 1);

    

    int j = 0;
    auto sta = clock();
    while(flag > 0){
        flag = 0 ;

        CHECK(cudaMemcpy(symbols_device, symbols_host, sizeof(int)*N, cudaMemcpyHostToDevice));

        kmeans << < gridDim, blockDim >> > (data_device, kcen_device, symbols_device);
        updateCen << < gridDim1, blockDim1 >> > (data_device, symbols_device, kcen_device, new_kcen_device, N, k);

        // for(int i=0;i<N;i++){
        //     cnt_host[symbols_host[i]] += 1;
        //     sum_host[2*symbols_host[i]] += data_host[2*i];
        //     sum_host[2*symbols_host[i]+1] += data_host[2*i+1];
        // }

        // for(int i=0;i<k;i++){
        //     sum_host[2*i] = sum_host[2*i]/cnt_host[i];
        //     sum_host[2*i+1] = sum_host[2*i+1]/cnt_host[i];
        // }
        cudaDeviceSynchronize();

        CHECK(cudaMemcpy(new_kcen_host, new_kcen_device, sizeof(float)*2*k, cudaMemcpyDeviceToHost));
        // printf("-------------epchos: %d-------------\n", j);
        for(int i=0;i<2*k;i+=2){
            if(pow(new_kcen_host[i]-kcen_host[i], 2)+pow(new_kcen_host[i+1]-kcen_host[i+1], 2) > exp){
                flag += 1;
            }
            kcen_host[i] = new_kcen_host[i];
            kcen_host[i+1] = new_kcen_host[i+1];
            // printf("%f %f\n", kcen_host[i], kcen_host[i+1]);
        }

        // printf("------------kcen: %d.---------------\n", j);
        // for(int i=0;i<2*k;i+=2){
        //     printf("%f %f\n", kcen_host[i], kcen_host[i+1]);
        // }

        // CHECK(cudaMemcpy(kcen_device, kcen_host, sizeof(float)*2*k, cudaMemcpyHostToDevice));
        j++;
    }
    auto end = clock();
    auto time = (float)(end - sta)/CLOCKS_PER_SEC;
    printf("USED TIME: %lf \n", time);
    

}
