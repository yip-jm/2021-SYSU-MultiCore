#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<iostream>
#include "config.h"
#include <fstream> // c++文件操作
#include <iomanip> // 设置输出格式
#define K 4
#define epsilon 0.0000001//误差
using namespace std;
/*
串行版本kmeans
数据用一个矩阵存储。矩阵的每个行向量是一个数据。
*/


/*
函数功能：加载数据
数据存储在二进制文件中，存储格式是先是两个正整数，表示数据个数和数据维度
Load a matrix from [stream] into [sample].
*/
bool loadMatrix(FILE* stream, int* width, int* height, float** sample) {
    if (fread(width, sizeof(int), 1, stream) == 0) {
        // eof
        return false;
    }
    fread(height, sizeof(int), 1, stream);
    int size = (*width) * (*height);

    *sample = (float*)malloc(sizeof(float) * size);
    if (*sample == NULL) {
        printf("failed to allocate\n");
        return false;
    }

    fread(*sample, sizeof(float), size, stream);

    return true;
}


/*
函数功能：随机选取k个中心
input:  需要选取的中心数k
        n数据个数
        center随机选出来的数据
*/
void kCenters(int k, int n, float* center, int dim, float* data){
    srand((unsigned)time(NULL));
    int j;
    int* tmpK = (int*)malloc(sizeof(int)*k);
    int i = 0;
    bool e = false;
    while(i<k){
        //保证每次的j都是不一样的
        j = rand() % n;
        if( i == 0 ){
            tmpK[i] = j;
            printf("j = %d\n",j);
        }
        else{
            e = false;
            for(int m = 0; m < i; m++){
                if(j == tmpK[m])
                    e = true;
            }
            if(e == true) continue;
            tmpK[i] = j;
            printf("j = %d\n",j);
        }
        
        for(int o = 0; o < dim; o++){
            center[i*dim+o] = data[j*dim+o];
            printf("data[%d]=%f ",j*dim+o,data[j*dim+o]);
        }
        printf("\n");
        i++;
    }
}

//计算两个向量的距离
/*
static __global__ void getDistance(float* a, float* b, int dim, float* res){
    int i;
    float sum = 0.0;
    //float res = 0;
    res[0] = 0;
    for( int i = 0; i < dim; i++ ){
        sum += pow(a[i]-b[i],2);
    }

    res[0] += sqrt(sum);
}
*/

//计算所有center的距离
static __global__ void GetDistance(float* a, float* b, int dim, float* dis_d, int k){
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; //该线程对应的行坐标
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //该线程对应的列坐标
    float res = 0;
    float sum = 0.0;
    //CHECK(cudaMemset(dis_d, 0, 1));
    if(idy == 0 && idx < K){
        for( int i = 0; i < dim; i++ ){
            sum += pow(a[idx*dim+i]-b[idx*dim+i],2 );
            printf("sum = pow(a[%d]-b[%d],2) = pow(%f,%f,2) = %f\n",idx*dim+i, idx*dim+i, a[idx*dim+i], b[idx*dim+i], sum);
        }
        res += sqrt(sum);
        atomicAdd(dis_d, res);
    }
    
}

/*
函数功能：计算数据点到每个中心的距离，选取距离最短的中心点作为其聚类中心
input: 
        center:存储中心点的数组
        dim: 数据的维度
        mydata: 当前数据点
        k: 中心点的个数
        n: 数据个数
        symbols: 存储每个点所属的中心
return:
        距离该样本点最近的中心下标
*/
//每个线程负责一个数据点的计算
static __global__ void chooseCenter(float* center, int dim, float* mydata, int k, int n, int* symbols, int width, int height){
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; //该线程对应的行坐标
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //该线程对应的列坐标
    if( idy < height && idx < width ){
        int j = idy * width + idx;
        int index = 0;
        float dis = 0;
        float mindis = 100860;
        for(int i = 0; i < k ; i++){
            float sum = 0.0;
            dis = 0;
            for( int m = 0; m < dim; m++ ){
                sum += pow(mydata[j*dim+m]-center[i*dim+m],2);
            }
            dis += sqrt(sum);
            if( dis < mindis ){
                mindis = dis; 
                index = i;
            }
        }
        symbols[j] = index;
        //printf("idx is %d = %d\n",j, index);
    }   
}




/*
函数功能：实现两个向量相加
Input：vector1， vector2
return： vector1 = vector1 + vector2
*/
/*
static __global__ void AddVec(float* vec1, float* vec2, int dim){
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; //该线程对应的行坐标
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //该线程对应的列坐标
    if( idy == 0 && idx == 0  )
        for(int i = 0; i < dim; i++){
            vec1[i] = vec1[i] + vec2[i];
        }
}
*/
/*
函数功能：更新聚类中心
Input: 
        mydata为数据信息
        center为中心点信息
        symbols为各数据点所属的类别标记
        k为类别个数
        n为数据个数
        dim为每个数据的维度
*/
static __global__ void updateCenter(float* mydata, float* center, int* symbols, int k, int n, int dim){
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; //该线程对应的行坐标
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //该线程对应的列坐标
    if(idy == 0 && idx < k){
        int vec_num;//记录第k个类有几个向量，用于最后求均值    
        vec_num = 0;
        for(int j = 0; j < dim; j++){
            center[idx*dim+j] = 0;
        }
        
        for(int i = 0; i < n; i++){
            if( symbols[i] == idx )
                vec_num ++;
            for(int o = 0; o < dim; o++)
                center[idx*dim+o] = center[idx*dim+o] + mydata[i*dim+o];
        }
        //算平均
        for(int j = 0; j < dim; j++){
            center[idx*dim+j] /= vec_num; 
        }
        printf("idx = %d, %f %f\n", idx, center[idx*dim], center[idx*dim+1]);
    }
}

/*
函数功能：向量赋值 vec1 = vec2
*/
static __global__ void equ(float* vec1, float* vec2, int dim, int height, int width){
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; //该线程对应的行坐标
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //该线程对应的列坐标
    if( idy < height && idx < width ){
        for( int j = 0; j < dim; j++){
            vec1[(idy * width + idx) * dim + j] = vec2[ (idy * width + idx) * dim + j];
        }
    }
}


int main(int argc, char* argv[]){
    if( argc == 3 ){
        inputPath = argv[1];
        outputPath = argv[2];
    }
    //Open the input file
    FILE* stream = fopen(inputPath, "rb");
    if(stream == NULL){
        printf("failed to open the data file %s\n", inputPath);
        return -1;
    }

    //Read in and process the input matrix one-by-one
    int dim, n, size;
    float *data;
    clock_t start, end;
    loadMatrix(stream, &dim, &n, &data);
    size = K * dim;
    printf("dim = %d\n",dim);
    printf("n = %d\n",n);

    float*center;
    //float* precenter;//更新前的center信息，用于比较center是否收敛
    //precenter = (float*)malloc(K*dim*sizeof(float));
    center = (float*)malloc(size*sizeof(float));
    
    int* symbols = (int*)malloc(n*sizeof(int));
    float* center_d;
    float* precenter_d;
    float *input_d;
    int* symbols_d;
    int width, height;
    width = height = sqrt(n);

    CHECK(cudaMalloc((void**)&center_d, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&precenter_d, sizeof(float) * size));
    CHECK(cudaMalloc((void**)&symbols_d, sizeof(int) * n));
    CHECK(cudaMalloc((void**)&input_d, sizeof(float) * n * dim));
    CHECK(cudaMemcpy( input_d, data, sizeof(float) * n * dim, cudaMemcpyHostToDevice));

    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 32;

    start = clock();
    float time = 0;

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim(divup(width, BLOCK_DIM_X), divup(height, BLOCK_DIM_Y));
    
    //第一步：随机选取k个数据中心
    kCenters(K, n, center, dim, data);
    CHECK(cudaMemcpy( center_d, center, sizeof(float) * size, cudaMemcpyHostToDevice));
    
    equ<<<gridDim, blockDim>>>(precenter_d, center_d, dim, width, height);
    
    
    //第二步：分别计算每个数据点到每个中心的距离，选取距离最短的中心点作为其聚类中心
    chooseCenter<<<gridDim, blockDim>>>(center_d, dim, input_d, K, n, symbols_d, width, height);
    cudaDeviceSynchronize();
    
    
    CHECK(cudaMemcpy( symbols, symbols_d, sizeof(int) * n, cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++){
        cout << symbols[i] << ' ';
     }
    cout << endl;
    
    //第三步：利用目前得到的聚类重新计算中心点
    updateCenter<<<gridDim, blockDim>>>(input_d, center_d, symbols_d, K, n, dim);
    cudaDeviceSynchronize();
    
    CHECK(cudaMemcpy( center, center_d, sizeof(float) * size, cudaMemcpyDeviceToHost));
    cout << "-------------------step 3--------------------------" << endl;
    for(int i = 0; i < K; i++){
        cout << center[i*dim] << ' ';
        cout << center[i*dim+1] << endl;
     }
    cout << endl;
    
    
    
    //第四步：重复第二步和第三步，直到收敛
    float dis = 0;
    float* dis_d;
    CHECK(cudaMalloc((void**)&dis_d, sizeof(float) ));

    CHECK(cudaMemset(dis_d, 0, 1));
    GetDistance<<<gridDim, blockDim>>>(precenter_d, center_d, dim, dis_d, K);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(&dis, dis_d, sizeof(float), cudaMemcpyDeviceToHost));
    printf("line 307, dis = %f\n",dis);
    while(dis > epsilon){
        printf("dis = %f\n",dis);
        //dis = 0;
        //for(int i = 0; i < K; i++){
        //    dis += getDistance(&precenter_d[i*dim],&center_d[i*dim],dim);
        //}
        CHECK(cudaMemset(dis_d, 0, 1));
        GetDistance<<<gridDim, blockDim>>>(precenter_d, center_d, dim, dis_d, K);
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(&dis, dis_d, sizeof(float)*1, cudaMemcpyDeviceToHost));
        
        equ<<<gridDim, blockDim>>>(precenter_d, center_d, dim, width, height);
        //第二步：分别计算每个数据点到每个中心的距离，选取距离最短的中心点作为其聚类中心
        chooseCenter<<<gridDim, blockDim>>>(center_d, dim, input_d, K, n, symbols_d, width, height);
        cudaDeviceSynchronize();
        //第三步：利用目前得到的聚类重新计算中心点
        updateCenter<<<gridDim, blockDim>>>(input_d, center_d, symbols_d, K, n, dim);
        cudaDeviceSynchronize();
    }
    //将结果拷贝回主机端
    CHECK(cudaMemcpy(center, center_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(symbols, symbols_d, sizeof(int)*K, cudaMemcpyDeviceToHost));
    //释放内存
    CHECK(cudaFree(center_d));
    CHECK(cudaFree(precenter_d));
    CHECK(cudaFree(symbols_d));
    CHECK(cudaFree(input_d));
    end = clock();//计时结束  
    time = (float)(end - start)/CLOCKS_PER_SEC;
    printf("Time of Cuda: %f\n",time);
    //将聚类结果保存在txt文件中
    ofstream out_txt_file;
    out_txt_file.open("./result.txt", ios::out | ios::trunc);
    //先将k个中心保存在result.txt文件中
    out_txt_file << K << endl;
    for(int i = 0; i < K; i++){
        out_txt_file << setprecision(4) << center[i*dim] << ' ' <<  center[i*dim+1]  << endl;
    }
    //再将数据的分类结果保存
    out_txt_file << n << endl;
    for(int i = 0; i < n; i++){
        out_txt_file << symbols[i] << endl;
    }
    out_txt_file.close();
    printf("dis = %f\n",dis);
    free(center);
    free(symbols);
    free(data);
    fclose(stream);
}
