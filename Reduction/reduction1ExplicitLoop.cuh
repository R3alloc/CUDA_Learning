/*
 *
 * reduction1ExplicitLoop.cuh
 *
 */

//
// reads N ints and writes an intermediate sum per block
// blockDim.x must be a power of 2!
//
//ע�⣺blockDim.xһ����2�������η�������û��˵
//���������Ӧ����һ��һά����
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void
Reduction1_kernel( int *out, const int *in, size_t N )
{
	//�������Ĵ�С��blockSize�йأ�Ҳ����blockDim.x��
	//ע��������������ﶨ���ʱ����Ȼû��ָ����С�������ڵ������kernel��ʱ����һ���˺�������������������kernel�ڲ�ʹ�ù����ڴ�Ĵ�С��
    extern __shared__ int sPartials[];	
    int sum = 0;
	//tid�ǵ�ǰ�߳��ڵ�ǰblock�е�����
    const int tid = threadIdx.x;
	//i�ǵ�ǰ�߳��������߳��е�����
	//i�Ĳ�����grid���е�block����*block���̵߳�����
	//in[]�洢��ȫ���ڴ��� ����ָ�뱻ǡ���ض��룬����δ��뷢���ȫ���ڴ����񽫱��ϲ����⽫����޶ȵ�����ڴ����
	//Ҳ����˵һ��cuda�߳�Ҫȥ��η���ȫ���ڴ棬Ȼ�����Щֵ������
    for ( size_t i = blockIdx.x*blockDim.x + tid;
          i < N;
          i += blockDim.x*gridDim.x ) 
	{
        sum += in[i];
    }
	
	//ÿ���̰߳����õ����ۼ�ֵд�빲���ڴ�
    sPartials[tid] = sum;
	//��ִ�ж��������Ĺ�Լǰ����ͬ������
    __syncthreads();

	//blockSize������2�������η���ԭ�������ÿһ�ֶ�ֻ����һ��һ����̻߳��ڹ���
	//���ڹ����ڴ��е�ֵ ִ�ж��������Ĺ�Լ����
	//�����ڴ��к�벿�ֵ�ֵ����ӵ�ǰ�벿�ֵ�ֵ�ϣ�
	//����blockDim.x == 1024�����һ��activeThreads=512
    for ( int activeThreads = blockDim.x>>1; 
              activeThreads; 
              activeThreads >>= 1 ) //>>�Ƕ�������������� �ȼ�������2
									//>>=�������Ҹ�ֵ����� Ҳ����activeThreads = activeThreads>>1
	{
        if ( tid < activeThreads ) 
		{
            sPartials[tid] += sPartials[tid+activeThreads];
        }
		//ÿһ�ּ���֮��Ҫ�߳�ͬ��
        __syncthreads();
    }
	
	//ÿ��block��0���̴߳洢һ�������һ����numBlocks���̣߳����Դ洢����ô��������
    if ( tid == 0 ) 
	{
        out[blockIdx.x] = sPartials[0];
    }
}

//�����������kernel�����Ǳ����
void
Reduction1( int *answer, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    unsigned int sharedSize = numThreads*sizeof(int);
	//��һ�εĽ��partialֻ��һ���м�������δ��ȫ����
    Reduction1_kernel<<< 
        numBlocks, numThreads, sharedSize>>>( 
            partial, in, N );
	//�ڶ��ν��answer�������յļ�������
    Reduction1_kernel<<< 
        1, numThreads, sharedSize>>>( 
            answer, partial, numBlocks );
}
