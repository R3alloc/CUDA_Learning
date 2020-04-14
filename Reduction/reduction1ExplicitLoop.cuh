/*
 *
 * reduction1ExplicitLoop.cuh
 *
 */

//
// reads N ints and writes an intermediate sum per block
// blockDim.x must be a power of 2!
//
//注意：blockDim.x一定是2的整数次方，但是没有说
//输入的数组应当是一个一维数组
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void
Reduction1_kernel( int *out, const int *in, size_t N )
{
	//这个数组的大小与blockSize有关（也就是blockDim.x）
	//注意这个数组在这里定义的时候虽然没有指定大小，但是在调用这个kernel的时候有一个核函数参数就是用来控制kernel内部使用共享内存的大小。
    extern __shared__ int sPartials[];	
    int sum = 0;
	//tid是当前线程在当前block中的索引
    const int tid = threadIdx.x;
	//i是当前线程在所有线程中的索引
	//i的步长是grid当中的block数量*block中线程的数量
	//in[]存储在全局内存中 输入指针被恰当地对齐，由这段代码发起的全部内存事务将被合并，这将最大限度地提高内存带宽。
	//也就是说一个cuda线程要去多次访问全局内存，然后把这些值加起来
    for ( size_t i = blockIdx.x*blockDim.x + tid;
          i < N;
          i += blockDim.x*gridDim.x ) 
	{
        sum += in[i];
    }
	
	//每个线程把它得到的累计值写入共享内存
    sPartials[tid] = sum;
	//在执行对数步长的规约前进行同步操作
    __syncthreads();

	//blockSize必须是2的整数次方的原因在这里：每一轮都只有上一次一半的线程还在工作
	//对于共享内存中的值 执行对数步长的规约操作
	//共享内存中后半部分的值被添加到前半部分的值上，
	//假设blockDim.x == 1024，则第一轮activeThreads=512
    for ( int activeThreads = blockDim.x>>1; 
              activeThreads; 
              activeThreads >>= 1 ) //>>是二进制右移运算符 等价于整除2
									//>>=是右移且赋值运算符 也就是activeThreads = activeThreads>>1
	{
        if ( tid < activeThreads ) 
		{
            sPartials[tid] += sPartials[tid+activeThreads];
        }
		//每一轮加完之后要线程同步
        __syncthreads();
    }
	
	//每个block的0号线程存储一个结果，一共有numBlocks个线程，所以存储了这么多个结果。
    if ( tid == 0 ) 
	{
        out[blockIdx.x] = sPartials[0];
    }
}

//这里调用两遍kernel函数是必须的
void
Reduction1( int *answer, int *partial, 
            const int *in, size_t N, 
            int numBlocks, int numThreads )
{
    unsigned int sharedSize = numThreads*sizeof(int);
	//第一次的结果partial只是一个中间结果，并未完全做和
    Reduction1_kernel<<< 
        numBlocks, numThreads, sharedSize>>>( 
            partial, in, N );
	//第二次结果answer才是最终的计算结果。
    Reduction1_kernel<<< 
        1, numThreads, sharedSize>>>( 
            answer, partial, numBlocks );
}
