#include "GenerateResult.h"
#include "GenerateResult.cuh"
#include "CudaUtils.cuh"
#include "CudaBasic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ACMB_CUDA_NAMESPACE_BEGIN

template<typename ChannelType>
__global__ void kernel( const float* pMeans, ChannelType* pOutput, const size_t size )
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    pOutput[index] = ChannelType( pMeans[index] );
}

template<typename ChannelType>
void GeneratingResultKernel( const float* pMeans, ChannelType* pOutput, const size_t size )
{
    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    kernel<< <numBlocks, maxThreadsPerBlock >> > ( pMeans, pOutput, size );
    //for ( size_t index = 0; index < size; ++index )
      //  pOutput[index] = ChannelType( pMeans[index] );
}

template void GeneratingResultKernel<uint8_t>( const float*, uint8_t*, const size_t );
template void GeneratingResultKernel<uint16_t>( const float*, uint16_t*, const size_t );

ACMB_CUDA_NAMESPACE_END
