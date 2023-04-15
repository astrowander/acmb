#include "AddBitmap.h"
#include "CudaUtils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ACMB_CUDA_NAMESPACE_BEGIN

template<typename ChannelType>
__global__ void AddBitmapKernel( const ChannelType* pPixels, float* pMeans, float* pDevs, uint16_t* pCounts, size_t size )
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    auto& mean = pMeans[index];
    auto& dev = pDevs[index];
    auto& n = pCounts[index];
    const auto& channel = pPixels[index];

    const auto sigma = sqrt( dev );
    constexpr auto kappa = 3.0;

    if ( n <= 5 || fabs( mean - channel ) < kappa * sigma )
    {
        dev = n * ( dev + ( channel - mean ) * ( channel - mean ) / ( n + 1 ) ) / ( n + 1 );
        mean = Clamp( ( n * mean + channel ) / ( n + 1 ), 0.0f, float( MaxValue<ChannelType>() ) );
        ++n;
    }
}

template<typename ChannelType>
void AddBitmapHelper( const ChannelType* pPixels, float* pMeans, float* pDevs, uint16_t* pCounts, size_t size )
{
    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    AddBitmapKernel<ChannelType> << <numBlocks, maxThreadsPerBlock >> > ( pPixels, pMeans, pDevs, pCounts, size );
    /*for ( size_t index = 0; index < size; ++index )
    {
        auto& mean = pMeans[index];
        auto& dev = pDevs[index];
        auto& n = pCounts[index];
        const auto& channel = pPixels[index];

        const auto sigma = sqrt( dev );
        constexpr auto kappa = 3.0;

        if ( n <= 5 || fabs( mean - channel ) < kappa * sigma )
        {
            dev = n * ( dev + ( channel - mean ) * ( channel - mean ) / ( n + 1 ) ) / ( n + 1 );
            mean = Clamp( ( n * mean + channel ) / ( n + 1 ), 0.0f, float( MaxValue<ChannelType>() ) );
            ++n;
        }
    }*/
};

template void AddBitmapHelper<>( const uint8_t*, float*, float*, uint16_t*, size_t );
template void AddBitmapHelper<>( const uint16_t*, float*, float*, uint16_t*, size_t );


ACMB_CUDA_NAMESPACE_END
