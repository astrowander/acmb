#include "CudaBasic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ACMB_CUDA_NAMESPACE_BEGIN

static const size_t availMem = getCudaAvailableMemory();

void setToZero( DynamicArrayF& devArray )
{
    if ( devArray.size() == 0 )
        return;   
}

bool isCudaAvailable()
{
    int n;
    cudaError err = cudaGetDeviceCount( &n );
    if ( err != cudaError::cudaSuccess )
        return false;
    return n > 0;
}

size_t getCudaAvailableMemory()
{
    if ( !isCudaAvailable() )
        return 0;
    cudaSetDevice( 0 );
    size_t memFree = 0, memTot = 0;
    cudaMemGetInfo( &memFree, &memTot );
    // minus extra 128 MB
    return memFree - 128 * 1024 * 1024;
}

ACMB_CUDA_NAMESPACE_END