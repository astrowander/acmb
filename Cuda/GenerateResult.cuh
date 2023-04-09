#include "./../Core/macros.h"

ACMB_CUDA_NAMESPACE_BEGIN

template<typename ChannelType>
void GeneratingResultKernel( const float* pMeans, ChannelType* pOutput, const size_t size );

ACMB_CUDA_NAMESPACE_END