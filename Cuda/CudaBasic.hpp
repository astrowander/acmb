#include "cuda_runtime.h"
#include <numeric>
#include <stdexcept>
ACMB_CUDA_NAMESPACE_BEGIN

template<typename T>
DynamicArray<T>::DynamicArray( size_t size )
{
    resize( size );
    if constexpr ( std::is_trivial_v<T> )
        cudaMemset( data_, 0, size_ * sizeof( T ) );
}

template<typename T>
template<typename U>
DynamicArray<T>::DynamicArray( const std::vector<U>& vec )
{
    fromVector( vec );
}

template<typename T>
template<typename U>
DynamicArray<T>::DynamicArray( const std::vector<std::vector<U>>& vec )
{
    fromVectors( vec );
}

template<typename T>
DynamicArray<T>::~DynamicArray()
{
    resize( 0 );
}

template<typename T>
template<typename U>
inline void DynamicArray<T>::fromVector( const std::vector<U>& vec )
{
    static_assert ( sizeof( T ) == sizeof( U ), "size of types must be equal" );
    resize( vec.size() );
    if ( cudaMemcpy( data_, vec.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice ) != cudaSuccess )
        throw std::runtime_error( "unable to copy memory from host to GPU" );
}

template<typename T>
template<typename U>
inline void DynamicArray<T>::fromVectors( const std::vector<std::vector<U>>& vec )
{
    static_assert ( sizeof( T ) == sizeof( U ), "size of types must be equal" );

    size_t totalSize = 0;
    for ( size_t i = 0; i < vec.size(); ++i )
        totalSize += vec[i].size();

    resize( totalSize );
    size_t offset = 0;
    for ( size_t i = 0; i < vec.size(); ++i )
    {
        if ( cudaMemcpy( (void*)( data_ + offset ), vec[i].data(), vec[i].size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess )
            throw std::runtime_error( "unable to copy memory from host to GPU" );

        offset += vec[i].size();
    }
}

template<typename T>
void DynamicArray<T>::resize( size_t size )
{
    if ( size == size_ )
        return;
    if ( size_ != 0 )
        cudaFree( data_ );

    size_ = size;
    if ( size_ != 0 )
    {
        const auto errCode = cudaMalloc( ( void** ) &data_, size_ * sizeof( T ) );
        if (  errCode!= cudaSuccess )
            throw std::runtime_error( "unable to allocate memory on GPU" );

        if constexpr ( std::is_trivial_v<T> )
            if ( cudaMemset( data_, 0, size_ * sizeof( T ) != cudaSuccess ) )
                 throw std::runtime_error( "unable to set memory on GPU" );
    }
}

template<typename T>
template<typename U>
void DynamicArray<T>::toVector( std::vector<U>& vec ) const
{
    static_assert ( sizeof( T ) == sizeof( U ), "size of types must be equal" );
    vec.resize( size_ );
    cudaMemcpy( vec.data(), data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost );
}

ACMB_CUDA_NAMESPACE_END