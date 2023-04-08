#include "cuda_runtime.h"
ACMB_CUDA_NAMESPACE_BEGIN

template<typename T>
DynamicArray<T>::DynamicArray( size_t size )
{
    resize( size );
    if constexpr ( std::is_trivial_v<T> )
        cudaMemset( data_, 0, size_ * sizeof( T ) );
    else
        for ( int i = 0; i < size_; ++i )
            data_[i] = std::move( T{} );
}

template<typename T>
DynamicArray<T>::DynamicArray( const std::vector<T>& vec )
{
    fromVector( vec );
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
    cudaMemcpy( data_, vec.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
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
        cudaMalloc( ( void** ) &data_, size_ * sizeof( T ) );
        if constexpr ( std::is_trivial_v<T> )
            cudaMemset( data_, 0, size_ * sizeof( T ) );
        else
            for ( int i = 0; i < size_; ++i )
                data_[i] = std::move( T{} );
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