#pragma once
#include "./../Core/macros.h"
#include <vector>
#include <cstdint>

using std::size_t;

ACMB_CUDA_NAMESPACE_BEGIN

// This struct is present to simplify GPU memory control
template<typename T>
class DynamicArray
{
public:
    DynamicArray() = default;
    // malloc given size on GPU
    DynamicArray( size_t size );
    // copy given vector to GPU
    template<typename U>
    DynamicArray( const std::vector<U>& vec );

    template<typename U>
    DynamicArray( const std::vector<std::vector<U>>& vec );
    // free this array from GPU (if needed)
    ~DynamicArray();

    DynamicArray( const DynamicArray& ) = delete;
    DynamicArray( DynamicArray&& other )
    {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    DynamicArray& operator=( DynamicArray&& other )
    {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }
    DynamicArray& operator=( const DynamicArray& other ) = delete;

    // copy given vector to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    template <typename U>
    void fromVector( const std::vector<U>& vec );

    template <typename U>
    void fromVectors( const std::vector<std::vector<U>>& vec );

    // copy this GPU array to given vector
    template <typename U>
    void toVector( std::vector<U>& vec ) const;

    // resize (free and malloc againg if size inconsistent) this GPU array (if size == 0 free it (if needed))
    void resize( size_t size );

    // pointer to GPU array
    T* data()
    {
        return data_;
    }
    // const pointer to GPU array
    const T* data() const
    {
        return data_;
    }
    // size of GPU array
    size_t size() const
    {
        return size_;
    }

private:
    T* data_{ nullptr };
    size_t size_{ 0 };
};

using DynamicArrayU8 = DynamicArray<uint8_t>;
using DynamicArrayU16 = DynamicArray<uint16_t>;
using DynamicArrayU32 = DynamicArray<uint32_t>;
using DynamicArrayF = DynamicArray<float>;

// Sets all float values of GPU array to zero
inline void setToZero( DynamicArrayF& devArray )
{
    if ( devArray.size() == 0 )
        return;
}


ACMB_CUDA_NAMESPACE_END

#include "CudaBasic.hpp"
