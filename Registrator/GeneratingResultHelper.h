#ifndef GENERATINGRESULTHELPER_H
#define GENERATINGRESULTHELPER_H
#include "./stacker.h"

#include <xmmintrin.h>
#include <emmintrin.h>

#define SSE_STRIDE 4
#define SSE_STRIDE_DOUBLE 8
#define SSE_STRIDE_TRIPLE 12
#define SSE_STRIDE_QUAD 16

template<typename ChannelType>
void ConvertToUint(ChannelType* mO, float* mI, int numRows, int numCols, int numColsPad, float scalingFctr)
{
    //float* ptrInputImage;
    //int* ptrOutputImage;

    //__m128  floatPx1, floatPx2, floatPx3, floatPx4;
    //__m128  scalingFactor;
    //__m128i uint32Px1, uint32Px2, uint32Px3, uint32Px4;
    //__m128i uint16Px1, uint16Px2;
    //__m128i uint8Px1, uint8Px2;
   // __m128i* ptrOutputImageSse;

    auto numColsQuadPack = numCols - (numCols % SSE_STRIDE_QUAD);

    auto scalingFactor = _mm_set1_ps(scalingFctr);

    //#pragma omp parallel for private(jj, ptrInputImage, ptrOutputImage, floatPx1, floatPx2, floatPx3, floatPx4, uint16Px1, uint16Px2, uint16Px3, uint16Px4, uint8Px1, uint8Px2)
    for (int ii = 0; ii < numRows; ii++) {
        auto ptrInputImage = &mI[ii * numColsPad];
        auto ptrOutputImageSse = (__m128i*)(&mO[ii * numColsPad]);
        for (int jj = 0; jj < numColsQuadPack; jj += SSE_STRIDE_QUAD) {
            // SSE Pack is 4 Floats (4 * 32 Byte) -> 16 UINT8 (16 * 1 Byte)
            // Hence loading 16 Floats which will be converted into 16 UINT8

            auto floatPx1 = _mm_loadu_ps(ptrInputImage);
            auto floatPx2 = _mm_loadu_ps(ptrInputImage + SSE_STRIDE);
            auto floatPx3 = _mm_loadu_ps(ptrInputImage + SSE_STRIDE_DOUBLE);
            auto floatPx4 = _mm_loadu_ps(ptrInputImage + SSE_STRIDE_TRIPLE);

            ptrInputImage += SSE_STRIDE_QUAD;

            // _mm_cvtps_epi32 - Rounds to nearest integer
            // _mm_cvttps_epi32 - Truncates (Rounding towards zero)
            auto uint32Px1 = _mm_cvtps_epi32(_mm_mul_ps(floatPx1, scalingFactor)); // Converts the 4 SP FP values of a to 4 Signed Integers (32 Bit).
            auto uint32Px2 = _mm_cvtps_epi32(_mm_mul_ps(floatPx2, scalingFactor));
            auto uint32Px3 = _mm_cvtps_epi32(_mm_mul_ps(floatPx3, scalingFactor));
            auto uint32Px4 = _mm_cvtps_epi32(_mm_mul_ps(floatPx4, scalingFactor));
            // See Intel Miscellaneous Intrinsics (https://software.intel.com/en-us/node/695374)
            auto uint16Px1 = _mm_packs_epi32(uint32Px1, uint32Px2); // Saturating and packing 2 of 4 Integers into 8 of INT16
            auto uint16Px2 = _mm_packs_epi32(uint32Px3, uint32Px4); // Saturating and packing 2 of 4 Integers into 8 of INT16

            if constexpr (std::is_same_v<ChannelType, uint8_t>)
            {
                auto uint8Px1 = _mm_packus_epi16(uint16Px1, uint16Px2); // Saturating and packing 2 of 8 INT16 into 16 of UINT8
                _mm_storeu_si128(ptrOutputImageSse++, uint8Px1); // Storing 16 UINT8, Promoting the pointer
            }
            else
            {
                _mm_storeu_si128(ptrOutputImageSse++, uint16Px1);
                _mm_storeu_si128(ptrOutputImageSse++, uint16Px2);
            }

        }

        auto ptrOutputImage = (int*)(&mO[(ii * numColsPad) + numColsQuadPack]);

        for (int jj = numColsQuadPack; jj < numCols; jj += SSE_STRIDE) {

            auto floatPx1 = _mm_loadu_ps(ptrInputImage);

            ptrInputImage += SSE_STRIDE;

            auto uint32Px1 = _mm_cvtps_epi32(_mm_mul_ps(floatPx1, scalingFactor));

            auto uint16Px1 = _mm_packs_epi32(uint32Px1, uint32Px1);

            if constexpr (std::is_same_v<ChannelType, uint8_t>)
            {
                auto uint8Px1 = _mm_packus_epi16(uint16Px1, uint16Px1);
                *ptrOutputImage = _mm_cvtsi128_si32(uint8Px1);
            }
            else
            {
                *(uint64_t*)ptrOutputImage = _mm_cvtsi128_si64(uint16Px1);
                ptrOutputImage++;
            }

            ptrOutputImage++;

        }
    }


}


template<PixelFormat pixelFormat>
class GeneratingResultHelper
{
    Stacker& _stacker;
    std::shared_ptr<Bitmap<pixelFormat>> _pBitmap;

    GeneratingResultHelper(Stacker& stacker)
    : _stacker(stacker)
    , _pBitmap(std::make_shared<Bitmap<pixelFormat>>(stacker._width, stacker._height))
    {

    }

    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

    void Job(uint32_t i)
    {
        using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;

        ChannelType* pChannel = &_pBitmap->GetScanline(i)[0];
        float* pMean = &_stacker._means[i * _stacker._width * ChannelCount(pixelFormat)];

        for (uint32_t x = 0; x < _stacker._width; ++x)
        for (uint32_t ch = 0; ch < ChannelCount(pixelFormat); ++ch)
        {
            *pChannel = FastRound<ChannelType>(*pMean);
            ++pMean;
            ++pChannel;
        }
    }

public:
    static void Run(Stacker& stacker, std::shared_ptr<Bitmap<pixelFormat>> pBitmap)
    {
        GeneratingResultHelper helper(stacker);
        oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, pBitmap->GetHeight() ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
        {
            for ( int i = range.begin(); i < range.end(); ++i )
            {
                helper.Job( i );
            }
        } );
        pBitmap->SetData(helper._pBitmap->GetData());
    }
};

#endif
