#include "test.h"
#include "testtools.h"
#include "../Transforms/HaloRemovalTransform.h"
#include "../Transforms/ChannelEqualizer.h"
#include "../Transforms/HistogramBuilder.h"

BEGIN_SUITE( HaloRemoval )

BEGIN_TEST( BasicTest )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/halo.ppm" ) );
pSrcBitmap = BaseChannelEqualizer::AutoEqualize( pSrcBitmap );
auto pHistBuilder = BaseHistorgamBuilder::Create( pSrcBitmap );
pHistBuilder->BuildHistogram();
std::array<uint16_t, 3> medianRgb = 
{ 
    uint16_t(pHistBuilder->GetChannelStatistics( 0 ).centils[50]), 
    uint16_t(pHistBuilder->GetChannelStatistics( 1 ).centils[50]), 
    uint16_t(pHistBuilder->GetChannelStatistics( 2 ).centils[50])
};
auto medianHsl = RgbToHsl<uint16_t>( std::span( medianRgb ) );
pSrcBitmap = BaseHaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 250, 10);
pSrcBitmap = BaseHaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 270, 15 );
pSrcBitmap = BaseHaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 300, 10 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/BasicTest.ppm" ), pSrcBitmap ) );

END_TEST

BEGIN_TEST( TestHugeImage )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/hugehalo.ppm" ) );
pSrcBitmap = BaseChannelEqualizer::AutoEqualize( pSrcBitmap );
pSrcBitmap = BaseHaloRemovalTransform::AutoRemove( pSrcBitmap, 0.7f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/TestHugeImage.ppm" ), pSrcBitmap ) );

END_TEST
END_SUITE( HaloRemoval)
