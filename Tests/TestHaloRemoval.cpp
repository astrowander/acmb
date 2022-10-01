#include "test.h"
#include "testtools.h"
#include "../Transforms/HaloRemovalTransform.h"
#include "../Transforms/Converter.h"
#include "../Transforms/ChannelEqualizer.h"
#include "../Transforms/HistogramBuilder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( HaloRemoval )

BEGIN_TEST( TestRGB48 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/halo.ppm" ) );
pSrcBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
auto pHistBuilder = HistorgamBuilder::Create( pSrcBitmap );
pHistBuilder->BuildHistogram();
std::array<uint16_t, 3> medianRgb = 
{ 
    uint16_t(pHistBuilder->GetChannelStatistics( 0 ).centils[50]), 
    uint16_t(pHistBuilder->GetChannelStatistics( 1 ).centils[50]), 
    uint16_t(pHistBuilder->GetChannelStatistics( 2 ).centils[50])
};
auto medianHsl = RgbToHsl<uint16_t>( std::span( medianRgb ) );
pSrcBitmap = HaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 250, 10);
pSrcBitmap = HaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 270, 15 );
pSrcBitmap = HaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 300, 10 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/TestRGB48.ppm" ), pSrcBitmap ) );

END_TEST

BEGIN_TEST( TestRGB24 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/halo.ppm" ) );
pSrcBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
pSrcBitmap = Converter::Convert( pSrcBitmap, PixelFormat::RGB24 );
auto pHistBuilder = HistorgamBuilder::Create( pSrcBitmap );
pHistBuilder->BuildHistogram();
std::array<uint16_t, 3> medianRgb =
{
    uint16_t( pHistBuilder->GetChannelStatistics( 0 ).centils[50] ),
    uint16_t( pHistBuilder->GetChannelStatistics( 1 ).centils[50] ),
    uint16_t( pHistBuilder->GetChannelStatistics( 2 ).centils[50] )
};
auto medianHsl = RgbToHsl<uint16_t>( std::span( medianRgb ) );
pSrcBitmap = HaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 250, 10 );
pSrcBitmap = HaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 270, 15 );
pSrcBitmap = HaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f, medianHsl[2] * 2, 300, 10 );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/TestRGB24.ppm" ), pSrcBitmap ) );

END_TEST

BEGIN_TEST( TestHugeImage )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/hugehalo.ppm" ) );
pSrcBitmap = ChannelEqualizer::AutoEqualize( pSrcBitmap );
pSrcBitmap = HaloRemovalTransform::AutoRemove( pSrcBitmap, 0.7f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/TestHugeImage.ppm" ), pSrcBitmap ) );

END_TEST

BEGIN_TEST( TestNullArgs )
auto f = []
{
    HaloRemovalTransform::AutoRemove( nullptr, 0.7f );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
