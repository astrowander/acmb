#include "test.h"
#include "testtools.h"
#include "./../Transforms/HistogramBuilder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(HistogramBuilder)

BEGIN_TEST(TestRGB48)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/halo.ppm"));
auto pHistogramBuilder = HistogramBuilder::Create(pSrcBitmap);
pHistogramBuilder->BuildHistogram();

EXPECT_EQ( 6632, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 13389, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 65503, pHistogramBuilder->GetChannelStatistics( 0 ).max );

EXPECT_EQ( 0, pHistogramBuilder->GetChannelStatistics( 1 ).min );
EXPECT_EQ( 12209, pHistogramBuilder->GetChannelStatistics( 1 ).peak );
EXPECT_EQ( 65535, pHistogramBuilder->GetChannelStatistics( 1 ).max );

EXPECT_EQ( 3210, pHistogramBuilder->GetChannelStatistics( 2 ).min );
EXPECT_EQ( 8669, pHistogramBuilder->GetChannelStatistics( 2 ).peak );
EXPECT_EQ( 65435, pHistogramBuilder->GetChannelStatistics( 2 ).max );
END_TEST

BEGIN_TEST( TestRGB24 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/IMG_8970.ppm" ) );
auto pHistogramBuilder = HistogramBuilder::Create( pSrcBitmap );
pHistogramBuilder->BuildHistogram();

EXPECT_EQ( 0, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 67, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 255, pHistogramBuilder->GetChannelStatistics( 0 ).max );

EXPECT_EQ( 0, pHistogramBuilder->GetChannelStatistics( 1 ).min );
EXPECT_EQ( 56, pHistogramBuilder->GetChannelStatistics( 1 ).peak );
EXPECT_EQ( 255, pHistogramBuilder->GetChannelStatistics( 1 ).max );

EXPECT_EQ( 0, pHistogramBuilder->GetChannelStatistics( 2 ).min );
EXPECT_EQ( 31, pHistogramBuilder->GetChannelStatistics( 2 ).peak );
EXPECT_EQ( 255, pHistogramBuilder->GetChannelStatistics( 2 ).max );
END_TEST

BEGIN_TEST( TestGray16 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "/gray.pgm" ) );
auto pHistogramBuilder = HistogramBuilder::Create( pSrcBitmap );
pHistogramBuilder->BuildHistogram();

EXPECT_EQ( 0, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 65535, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 65535, pHistogramBuilder->GetChannelStatistics( 0 ).max );

END_TEST

BEGIN_TEST( TestGray8 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "TIFF/Gray8.tiff" ) );
auto pHistogramBuilder = HistogramBuilder::Create( pSrcBitmap );
pHistogramBuilder->BuildHistogram();

EXPECT_EQ( 0, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 20, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 255, pHistogramBuilder->GetChannelStatistics( 0 ).max );

END_TEST

BEGIN_TEST( TestROI)

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "TIFF/jupiter_best_laplacian.tif" ) );
auto pHistogramBuilder = HistogramBuilder::Create( pSrcBitmap );
pHistogramBuilder->BuildHistogram( Rect{ 0, 0, 28, 28 } );

EXPECT_EQ( 6, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 369, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 37087, pHistogramBuilder->GetChannelStatistics( 0 ).max );
EXPECT_EQ( 1858, pHistogramBuilder->GetChannelStatistics( 0 ).median );
EXPECT_EQ( 30156, pHistogramBuilder->GetChannelStatistics( 0 ).topPercentile );

pHistogramBuilder->BuildHistogram( Rect{ 28, 28, 27, 25 } );

EXPECT_EQ( 4, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 17574, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 37263, pHistogramBuilder->GetChannelStatistics( 0 ).max );
EXPECT_EQ( 3368, pHistogramBuilder->GetChannelStatistics( 0 ).median );
EXPECT_EQ( 26616, pHistogramBuilder->GetChannelStatistics( 0 ).topPercentile );

pHistogramBuilder = HistogramBuilder::Create( pSrcBitmap );
pHistogramBuilder->BuildHistogram( Rect{ 28, 28, 27, 25 } );
EXPECT_EQ( 4, pHistogramBuilder->GetChannelStatistics( 0 ).min );
EXPECT_EQ( 17574, pHistogramBuilder->GetChannelStatistics( 0 ).peak );
EXPECT_EQ( 37263, pHistogramBuilder->GetChannelStatistics( 0 ).max );
EXPECT_EQ( 3368, pHistogramBuilder->GetChannelStatistics( 0 ).median );
EXPECT_EQ( 26616, pHistogramBuilder->GetChannelStatistics( 0 ).topPercentile );

END_TEST

BEGIN_TEST( TestNullArgs )
auto f = []
{
    HistogramBuilder::Create( nullptr );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
