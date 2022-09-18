#include "test.h"
#include "testtools.h"
#include "./../Transforms/HistogramBuilder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(HistogramBuilder)

BEGIN_TEST(TestRGB48)

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("PPM/halo.ppm"));
auto pHistogramBuilder = HistorgamBuilder::Create(pSrcBitmap);
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

BEGIN_TEST( TestNullArgs )
auto f = []
{
    HistorgamBuilder::Create( nullptr );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END