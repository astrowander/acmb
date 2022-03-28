#include "test.h"
#include "testtools.h"
#include "./../Transforms/HistogramBuilder.h"

BEGIN_SUITE(HistogramBuilder)

BEGIN_TEST(HistogramBuilder, TestRGB48)

auto pSrcBitmap = IBitmap::Create(GetPathToPattern("Stacker/TestMilkyWay.ppm"));
auto pHistogramBuilder = BaseHistorgamBuilder::Create(pSrcBitmap);
pHistogramBuilder->BuildHistogram();
std::cout << "red: " << pHistogramBuilder->GetChannelStatistics(0).min << " " << pHistogramBuilder->GetChannelStatistics(0).peak << " " << pHistogramBuilder->GetChannelStatistics(0).max << std::endl;
std::cout << "green: " << pHistogramBuilder->GetChannelStatistics(1).min << " " << pHistogramBuilder->GetChannelStatistics(1).peak << " " << pHistogramBuilder->GetChannelStatistics(1).max << std::endl;
std::cout << "blue: " << pHistogramBuilder->GetChannelStatistics(2).min << " " << pHistogramBuilder->GetChannelStatistics(2).peak << " " << pHistogramBuilder->GetChannelStatistics(2).max << std::endl;
END_TEST
END_SUITE