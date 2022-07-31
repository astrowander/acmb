#include "test.h"
#include "testtools.h"
#include "../Transforms/HaloRemovalTransform.h"
#include "../Transforms/ChannelEqualizer.h"
#include "../Transforms/HistogramBuilder.h"

BEGIN_SUITE( HaloRemoval )

BEGIN_TEST( HaloRemoval, BasicTest )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/halo.ppm" ) );
pSrcBitmap = BaseChannelEqualizer::AutoEqualize( pSrcBitmap );
pSrcBitmap = BaseHaloRemovalTransform::RemoveHalo( pSrcBitmap, 0.9f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/BasicTest.ppm" ), pSrcBitmap ) );

END_TEST

BEGIN_TEST( HaloRemoval, TestHugeImage )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "PPM/hugehalo.ppm" ) );
pSrcBitmap = BaseChannelEqualizer::AutoEqualize( pSrcBitmap );
pSrcBitmap = BaseHaloRemovalTransform::RemoveHalo( pSrcBitmap, 1.0f );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "HaloRemoval/TestHugeImage.ppm" ), pSrcBitmap ) );

END_TEST
END_SUITE( HaloRemoval)
