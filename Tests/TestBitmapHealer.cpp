#include "test.h"
#include "testtools.h"
#include "../Transforms/BitmapHealer.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( BitmapHealer )

BEGIN_TEST ( TestRGB24 )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "JPEG/IMG_4175.JPG" ) );

std::vector <BitmapHealer::Patch > patches = { {.from = { 4984, 435 }, .to = { 4904, 435 }, .radius = 50, .gamma = 0.1f } };

auto pDstBitmap = BitmapHealer::ApplyTransform( pSrcBitmap, patches );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapHealer/TestRGB48.ppm" ), pDstBitmap ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END