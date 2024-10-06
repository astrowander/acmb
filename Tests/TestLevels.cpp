#include "test.h"
#include "testtools.h"

#include "../Transforms/LevelsTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( LevelsTransform )

BEGIN_TEST( TestRGB24 )
auto pBitmap = IBitmap::Create(GetPathToTestFile("PPM/rgb24.ppm"));
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 0.8f, 1.0f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestRGB24.ppm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestRGB48)
auto pBitmap = IBitmap::Create( GetPathToTestFile( "TIFF/rgb48.tiff" ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.1f, 0.4f, 0.7f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestRGB48.tiff" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pBitmap = IBitmap::Create( GetPathToTestFile( "TIFF/gray8.tiff" ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 1.4f, 0.7f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestGray8.tiff" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestGray16 )
auto pBitmap = IBitmap::Create( GetPathToTestFile( "TIFF/gray16.tiff" ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.5f, 1.4f, 1.0f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestGray16.tiff" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestOverflow )
auto pBitmap = IBitmap::Create( 1, 1, IColor::MakeGray16( NamedColor64::Gray ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 1.5f, 0.4f } } } );
EXPECT_EQ( 0xFFFF, pBitmap->GetChannel( 0, 0, 0 ) );
END_TEST

BEGIN_TEST( TestLightening )
auto pBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 1.5f, 1.0f } } } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestLightening.ppm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestChannelsAdjusting )
auto pBitmap = IBitmap::Create( GetPathToTestFile( "PPM/rgb24.ppm" ) );
pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { LevelsTransform::ChannelLevels{ 0.0f, 1.5f, 1.0f }, { 0.1f, 0.8f, 0.9f }, { 0.2f, 0.7f, 0.8f }, { 0.3f, 0.6f, 0.7f } }, .adjustChannels = true } );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "LevelsTransform/TestChannelsAdjusting.ppm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestWrongLevels )
auto f = []
{
    auto pBitmap = IBitmap::Create( 1, 1, IColor::MakeGray16( NamedColor64::Gray ) );
    pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.4f, 1.5f, 0.0f } } } );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestZeroGamma )
auto f = []
{
    auto pBitmap = IBitmap::Create( 1, 1, IColor::MakeGray16( NamedColor64::Gray ) );
    pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.0f, 0.0f, 1.0f } } } );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestEqualMinMax) 
auto f = []
{
    auto pBitmap = IBitmap::Create( 1, 1, IColor::MakeGray16( NamedColor64::Gray ) );
    pBitmap = LevelsTransform::ApplyLevels( pBitmap, { .levels = { { 0.5f, 1.0f, 0.5f } } } );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END