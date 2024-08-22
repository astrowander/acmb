#include "test.h"
#include "testtools.h"
#include "../Transforms/BitmapSubtractor.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(BitmapSubtractor)

BEGIN_TEST(TestWrongArgs)

auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::Black ) );
auto f = [pSrcBitmap] {  BitmapSubtractor::Create( pSrcBitmap, { nullptr } ); };
ASSERT_THROWS( f, std::invalid_argument );
auto f2 = [pSrcBitmap] { BitmapSubtractor::Create( nullptr, { pSrcBitmap } ); };
ASSERT_THROWS( f2, std::invalid_argument );
END_TEST

BEGIN_TEST(SubtractFromBlack)
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::Black ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::Azure ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );
END_TEST

BEGIN_TEST( SubtractFromGray )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::Gray ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::Azure ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 127, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );
END_TEST

BEGIN_TEST( SubtractFromWhite )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::White ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( NamedColor32::Azure ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 255, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 128, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::Gray8>>( 1, 1, IColor::MakeGray8( 146 ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::Gray8>>( 1, 1, IColor::MakeGray8( 23 ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 123, pResult->GetChannel( 0, 0, 0 ) );

END_TEST

BEGIN_TEST( TestGray16 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::Gray16>>( 1, 1, IColor::MakeGray16( 34569 ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::Gray16>>( 1, 1, IColor::MakeGray16( 9876 ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 24693, pResult->GetChannel( 0, 0, 0 ) );

END_TEST

BEGIN_TEST( TestRGB24 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24(55, 143, 198 ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, IColor::MakeRGB24( 12, 43, 134 ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 43, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 100, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 64, pResult->GetChannel( 0, 0, 2 ) );

END_TEST

BEGIN_TEST( TestRGB48 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB48>>( 1, 1, IColor::MakeRGB48( 34569, 16252, 1324 ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB48>>( 1, 1, IColor::MakeRGB48( 2342, 6543, 2678 ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pBitmapToSubtract } );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 32227, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 9709, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );

END_TEST


BEGIN_TEST( SubtractDarkFrame )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "DarkFrame/IMG_0990.ppm" ));
auto pDarkFrame = IBitmap::Create( GetPathToTestFile( "DarkFrame/masterdark.ppm" ) );
auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pDarkFrame } );
auto pResult = pSubtractor->RunAndGetBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapSubtractor/SubtractDarkFrame.ppm" ), pResult ) );

END_TEST

BEGIN_TEST( SubtractUndebayered )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "FlatField/IMG_0914.CR2" ) );
auto pDarkFrame = IBitmap::Create( GetPathToTestFile( "FlatField/masterdark.tif" ), PixelFormat::Bayer16 );
auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pDarkFrame } );
auto pResult = pSubtractor->RunAndGetBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapSubtractor/SubtractUndebayered.tif" ), pResult ) );
END_TEST

BEGIN_TEST( TestIntensity )
auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "DarkFrame/IMG_5090.CR2" ) );
auto pDarkFrame = IBitmap::Create( GetPathToTestFile( "DarkFrame/masterdark_intensity.tif" ), PixelFormat::Bayer16 );
auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, { pDarkFrame, 2.0f } );
auto pResult = pSubtractor->RunAndGetBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapSubtractor/TestIntensity.tif" ), pResult ) );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
