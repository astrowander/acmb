#include "test.h"
#include "testtools.h"
#include "../Transforms/BitmapSubtractor.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE(BitmapSubtractor)

BEGIN_TEST(TestWrongArgs)

auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Black );
auto f = [pSrcBitmap]{ BitmapSubtractor::Create( pSrcBitmap, nullptr ); };  
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST(SubtractFromBlack)
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Black );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Azure );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );
END_TEST

BEGIN_TEST( SubtractFromGray )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Gray );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Azure );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 127, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );
END_TEST

BEGIN_TEST( SubtractFromWhite )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::White );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Azure );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 255, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 128, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::Gray8>>( 1, 1, 146 );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::Gray8>>( 1, 1, 23 );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 123, pResult->GetChannel( 0, 0, 0 ) );

END_TEST

BEGIN_TEST( TestGray16 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::Gray16>>( 1, 1, 34569 );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::Gray16>>( 1, 1, 9876 );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 24693, pResult->GetChannel( 0, 0, 0 ) );

END_TEST

BEGIN_TEST( TestRGB24 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, MakeRGB24(55, 143, 198 ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, MakeRGB24( 12, 43, 134 ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 43, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 100, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 64, pResult->GetChannel( 0, 0, 2 ) );

END_TEST

BEGIN_TEST( TestRGB48 )
auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB48>>( 1, 1, MakeRGB48( 34569, 16252, 1324 ) );
auto pBitmapToSubtract = std::make_shared<Bitmap<PixelFormat::RGB48>>( 1, 1, MakeRGB48( 2342, 6543, 2678 ) );

auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pBitmapToSubtract );
auto pResult = pSubtractor->RunAndGetBitmap();

EXPECT_EQ( 32227, pResult->GetChannel( 0, 0, 0 ) );
EXPECT_EQ( 9709, pResult->GetChannel( 0, 0, 1 ) );
EXPECT_EQ( 0, pResult->GetChannel( 0, 0, 2 ) );

END_TEST


BEGIN_TEST( SubtractDarkFrame )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "DarkFrame/IMG_0990.ppm" ));
auto pDarkFrame = IBitmap::Create( GetPathToTestFile( "DarkFrame/masterdark.ppm" ) );
auto pSubtractor = BitmapSubtractor::Create( pSrcBitmap, pDarkFrame );
auto pResult = pSubtractor->RunAndGetBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapSubtractor/SubtractDarkFrame.ppm" ), pResult ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
