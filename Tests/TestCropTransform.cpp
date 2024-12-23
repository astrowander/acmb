#include "test.h"
#include "testtools.h"
#include "../Transforms/CropTransform.h"

#define TEST_PIXEL_FORMAT( fmt ) \
BEGIN_TEST( Test ##fmt ) \
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CropTransform/Test" #fmt ".tiff"), CropTransform::Crop(IBitmap::Create(GetPathToTestFile("TIFF/" #fmt ".tiff")), {.x = 504, .y = 236, . width = 347, .height = 310}) ));\
END_TEST

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( CropTransform )


TEST_PIXEL_FORMAT( Gray8 );
TEST_PIXEL_FORMAT( Gray16 );
TEST_PIXEL_FORMAT( RGB24 );
TEST_PIXEL_FORMAT( RGB48 );

BEGIN_TEST( TestZeroSize )

auto f1 = []
{
    CropTransform::Create( PixelFormat::Gray8, { 0, 0, 0, 2 } );
};

ASSERT_THROWS( f1, std::invalid_argument );

auto f2 = []
{
    CropTransform::Create( PixelFormat::Gray8, { 0, 0, 2, 0 } );
};

ASSERT_THROWS( f2, std::invalid_argument );

END_TEST

BEGIN_TEST( NegativeOrigin )
auto f1 = []
{
    CropTransform::Create( PixelFormat::Gray8, { -1, 0, 3, 2 } );
};

ASSERT_THROWS( f1, std::invalid_argument );

auto f2 = []
{
    CropTransform::Create( PixelFormat::Gray8, { 0, -1, 3, 2 } );
};

ASSERT_THROWS( f2, std::invalid_argument );
END_TEST

BEGIN_TEST( TooLargeCropRect )
auto f1 = []
{
    auto pTransform = CropTransform::Create( PixelFormat::Gray8, { 0, 0, 1000, 1000 } );
    pTransform->SetSrcBitmap( IBitmap::Create( 200, 200, PixelFormat::Gray8 ) );
};

ASSERT_THROWS( f1, std::runtime_error );
END_TEST

BEGIN_TEST( TooLargeCropRectOnCreate )
auto f1 = []
{
    CropTransform::Create( IBitmap::Create( 20, 20, PixelFormat::Gray8 ), { 0, 0, 1000, 1000 } );
};
ASSERT_THROWS( f1, std::runtime_error );
END_TEST

BEGIN_TEST( TestNullArg )
auto f = [] ()
{
    CropTransform::Create( nullptr, { 0, 0, 3, 3 } );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( Test1x1 )
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CropTransform/Test1x1.tiff" ), CropTransform::Crop( IBitmap::Create( GetPathToTestFile( "TIFF/RGB24.tiff" ) ), { 0, 0, 1, 1 } ) ) );
END_TEST

BEGIN_TEST( TestCropAndFill )
auto pBitmap = IBitmap::Create( 100, 100, IColor::MakeRGB24( NamedColor32::White ) );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CropTransform/CropAndFill_upper_left.tiff" ), CropTransform::CropAndFill( pBitmap, { -50, -50, 100, 100 }, IColor::MakeRGB24( NamedColor32::Blue ) ) ) );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CropTransform/CropAndFill_upper_right.tiff" ), CropTransform::CropAndFill( pBitmap, { 50, -50, 100, 100 }, IColor::MakeRGB24( NamedColor32::Blue ) ) ) );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CropTransform/CropAndFill_bottom_left.tiff" ), CropTransform::CropAndFill( pBitmap, { -50, 50, 100, 100 }, IColor::MakeRGB24( NamedColor32::Blue ) ) ) );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "CropTransform/CropAndFill_bottom_right.tiff" ), CropTransform::CropAndFill( pBitmap, { 50, 50, 100, 100 }, IColor::MakeRGB24( NamedColor32::Blue ) ) ) );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END
