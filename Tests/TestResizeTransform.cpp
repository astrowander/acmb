#include "test.h"
#include "testtools.h"
#include "../Transforms/ResizeTransform.h"

#define TEST_PIXEL_FORMAT( fmt ) \
BEGIN_TEST( Test ##fmt ) \
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "ResizeTransform/Test" #fmt ".tiff"), ResizeTransform::Resize(IBitmap::Create(GetPathToTestFile("TIFF/" #fmt ".tiff")), {1280, 720}) ));\
END_TEST

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( ResizeTransform )


TEST_PIXEL_FORMAT( Gray8 );
TEST_PIXEL_FORMAT( Gray16 );
TEST_PIXEL_FORMAT( RGB24 );
TEST_PIXEL_FORMAT( RGB48 );

BEGIN_TEST( TestZeroSize )

auto f1 = []
{
    ResizeTransform::Create( PixelFormat::Gray8, { 0, 2 } );
};

ASSERT_THROWS( f1, std::invalid_argument );

auto f2 = []
{
    ResizeTransform::Create( PixelFormat::Gray8, { 2, 0 } );
};

ASSERT_THROWS( f2, std::invalid_argument );

END_TEST

BEGIN_TEST( TestNullArg )
auto f = [] ()
{
    ResizeTransform::Create( nullptr, { 3, 3 } );
};
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( Test1x1 )
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "ResizeTransform/Test1x1.tiff" ), ResizeTransform::Resize( IBitmap::Create( GetPathToTestFile( "TIFF/RGB24.tiff" ) ), { 1, 1 } ) ) );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END
