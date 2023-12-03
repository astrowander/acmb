#include "test.h"
#include "testtools.h"
#include "../Codecs/FITS/FitsEncoder.h"

ACMB_TESTS_NAMESPACE_BEGIN


BEGIN_SUITE( FitsEncoder )

BEGIN_TEST( TestGray8 )
EXPECT_TRUE( TestPixelFormat<FitsEncoder>( "Gray8" ) );
END_TEST

BEGIN_TEST( TestGray16 )
EXPECT_TRUE( TestPixelFormat<FitsEncoder>( "Gray16" ) );
END_TEST

BEGIN_TEST( TestRGB24 )
EXPECT_TRUE( TestPixelFormat<FitsEncoder>( "RGB24" ) );
END_TEST

BEGIN_TEST( TestRGB48 )
EXPECT_TRUE( TestPixelFormat<FitsEncoder>( "RGB48" ) );
END_TEST

BEGIN_TEST( TestNullBitmap )
auto f = [] ()
{
    IBitmapPtr pBitmap = nullptr;
    FitsEncoder fitsEncoder;
    fitsEncoder.Attach( GetPathToPattern( "TiffEncoder/TestNullBitmap.tiff" ) );
    fitsEncoder.WriteBitmap( nullptr );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( TestWritingToStream )

auto pStream = std::make_shared<std::ostringstream>();
auto pRefBitmap = IBitmap::Create( GetPathToTestFile( std::string( "TIFF/RGB48.tiff" ) ) );

FitsEncoder fitsEncoder;
fitsEncoder.Attach( pStream );
fitsEncoder.WriteBitmap( pRefBitmap );
const auto str = pStream->str();
EXPECT_EQ( 7375680, str.size() );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
