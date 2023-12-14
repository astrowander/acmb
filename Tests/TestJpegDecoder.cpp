#include "test.h"
#include "testtools.h"
#include "../Codecs/Tiff/TiffDecoder.h"
#include "../Codecs/JPEG/JpegDecoder.h"
#include "../Core/bitmap.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( JpegDecoder )

BEGIN_TEST( TestRGB24 )

    JpegDecoder decoder;
    decoder.Attach( GetPathToTestFile( "JPEG/4.jpg" ) );
    const auto pBitmap = decoder.ReadBitmap();
    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "JpegDecoder/4.ppm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestLargeImage )

JpegDecoder decoder;
decoder.Attach( GetPathToTestFile( "JPEG/large.jpg" ) );
const auto pBitmap = decoder.ReadBitmap();
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "JpegDecoder/large.ppm" ), pBitmap ) );

END_TEST

BEGIN_TEST( TestEmptyFile )
auto f = [] ()
{
    JpegDecoder decoder;
    decoder.Attach( GetPathToTestFile( "JPEG/empty.jpg" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestCorruptedFile )
auto f = [] ()
{
    JpegDecoder decoder;
    decoder.Attach( GetPathToTestFile( "JPEG/corrupted.jpg" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END
