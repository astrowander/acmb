#include "test.h"
#include "testtools.h"
#include "../Codecs/SER/SerDecoder.h"
#include "../Core/bitmap.h"
#include "../Transforms/DebayerTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( SerDecoder )

BEGIN_TEST( SimpleTest )
    SerDecoder decoder;
    decoder.Attach( GetPathToTestFile( "SER/19_45_36.ser" ) );
    EXPECT_EQ( 800, decoder.GetWidth() );
    EXPECT_EQ( 600, decoder.GetHeight() );
    EXPECT_EQ( PixelFormat::Bayer16, decoder.GetPixelFormat() );
    EXPECT_EQ( 10, decoder.GetFrameCount() );

    auto pBitmap = decoder.ReadBitmap();
    pBitmap = DebayerTransform::Debayer( pBitmap, std::make_shared<CameraSettings>() );

    EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "SerDecoder/SimpleTest.ppm" ), pBitmap ) );
END_TEST

BEGIN_TEST( TestEmptyFile )
    auto f = [] ()
    {
        SerDecoder decoder;
        decoder.Attach( GetPathToTestFile( "SER/empty.ser" ) );
    };
    ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestCorruptedHeader )
auto f = [] ()
{
    SerDecoder decoder;
    decoder.Attach( GetPathToTestFile( "SER/corrupted_header.ser" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

BEGIN_TEST( TestCorruptedBitmap )
auto f = [] ()
{
    SerDecoder decoder;
    decoder.Attach( GetPathToTestFile( "SER/corrupted_bitmap.ser" ) );
};
ASSERT_THROWS( f, std::runtime_error );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END