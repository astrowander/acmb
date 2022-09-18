#include "test.h"
#include "testtools.h"
#include "../Codecs/PPM/ppmdecoder.h"
#include "../Codecs/PPM/ppmencoder.h"
#include "../Core/bitmap.h"

ACMB_TESTS_NAMESPACE_BEGIN

template<PixelFormat pixelFormat, PpmMode mode>
static bool TestPixelFormat()
{
    using ColorEnumType = std::conditional_t<BytesPerChannel( pixelFormat ) == 1, ARGB32Color, ARGB64Color>;
    auto pBitmap = std::make_shared<Bitmap<pixelFormat>>( 20, 23, static_cast< ColorEnumType >( ARGB64Color::Gray ) );
    auto openMode = ( mode == PpmMode::Binary ) ? std::ios_base::out : std::ios_base::out | std::ios_base::binary;

    auto pOutStream = std::make_shared<std::ostringstream>( openMode );
    auto pEncoder = std::make_shared<PpmEncoder>( mode );
    pEncoder->Attach( pOutStream );
    pEncoder->WriteBitmap( pBitmap );

    openMode = ( mode == PpmMode::Binary ) ? std::ios_base::in : std::ios_base::in | std::ios_base::binary;
    auto pInStream = std::make_shared<std::istringstream>( pOutStream->str(), openMode );

    auto pDecoder = std::make_shared<PpmDecoder>();
    pDecoder->Attach( pInStream );
    auto pActual = pDecoder->ReadBitmap();

    return BitmapsAreEqual( pBitmap, pActual );
}

BEGIN_SUITE( PpmEncoder )

BEGIN_TEST(TestGray8)

   EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray8,PpmMode::Binary>()));
   EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray8,PpmMode::Text>()));

END_TEST

BEGIN_TEST(TestGray16)

    EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray16, PpmMode::Binary>()));
    EXPECT_TRUE((TestPixelFormat<PixelFormat::Gray16, PpmMode::Text>()));

END_TEST

BEGIN_TEST(TestRgb24)

    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB24, PpmMode::Binary>()));
    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB24, PpmMode::Text>()));

END_TEST

BEGIN_TEST(TestRgb48)

    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB48, PpmMode::Binary>()));
    EXPECT_TRUE((TestPixelFormat<PixelFormat::RGB48, PpmMode::Text>()));

END_TEST

BEGIN_TEST( TestNullBitmap )
auto f = [] ()
{
    IBitmapPtr pBitmap = nullptr;
    PpmEncoder encoder( PpmMode::Binary );
    encoder.Attach( GetPathToPattern( "TestNullBitmap.ppm" ) );
    encoder.WriteBitmap( nullptr );
};

ASSERT_THROWS( f, std::invalid_argument );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END