#include "test.h"
#include "testtools.h"
#include "../Transforms/deaberratetransform.h"
#include "../Transforms/converter.h"
#include "../Codecs/Raw/RawDecoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( DeaberrateTransform )

BEGIN_TEST( TestNullArgs )

    auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB24 );
    pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
    auto pCameraSettings = pDecoder->GetCameraSettings();
    auto pBitmap = IBitmap::Create( 1, 1, PixelFormat::RGB24 );
    
    auto f1 = [pBitmap, pCameraSettings]
    {
        DeaberrateTransform::Create( pBitmap, nullptr );
    };
    ASSERT_THROWS( f1, std::invalid_argument );

    auto f2 = [pBitmap, pCameraSettings]
    {
        DeaberrateTransform::Create( nullptr, pCameraSettings );
    };
    ASSERT_THROWS( f2, std::invalid_argument );

    auto f3 = [pBitmap, pCameraSettings]
    {
        DeaberrateTransform::Create( nullptr, nullptr );
    };
    ASSERT_THROWS( f2, std::invalid_argument );

END_TEST

BEGIN_TEST( TestRgb24 )
auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB24 );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/TestRgb24.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );
END_TEST

BEGIN_TEST( TestRgb48 )
auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB48 );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/TestRgb48.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );
END_TEST

BEGIN_TEST( TestDifferentLenses )

auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB48 );
pDecoder->Attach( GetPathToTestFile( "RAW/TestLensInfo/Canon_200_mm.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/Canon_200_mm.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );

pDecoder = std::make_unique<RawDecoder>( PixelFormat::RGB48 );
pDecoder->Attach( GetPathToTestFile( "RAW/TestLensInfo/Canon_24_105_mm.CR2" ) );
pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/Canon_24_105_mm.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );

END_TEST

BEGIN_TEST( TestGray16 )

auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::Gray16 );
pDecoder->Attach( GetPathToTestFile( "RAW/TestLensInfo/Canon_24_105_mm.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/TestNotDebayered.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );

END_TEST

BEGIN_TEST( TestGray8 )
auto pDecoder = std::make_unique<RawDecoder>( PixelFormat::Gray8 );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/TestGray8.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END