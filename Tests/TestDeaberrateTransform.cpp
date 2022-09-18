#include "test.h"
#include "testtools.h"
#include "../Transforms/deaberratetransform.h"
#include "../Codecs/Raw/RawDecoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( DeaberrateTransform )

BEGIN_TEST( TestRgb24 )
auto pDecoder = std::make_unique<RawDecoder>( RawSettings{ .halfSize = false, .extendedFormat = false } );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/TestRgb24.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );
END_TEST

BEGIN_TEST( TestRgb48 )
auto pDecoder = std::make_unique<RawDecoder>( RawSettings{ .halfSize = false, .extendedFormat = true } );
pDecoder->Attach( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) );
auto pDeaberrateTransform = DeaberrateTransform::Create( pDecoder->ReadBitmap(), pDecoder->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "DeaberrateTransform/TestRgb48.ppm" ), pDeaberrateTransform->RunAndGetBitmap() ) );
END_TEST

END_SUITE
ACMB_TESTS_NAMESPACE_END