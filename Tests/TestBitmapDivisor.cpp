#include "test.h"
#include "testtools.h"
#include "../Codecs/Raw/RawDecoder.h"
#include "../Transforms/BitmapDivisor.h"
#include "../Transforms/BitmapSubtractor.h"
#include "../Transforms/DebayerTransform.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( BitmapDivisor )

BEGIN_TEST( TestWrongArgs )

auto pSrcBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>( 1, 1, ARGB32Color::Black );
auto f = [pSrcBitmap]
{
    BitmapDivisor::Create( pSrcBitmap, { .pDivisor = nullptr } );
};
ASSERT_THROWS( f, std::invalid_argument );
auto f2 = [pSrcBitmap]
{
    BitmapDivisor::Create( nullptr, { .pDivisor = pSrcBitmap } );
};
ASSERT_THROWS( f2, std::invalid_argument );
END_TEST

BEGIN_TEST( TestUndebayeredDivision )

auto pSrcBitmap = IBitmap::Create(GetPathToTestFile("FlatField/IMG_0914.CR2"));
auto pFlatField = IBitmap::Create( GetPathToTestFile( "FlatField/masterflat.tif" ), PixelFormat::Bayer16 );
auto pResult = BitmapDivisor::Divide( pSrcBitmap, { .pDivisor = pFlatField, .intensity = 33 } );
pResult = DebayerTransform::Debayer( pResult, pSrcBitmap->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapDivisor/TestUndebayeredDivision.tif" ), pResult ) );

END_TEST

BEGIN_TEST( TestUndebayeredSubtractionAndDivision )

auto pSrcBitmap = IBitmap::Create( GetPathToTestFile( "FlatField/IMG_0914.CR2" ) );
auto pDarkFrame = IBitmap::Create( GetPathToTestFile( "FlatField/masterdark.tif" ), PixelFormat::Bayer16  );
auto pFlatField = IBitmap::Create( GetPathToTestFile( "FlatField/masterflat.tif" ), PixelFormat::Bayer16  );
auto pResult = BitmapSubtractor::Subtract( pSrcBitmap, pDarkFrame );
pResult = BitmapDivisor::Divide( pResult, { .pDivisor = pFlatField, .intensity = 33 } );
pResult = DebayerTransform::Debayer( pResult, pSrcBitmap->GetCameraSettings() );
EXPECT_TRUE( BitmapsAreEqual( GetPathToPattern( "BitmapDivisor/TestUndebayeredSubtractionAndDivision.tif" ), pResult ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END