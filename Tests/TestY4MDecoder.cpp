#include "test.h"
#include "testtools.h"
#include "../Codecs/Y4M/Y4MDecoder.h"
#include "../Transforms/converter.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Y4MDecoder )

BEGIN_TEST( TestReading )

Y4MDecoder decoder;
decoder.Attach( GetPathToTestFile( "Y4M/blink.y4m" ) );
const auto frameCount = decoder.GetFrameCount();
EXPECT_EQ( frameCount, 50 );

auto frame0 = decoder.ReadBitmap( 0 );
auto frame1 = decoder.ReadBitmap( 1 );
auto frame49 = decoder.ReadBitmap( 49 );
EXPECT_TRUE( BitmapsAreEqual( frame0, frame49 ) );
EXPECT_FALSE( BitmapsAreEqual( frame0, frame1 ) );

//EXPECT_TRUE( BitmapsAreEqual( frames[24], frames[25] ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END