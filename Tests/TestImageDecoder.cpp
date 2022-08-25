#include "test.h"
#include "testtools.h"
#include "./../Codecs/ImageDecoder.h"

BEGIN_SUITE( ImageDecoder )

BEGIN_TEST( TestExtensions )

auto decoders = ImageDecoder::GetDecodersFromDir( GetPathToTestFile( "AllFormats/" ) );
EXPECT_EQ( 4, decoders.size() );

END_TEST

END_SUITE( ImageDecoder )