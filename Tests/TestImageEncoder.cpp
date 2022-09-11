#include "test.h"
#include "testtools.h"
#include "./../Codecs/ImageEncoder.h"

BEGIN_SUITE( ImageEncoder )

BEGIN_TEST( TestExtensions )

auto extensions = ImageEncoder::GetAllExtensions();
EXPECT_EQ( 7, extensions.size() );

END_TEST

END_SUITE( ImageEncoder )