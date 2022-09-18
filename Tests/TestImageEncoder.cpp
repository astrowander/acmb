#include "test.h"
#include "testtools.h"
#include "./../Codecs/imageencoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( ImageEncoder )

BEGIN_TEST( TestExtensions )

auto extensions = ImageEncoder::GetAllExtensions();
EXPECT_EQ( 7, extensions.size() );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END