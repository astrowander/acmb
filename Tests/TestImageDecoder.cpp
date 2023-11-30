#include "test.h"
#include "testtools.h"
#include "./../Codecs/imagedecoder.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( ImageDecoder )

BEGIN_TEST( TestExtensions )

auto pipelines = ImageDecoder::GetPipelinesFromDir( GetPathToTestFile( "AllFormats/" ) );
EXPECT_EQ( 4, pipelines.size() );

END_TEST

BEGIN_TEST( TestExtensionsSize )

EXPECT_EQ( 34, ImageDecoder::GetAllExtensions().size() );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END