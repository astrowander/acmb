#include "test.h"
#include "testtools.h"
#include "../Core/color.h"
ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Color )

BEGIN_TEST( TestRGB24 )
auto pColor = IColor::Create( PixelFormat::RGB24, { 255, 127, 1 } );
EXPECT_EQ( 255, pColor->GetChannel( 0 ) );
EXPECT_EQ( 127, pColor->GetChannel( 1 ) );
EXPECT_EQ( 1, pColor->GetChannel( 2 ) );
END_TEST

BEGIN_TEST( TestRGB48 )
auto pColor = IColor::Create( PixelFormat::RGB48, { 65535, 32787, 1256 } );
EXPECT_EQ( 65535, pColor->GetChannel( 0 ) );
EXPECT_EQ( 32787, pColor->GetChannel( 1 ) );
EXPECT_EQ( 1256, pColor->GetChannel( 2 ) );
END_TEST

BEGIN_TEST( TestRGBA32 )
auto pColor = IColor::Create( PixelFormat::RGBA32, { 255, 127, 1, 255 } );
EXPECT_EQ( 255, pColor->GetChannel( 0 ) );
EXPECT_EQ( 127, pColor->GetChannel( 1 ) );
EXPECT_EQ( 1, pColor->GetChannel( 2 ) );
EXPECT_EQ( 255, pColor->GetChannel( 3 ) );
END_TEST

BEGIN_TEST( TestRGBA64 )
auto pColor = IColor::Create( PixelFormat::RGBA64, { 65535, 32787, 1256, 65535 } );
EXPECT_EQ( 65535, pColor->GetChannel( 0 ) );
EXPECT_EQ( 32787, pColor->GetChannel( 1 ) );
EXPECT_EQ( 1256, pColor->GetChannel( 2 ) );
EXPECT_EQ( 65535, pColor->GetChannel( 3 ) );
END_TEST

BEGIN_TEST( TestGray8 )
auto pColor = IColor::Create( PixelFormat::Gray8, { 255 } );
EXPECT_EQ( 255, pColor->GetChannel( 0 ) );
END_TEST

BEGIN_TEST( TestGray16 )
auto pColor = IColor::Create( PixelFormat::Gray16, { 65535 } );
EXPECT_EQ( 65535, pColor->GetChannel( 0 ) );
END_TEST

BEGIN_TEST( TestExceedingBounds )
auto pColor = IColor::Create( PixelFormat::Gray8, { 256 } );
EXPECT_EQ( 0, pColor->GetChannel( 0 ) );
END_TEST

BEGIN_TEST( TestChannelOutOfBounds )
auto f = []
{
    auto pColor = IColor::Create( PixelFormat::Gray8, { 0 } );
    pColor->GetChannel( 1 );
};
ASSERT_THROWS( f, std::out_of_range );
END_TEST

BEGIN_TEST( TestCreatingNamedColor )
auto pColor = IColor::MakeRGB24( NamedColor32::Azure );
EXPECT_EQ( 0x00, pColor->GetChannel( 0 ) );
EXPECT_EQ( 0x7F, pColor->GetChannel( 1 ) );
EXPECT_EQ( 0xFF, pColor->GetChannel( 2 ) );
END_TEST

BEGIN_TEST( TestCreatingNamedColorWithAlpha )
auto pColor = IColor::MakeRGBA32( NamedColor32::Azure );
EXPECT_EQ( 0x00, pColor->GetChannel( 0 ) );
EXPECT_EQ( 0x7F, pColor->GetChannel( 1 ) );
EXPECT_EQ( 0xFF, pColor->GetChannel( 2 ) );
EXPECT_EQ( 0xFF, pColor->GetChannel( 3 ) );
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END