#include "test.h"
#include "testtools.h"
#include "./../Tools/mathtools.h"
#include <array>

BEGIN_SUITE( RgbToHsl )

BEGIN_TEST( RgbToHsl, DoubleConversion )

const std::vector< std::array<uint8_t, 3>> rgbColors = 
{
    {0xFF, 0xFF, 0xFF}, //white
    {0x80, 0x80, 0x80}, //gray
    {0x00, 0x00, 0x00}, //black
    {0xFF, 0x00, 0x00}, //red
    {0xBF, 0xBF, 0x00}, //buddah gold
    {0x00, 0x80, 0x00}, // light green
    {0x80, 0xff, 0xff}, //electric blue
    {0x80, 0x80, 0xff}, //light slate blue
    {0xbf, 0x40, 0xbf}, //dark pink
    {0xa0, 0xa4, 0x24}, // gold
    {0x41, 0x1b, 0xea},
    {0x1e, 0xac, 0x41},
    {0xf0, 0xc8, 0x0e},
    {0xb4, 0x30, 0xe5},
    {0xed, 0x76, 0x51},
    {0xfe, 0xf8, 0x88},
    {0x19, 0xcb, 0x97},
    {0x36, 0x26, 0x98},
    {0x7e, 0x7e, 0xb8}
};

const std::vector< std::array<double, 3> > hslColors =
{
    {   0.0, 0.000, 1.000 }, //white
    {   0.0, 0.000, 0.500 }, //gray
    {   0.0, 0.000, 0.000 }, //black
    {   0.0, 1.000, 0.500 }, //red
    {  60.0, 1.000, 0.375 }, //buddah gold
    { 120.0, 1.000, 0.250 }, // light green
    { 180.0, 1.000, 0.750 }, //electric blue
    { 240.0, 1.000, 0.750 }, //light slate blue
    { 300.0, 0.500, 0.500 }, //dark pink
    {  61.8, 0.638, 0.393 },
    { 251.1, 0.832, 0.511 },
    { 134.9, 0.707, 0.396 },
    {  49.5, 0.893, 0.498 },
    { 283.7, 0.775, 0.543 },
    {  14.3, 0.817, 0.624 },
    {  56.9, 0.991, 0.765 },
    { 162.4, 0.779, 0.447 },
    { 248.3, 0.601, 0.373 },
    { 240.5, 0.290, 0.608 }
};

for ( size_t i = 0; i < rgbColors.size(); ++i )
{
    auto actualHsl = RgbToHsl( rgbColors[i] );
    EXPECT_NEAR( hslColors[i][0], actualHsl[0], 1.0 );
    EXPECT_NEAR( hslColors[i][1], actualHsl[1], 0.01 );
    EXPECT_NEAR( hslColors[i][2], actualHsl[2], 0.01 );

    auto convertedRgb = HslToRgb<uint8_t>( actualHsl );

    EXPECT_EQ( rgbColors[i][0], convertedRgb[0] );
    EXPECT_EQ( rgbColors[i][1], convertedRgb[1] );
    EXPECT_EQ( rgbColors[i][2], convertedRgb[2] );
}

END_TEST

BEGIN_TEST( RgbToHsl, TestRgb48 )

std::array<uint16_t, 3> rgb = { 13107, 39321, 52428 };
auto hsl = RgbToHsl( rgb );
auto convertedRgb = HslToRgb<uint16_t>( hsl );
EXPECT_EQ( convertedRgb[0], convertedRgb[0] );
EXPECT_EQ( convertedRgb[1], convertedRgb[1] );
EXPECT_EQ( convertedRgb[2], convertedRgb[2] );

END_TEST

BEGIN_TEST ( RgbToHsl, TestDesaturation )

std::array<uint8_t, 3> rgb = { 129, 0, 188 };
auto hsl = RgbToHsl( rgb );
hsl[1] *= 0.5f;
hsl[2] *= 0.5f;
rgb = HslToRgb<uint8_t>( hsl );

EXPECT_EQ( rgb[0], 56 );
EXPECT_EQ( rgb[1], 24 );
EXPECT_EQ( rgb[2], 71 );
END_TEST

END_SUITE( RgbToHsl )