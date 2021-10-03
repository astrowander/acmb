#include <iostream>
#include "ppmdecoder.h"
#include "ppmencoder.h"
#include "converter.h"

#include "TestBitmap.cpp"
#include "TestPpmDecoder.cpp"
#include "TestPpmEncoder.cpp"
#include "TestConverter.cpp"

int main()
{
    RUN_TEST(Bitmap, TestGray8);
    RUN_TEST(Bitmap, TestGray16);
    RUN_TEST(Bitmap, TestRgb48);
    RUN_TEST(Bitmap, TestRgb24);

    RUN_TEST(PpmDecoder, TestPlain);
    RUN_TEST(PpmDecoder, TestGray8);
    RUN_TEST(PpmDecoder, TestGray16);
    RUN_TEST(PpmDecoder, TestRgb24);
    RUN_TEST(PpmDecoder, TestRgb48);

    RUN_TEST(PpmEncoder, TestGray8);
    RUN_TEST(PpmEncoder, TestGray16);
    RUN_TEST(PpmEncoder, TestRgb24);
    RUN_TEST(PpmEncoder, TestRgb48);

    RUN_TEST(Converter, TestRgb24ToGray8);
    RUN_TEST(Converter, TestRgb48ToGray16);
    return 0;
}
