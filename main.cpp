#include "Tests/TestBitmap.cpp"
#include "Tests/TestPpmDecoder.cpp"
#include "Tests/TestPpmEncoder.cpp"
#include "Tests/TestConverter.cpp"
#include "Tests/TestRegistrator.cpp"

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
    RUN_TEST(PpmDecoder, TestByteOrdering);

    RUN_TEST(PpmEncoder, TestGray8);
    RUN_TEST(PpmEncoder, TestGray16);
    RUN_TEST(PpmEncoder, TestRgb24);
    RUN_TEST(PpmEncoder, TestRgb48);

    RUN_TEST(Converter, TestRgb24ToGray8);
    RUN_TEST(Converter, TestRgb48ToGray16);
    RUN_TEST(Converter, TestAstrophoto);

    RUN_TEST(Registrator, BasicTest);
    RUN_TEST(Registrator, TestVertical);
    RUN_TEST(Registrator, RegistrateHugePhoto);

    return 0;
}
