#include "test.h"
#include "testtools.h"
#include "../Codecs/Y4M/Y4MDecoder.h"
#include "../Transforms/converter.h"
#include <sstream>

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( Y4MDecoder )

BEGIN_TEST( RejectTruncatedHeaders )
for (const std::string header : {"YUV4MPEG2 ", "YUV4MPEG2 W2",
    "YUV4MPEG2 W2 ", "YUV4MPEG2 W2 H2 F25:1 C420"})
{
    Y4MDecoder decoder;
    bool rejected = false;
    try
    {
        decoder.Attach(std::make_shared<std::istringstream>(header));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    EXPECT_TRUE(rejected);
}
END_TEST

BEGIN_TEST( RejectOversizedHeadersAndTokens )
for (const std::string header : {
    "YUV4MPEG2 W2 H2 F25:1 C420 X" + std::string(256, 'a') + "\n",
    "YUV4MPEG2 W2 H2 F25:1 C420" + std::string(4096, ' ') + "\n"})
{
    Y4MDecoder decoder;
    bool rejected = false;
    try
    {
        decoder.Attach(std::make_shared<std::istringstream>(header + "FRAME\n" + std::string(6, '\0')));
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    EXPECT_TRUE(rejected);
}
END_TEST

BEGIN_TEST( AcceptHeaderAndTokenAtLimits )
std::string header = "YUV4MPEG2 W2 H2 F25:1 C420 X" + std::string(255, 'a');
header.append(4095 - header.size(), ' ');
header += '\n';
Y4MDecoder decoder;
decoder.Attach(std::make_shared<std::istringstream>(header + "FRAME\n" + std::string(6, '\0')));
EXPECT_EQ(1, decoder.GetFrameCount());
const auto bitmap = decoder.ReadBitmap(0);
EXPECT_EQ(2, bitmap->GetWidth());
EXPECT_EQ(2, bitmap->GetHeight());
END_TEST

BEGIN_TEST( RejectFailedStream )
auto stream = std::make_shared<std::istringstream>("YUV4MPEG2 W2 H2 F25:1 C420\n");
stream->setstate(std::ios::badbit);
Y4MDecoder decoder;
bool rejected = false;
try
{
    decoder.Attach(stream);
}
catch (const std::runtime_error&)
{
    rejected = true;
}
EXPECT_TRUE(rejected);
END_TEST

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
