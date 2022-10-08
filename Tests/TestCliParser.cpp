#include "test.h"
#include "testtools.h"
#include "../Tools/CliParser.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( CliParser )

BEGIN_TEST( InvalidArgc )
auto f = [] () { CliParser::Parse( 0, nullptr ); };
ASSERT_THROWS(f, std::invalid_argument );
END_TEST

BEGIN_TEST( InvalidArgv )
auto f = [] () { CliParser::Parse( 1, nullptr ); };
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( NoArgs )

std::vector<const char*> data = { "AstroCombiner" };
CliParser parser(data.size(), (&data[0]) );
auto [res, errMsg] = parser.Parse( true );
EXPECT_EQ(1, res);
EXPECT_EQ( std::string( "acmb version " ) + FULL_VERSION, errMsg);

END_TEST

BEGIN_TEST( NoOutput )

std::vector<const char*> data = { "AstroCombiner", "--input", "."};
CliParser parser( data.size(), ( &data[0] ) );
auto [res,errMsg] = parser.Parse( true );
EXPECT_EQ( 1, res );
EXPECT_EQ( "Output files must be specified in the last place", errMsg );

END_TEST

BEGIN_TEST( NoInput)

std::vector<const char*> data = { "acmb", "--output", "test.txt" };
CliParser parser( data.size(), ( &data[0] ) );
auto [res,errMsg] = parser.Parse( true );
EXPECT_EQ( 1, res );
EXPECT_EQ( "Input files must be specified in the first place", errMsg );

END_TEST

BEGIN_TEST( StackOneFile )

std::string path = GetPathToTestFile( "RAW/IMG_8899.CR2" );
std::vector<const char*> data = { "acmb", "--input", path.data(), "--stack", "--output", "test.txt"};
CliParser parser( data.size(), ( &data[0] ) );
auto [res,errMsg] = parser.Parse( true );
EXPECT_EQ( 0, res );
EXPECT_EQ( 1, parser._pipelinesBeforeStacker.size() );

END_TEST

BEGIN_TEST( StackTwoFiles )

std::string fileNames = GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) + ";" + GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" );
std::vector<const char*> data = { "acmb", "--input", fileNames.data(), "--stack", "--output", "test.txt"};
CliParser parser( data.size(), ( &data[0] ) );
auto [res,errMsg] = parser.Parse( true );
EXPECT_EQ( 0, res );
EXPECT_EQ( 2, parser._pipelinesBeforeStacker.size() );

END_TEST

BEGIN_TEST( StackWrongDirectory )

auto f = []
{
    std::string dirName = GetPathToTestFile( "RAW/MilkyWayCR4/" );
    std::vector<const char*> data = { "acmb", "--input", dirName.data(), "--stack", "--output", "test.txt" };
    CliParser parser( data.size(), ( &data[0] ) );
    auto [res, errMsg] = parser.Parse( true );
};
ASSERT_THROWS( f, std::invalid_argument );

END_TEST

BEGIN_TEST( StackDirectory )

std::string dirName = GetPathToTestFile( "RAW/MilkyWayCR2/" );
std::vector<const char*> data = { "acmb", "--input", dirName.data(), "--stack", "--output", "test.txt"};
CliParser parser( data.size(), ( &data[0] ) );
auto [res,errMsg] = parser.Parse( true );
EXPECT_EQ( 0, res );
EXPECT_EQ( 27, parser._pipelinesBeforeStacker.size() );

END_TEST

BEGIN_TEST( StackRange )

std::string fileRange = GetPathToTestFile( "RAW/MilkyWayCR2/" ) + "IMG_8947#63.CR2";
std::vector<const char*> data = { "acmb", "--input", fileRange.data(), "--stack", "--output", "test.txt" };
CliParser parser( data.size(), ( &data[0] ) );
auto [res, errMsg] = parser.Parse( true );

END_TEST

BEGIN_TEST( ManySemicolons )

auto f = []
{
    std::vector<const char*> data = { "acmb", "--input", ";;;;;;;;;;", "--stack", "--output", "test.txt" };
    CliParser parser( data.size(), ( &data[0] ) );
    auto [res, errMsg] = parser.Parse( true );
};

ASSERT_THROWS( f, std::invalid_argument );

END_TEST

BEGIN_TEST(StackDarks)

std::string darks = GetPathToTestFile( "RAW/StackWithDarks/Darks/" );
std::vector<const char*> data = { "acmb", "--input", darks.data(), "--stack", "noalign", "--output", "test.tif" };

CliParser parser( data.size(), ( &data[0] ) );
auto [res,errMsg] = parser.Parse( true );
EXPECT_EQ( 0, res );
EXPECT_EQ( 10, parser._pipelinesBeforeStacker.size() );

END_TEST

BEGIN_TEST( StackWithDarks )
std::string dark = GetPathToTestFile( "DarkFrame/masterdark.ppm" );
std::string lights = GetPathToTestFile( "RAW/StackWithDarks/Lights/" );
std::vector<const char*> data = { "acmb", "--input", lights.data(), "--subtract", dark.data(), "--stack", "--autowb", "--removehalo", "--output", "test.tif" };
CliParser parser( data.size(), ( &data[0] ) );
auto [res, errMsg] = parser.Parse( true );
EXPECT_EQ( 0, res );
EXPECT_EQ( 10, parser._pipelinesBeforeStacker.size() );
EXPECT_EQ( 3, parser._pipelineAfterStacker.GetSize() );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
