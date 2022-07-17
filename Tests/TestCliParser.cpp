#include "test.h"
#include "testtools.h"
#include "../Tools/CliParser.h"

BEGIN_SUITE( CliParser )

BEGIN_TEST( CliParser, InvalidArgc )
auto f = [] () { CliParser::Parse( 0, nullptr ); };
ASSERT_THROWS(f, std::invalid_argument );
END_TEST

BEGIN_TEST( CliParser, InvalidArgv )
auto f = [] () { CliParser::Parse( 1, nullptr ); };
ASSERT_THROWS( f, std::invalid_argument );
END_TEST

BEGIN_TEST( CliParser, NoArgs )

std::vector<const char*> data = { "AstroCombiner" };
CliParser parser(data.size(), (&data[0]) );
auto res = parser.Parse( true );
EXPECT_EQ(1, res.first);
EXPECT_EQ( "Nothing to do", res.second);

END_TEST

BEGIN_TEST( CliParser, NoOutput )

std::vector<const char*> data = { "AstroCombiner", "-stack" };
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 1, res.first );
EXPECT_EQ( "Output file is not specified", res.second );

END_TEST

BEGIN_TEST( CliParser, NothingToStack )

std::vector<const char*> data = {  "-stack", "-output", "test.txt"};
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 1, res.first );
EXPECT_EQ( "Nothing to stack", res.second );

END_TEST

BEGIN_TEST( CliParser, StackOneFile )

std::string path = GetPathToTestFile( "RAW/IMG_8899.CR2" );
std::vector<const char*> data = { "-stack", path.data(), "-output", "test.txt"};
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 0, res.first );
EXPECT_EQ( 1, parser._decoders.size() );

END_TEST

BEGIN_TEST( CliParser, StackTwoFiles )

std::string fileNames = GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) + ";" + GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" );
std::vector<const char*> data = { "-stack", fileNames.data(), "-output", "test.txt" };
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 0, res.first );
EXPECT_EQ( 2, parser._decoders.size() );

END_TEST

BEGIN_TEST( CliParser, StackWrongDirectory )

std::string dirName = GetPathToTestFile( "RAW/MilkyWayCR4/" );
std::vector<const char*> data = { "-stack", "-dir", dirName.data(), "-output", "test.txt" };
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 1, res.first );
EXPECT_EQ( "No such directory", res.second );

END_TEST

BEGIN_TEST( CliParser, StackDirectory )

std::string dirName = GetPathToTestFile( "RAW/MilkyWayCR2/" );
std::vector<const char*> data = { "-stack", "-dir", dirName.data(), "-output", "test.txt"};
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 0, res.first );
EXPECT_EQ( 27, parser._decoders.size() );

END_TEST

BEGIN_TEST( CliParser, StackRange )

std::string fileRange = GetPathToTestFile( "RAW/MilkyWayCR2/" ) + "IMG_8947~63.CR2";
std::vector<const char*> data = { "-stack", fileRange.data(), "-output", "test.txt" };
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 0, res.first );
EXPECT_EQ( 17, parser._decoders.size() );

END_TEST

BEGIN_TEST( CliParser, ManySemicolons)

std::vector<const char*> data = { "-stack", ";;;;;;;;;;", "-output", "test.txt" };
CliParser parser( data.size(), ( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 1, res.first );
EXPECT_EQ( "Nothing to stack", res.second );

END_TEST

END_SUITE( CliParser )