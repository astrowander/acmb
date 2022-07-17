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
CliParser parser(data.size(), const_cast<char**>(&data[0]) );
auto res = parser.Parse( true );
EXPECT_EQ(1, res.first);
EXPECT_EQ( "Nothing to do", res.second);

END_TEST

BEGIN_TEST( CliParser, NoOutput )

std::vector<const char*> data = { "AstroCombiner", "-stack" };
CliParser parser( data.size(), const_cast< char** >( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 1, res.first );
EXPECT_EQ( "Output file is not specified", res.second );

END_TEST

BEGIN_TEST( CliParser, NothingToStack )

std::vector<const char*> data = { "AstroCombiner", "-stack", "-output", "test.txt"};
CliParser parser( data.size(), const_cast< char** >( &data[0] ) );
auto res = parser.Parse( true );
EXPECT_EQ( 1, res.first );
EXPECT_EQ( "Nothing to stack", res.second );

END_TEST

END_SUITE( CliParser )