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

END_SUITE( CliParser )