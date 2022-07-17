#include "Tools/CliParser.h"

int main(int argc, char** argv)
{
    auto res = CliParser::Parse( argc, argv );
    if ( !res.second.empty() )
        std::cout << res.second << std::endl;

    return res.first;
}
