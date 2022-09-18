#include "../Tools/CliParser.h"

int main(int argc, const char** argv)
{
    auto [res, errMsg] = acmb::CliParser::Parse(argc, argv);
    if ( !errMsg.empty() )
        std::cout << errMsg << std::endl;

    return res;
}
