#include "../Tools/CliParser.h"

int main(int argc, const char** argv)
{
    try
    {
        auto [res, errMsg] = acmb::CliParser::Parse( argc, argv );
        if ( !errMsg.empty() )
            std::cout << errMsg << std::endl;
        return res;
    }
    catch (std::exception& e )
    {
        std::cout << e.what() << std::endl;
    }

    return 1;
}
