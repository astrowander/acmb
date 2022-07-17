#include "Tests/TestRunner.h"
#include <iostream>

int main(int argc, char** argv)
{
    std::unordered_map<std::string, std::string> kv;

    int i = 0;
    while ( i < argc )
    {
        if ( argv[i][0] != '-' )
        {
            ++i;
            continue;
        }

        if ( i == argc - 1  || argv[i + 1][0] == '-' )
        {
            kv.insert( { argv[i], "" });
            ++i;
            continue;
        }

        kv.insert( { argv[i], argv[i + 1] } );
        i += 2;
    }

    auto it = kv.find( "-runtests" );
    if ( it != std::end( kv ) )
    {
        it = kv.find( "-suite" );
        if ( it == std::end( kv ) )
        {
            TestRunner::GetInstance().RunAllTests();
            return 0;
        }

        std::string suite = it->second;

        it = kv.find( "-test" );
        
        if ( it == std::end( kv ) )
        {
            TestRunner::GetInstance().RunAllTestsInSuite( suite );
            return 0;
        }

        TestRunner::GetInstance().RunTest( suite, it->second );
        return 0;
    }
    return 0;
}
