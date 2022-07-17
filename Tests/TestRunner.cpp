#include "TestRunner.h"
#include "test.h"

bool TestRunner::RunAllTests()
{
    bool isTrue{ true };
    for ( auto& it : _suites )
        isTrue &= it.second->RunAll();
    return isTrue;
}

bool TestRunner::RunSuite( const std::string& suiteName )
{
    auto it = _suites.find( suiteName );
    if ( it == std::end( _suites ) )
    {
        std::cout << "Suite not found" << std::endl;
        return false;
    }
    return it->second->RunAll();
}

bool TestRunner::RunTest( const std::string& suiteName, const std::string& testName )
{
    auto it = _suites.find( suiteName );
    if ( it == std::end( _suites ) )
    {
        std::cout << "Suite not found" << std::endl;
        return false;
    }
    return it->second->RunTest( testName );
}

const bool TestRunner::AddSuite( const std::string& suiteName, std::shared_ptr<Suite> pSuite )
{
    if ( _suites.find(suiteName) != std::end( _suites ) )
    {
        std::cout << "Suite " << suiteName << " already exists" << std::endl;
        return false;
    }

    _suites.insert( { suiteName, pSuite } );
    return true;
}

TestRunner& TestRunner::GetInstance()
{
    static TestRunner instance;
    return instance;
}
