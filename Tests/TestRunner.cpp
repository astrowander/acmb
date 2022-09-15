#include "TestRunner.h"
#include "test.h"

bool TestRunner::RunAllTests()
{
    auto startTime = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, std::string>> allFailedTests;
    size_t testCount = 0;

    for ( auto& it : GetInstance()._suites )
    {
        auto failedTests = it.second->RunAll();
        testCount += it.second->GetTestCount();

        for ( const auto& failedTest : failedTests )
        {
            allFailedTests.push_back( { it.first, failedTest } );
        }
    }

    std::cout << std::endl << std::left << std::setfill( '-' ) << std::setw( 50 ) << "RESULTS" << std::endl;
    std::cout << testCount << " tests runned " << std::endl;
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    std::cout << "Elapsed in total:" << std::chrono::duration_cast< std::chrono::seconds >( elapsed ).count()
        << "s " << std::chrono::duration_cast< std::chrono::milliseconds >( elapsed ).count() % 1000 << "ms " << std::endl;
    if ( allFailedTests.empty() )
    {
        std::cout << "\x1b[38;5;40mSUCCESS \x1b[0m" << std::endl; // colored in Green
        return true;
    }
   
    std::cout << "\x1b[38;5;160m" << allFailedTests.size() << " FAILED TESTS: \x1b[0m" << std::endl;// colored in Red
    for ( const auto& failedTest : allFailedTests )
    {
        std::string testStr = failedTest.first + std::string( " --> " ) + failedTest.second;
        std::cout << std::left << std::setfill( '-' )
            << std::setw( 50 ) << testStr << std::endl;
    }
   

    return false;
}

bool TestRunner::RunSuite( const std::string& suiteName )
{
    auto startTime = std::chrono::steady_clock::now();

    auto it = GetInstance()._suites.find( suiteName );
    if ( it == std::end( GetInstance()._suites ) )
    {
        std::cout << "Suite not found" << std::endl;
        return false;
    }
    
    auto failedTests =  it->second->RunAll();    
    std::cout << std::endl << std::left << std::setfill( '-' )  << std::setw( 50 ) << "RESULTS" << std::endl;
    std::cout << it->second->GetTestCount() << " tests runned " << std::endl;
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    std::cout << "Elapsed in total:" << std::chrono::duration_cast< std::chrono::seconds >( elapsed ).count()
        << "s " << std::chrono::duration_cast< std::chrono::milliseconds >( elapsed ).count() % 1000 << "ms " << std::endl;
    if ( failedTests.empty() )
    {
        std::cout << "\x1b[38;5;40mSUCCESS \x1b[0m" << std::endl; // colored in Green
        return true;
    }
    
    std::cout << "\x1b[38;5;160m" << failedTests.size() << " FAILED TESTS: \x1b[0m"  << std::endl;// colored in Red
    for ( const auto& failedTest : failedTests )
    {
        std::string testStr = suiteName + std::string( " --> " ) + failedTest; 
        std::cout << std::left << std::setfill( '-' )
            << std::setw( 50 ) << testStr << std::endl;
    }

   

    return false;
}

bool TestRunner::RunTest( const std::string& suiteName, const std::string& testName )
{
    auto it = GetInstance()._suites.find( suiteName );
    if ( it == std::end( GetInstance()._suites ) )
    {
        std::cout << "Suite not found" << std::endl;
        return false;
    }
    return it->second->RunTest( testName );
}

bool TestRunner::AddSuite( const std::string& suiteName, std::shared_ptr<Suite> pSuite )
{
    if ( GetInstance()._suites.find(suiteName) != std::end( GetInstance()._suites ) )
    {
        std::cout << "Suite " << suiteName << " already exists" << std::endl;
        return false;
    }

    GetInstance()._suites.insert( { suiteName, pSuite } );
    return true;
}
