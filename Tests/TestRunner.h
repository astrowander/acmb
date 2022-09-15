#pragma once

#include <unordered_map>
#include <string>
#include <memory>

class Suite;

class TestRunner
{
    //inline static auto _suites = std::unordered_map<std::string, std::shared_ptr<Suite>>();

private:
    std::unordered_map<std::string, std::shared_ptr<Suite>> _suites;

    TestRunner() = default;
    TestRunner( const TestRunner& ) = delete;
    TestRunner( TestRunner&& ) = delete;

public:

    static TestRunner& GetInstance()
    {
        static TestRunner testRunner;
        return testRunner;
    }
    static bool RunAllTests();

    static bool RunSuite( const std::string& suiteName );

    static bool RunTest( const std::string& suiteName, const std::string& testName );

    static bool AddSuite( const std::string& suiteName, std::shared_ptr<Suite> pSuite );
};

