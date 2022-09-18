#pragma once

#include "TestRunner.h"
#include <iomanip>
#include <iostream>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <vector>

ACMB_TESTS_NAMESPACE_BEGIN

class Suite
{
protected:
    
public:
    virtual ~Suite() = default;
    virtual std::vector<std::string> RunAll() = 0;
    virtual bool RunTest(std::string testName) = 0;
    virtual size_t GetTestCount() const = 0;
};

#define BEGIN_SUITE( TestSuite ) \
class Test##TestSuite : public Suite \
{\
inline static auto _tests = std::unordered_map<std::string, std::function<bool( void )>>();\
inline static bool handle = TestRunner::GetInstance().AddSuite(#TestSuite, std::make_shared<Test##TestSuite>());\
public:\
Test##TestSuite()\
{\
const std::string suiteName(#TestSuite);\
std::string testName;

#define END_SUITE( TestSuite ) \
}\
virtual std::vector<std::string> RunAll() override\
{\
std::vector<std::string> res;\
for (auto&test:_tests)\
{\
if (!test.second())\
res.push_back(test.first);\
}\
return res;}\
virtual bool RunTest(std::string testName) override\
{\
    auto it = _tests.find(testName);\
    if (it == std::end(_tests))\
    {\
    std::cout << "Test not found" << std::endl;\
    return false;\
    }\
    return it->second();\
}\
virtual size_t GetTestCount() const override\
{\
return _tests.size();\
}\
};

#define TEST_ACCESS(TestSuite) \
    friend class tests::Test##TestSuite;

#define BEGIN_TEST(TestName)                                    \
testName = std::string(#TestName); \
   _tests[#TestName] =  [testName, suiteName]                          \
{                                                                          \
      bool isTrue{true}; \
auto startTime = std::chrono::system_clock::now();\
const std::string testStr = suiteName + std::string(" --> ") + testName + std::string(" started");\
    std::cout << std::left << std::setfill('-')                             \
   << std::setw(50) << testStr << std::endl; \
   try                                                                     \
   {                                                                        \

#define END_TEST                                                           \
 }                                                                        \
   catch (std::exception& e)                                         \
   {\
        std::cout << "Exception has been thrown. Message: " << e.what() << std::endl;\
        isTrue = false;\
   }\
   auto elapsed = std::chrono::system_clock::now() - startTime;            \
   std::cout << std::left << std::setfill('-')                             \
   << std::setw(50) << testName << " " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() \
    << "s " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() % 1000 << "ms ";                   \
                                                                           \
   if(isTrue)                                                                 \
   {                                                                       \
      std::cout << std::setw(10)                                           \
      << std::left << "\x1b[38;5;40m   OK \x1b[0m" /* colored in Green*/   \
      << std::endl;                                                        \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      std::cout << std::setw(10)                                           \
      << std::left << "\x1b[38;5;160m   FAILED \x1b[0m" /* colored in Red*/\
      << std::endl;                                                        \
   }                                                                       \
   return isTrue;                                                          \
};

#define EXPECT_EQ(arg1, arg2) isTrue &= (arg1 == arg2); \
                              if (arg1 != arg2) \
                                std::cout << "Expected " << arg1 << ", but was " << arg2 << std::endl;

#define EXPECT_TRUE(arg) isTrue &= arg;
#define EXPECT_FALSE(arg) isTrue &= !arg;
#define EXPECT_NEAR(arg1, arg2, eps) isTrue &= (arg1 > arg2 - eps) && (arg1 < arg2 + eps); \
                                        if ((arg1 < arg2 - eps) || (arg1 > arg2 + eps)) \
                                std::cout << "Expected " << arg1 << " +/- " << eps << ", but was " << arg2 << std::endl;
#define ASSERT_NO_THROW(f) try { f(); } catch(...) { isTrue = false;}
#define ASSERT_THROWS(f, ExceptionType) try {f();} catch(ExceptionType&) {isTrue &= true;}

ACMB_TESTS_NAMESPACE_END
