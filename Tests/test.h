#ifndef TEST_H
#define TEST_H
#include <iomanip>
#include <iostream>
#include <chrono>

#define BEGIN_SUITE( TestSuite ) \
class Test##TestSuite \
{\
public:

#define END_SUITE \
};

#define TEST_ACCESS(TestSuite) \
    friend class Test##TestSuite;

#define BEGIN_TEST(TestSuite, TestName)                                    \
   static bool test__##TestSuite##__##TestName(void)                              \
{                                                                          \
      bool isTrue{true};

#define END_TEST                                                           \
   return isTrue;                                                          \
}

#define EXPECT_EQ(arg1, arg2) isTrue &= (arg1 == arg2); \
                              if (arg1 != arg2) \
                                std::cout << "Expected " << arg1 << ", but was " << arg2 << std::endl;

#define EXPECT_TRUE(arg) isTrue &= arg;
#define EXPECT_FALSE(arg) isTrue &= !arg;
#define EXPECT_NEAR(arg1, arg2, eps) isTrue &= (arg1 > arg2 - eps) && (arg1 < arg2 + eps); \
                                        if ((arg1 < arg2 - eps) || (arg1 > arg2 + eps)) \
                                std::cout << "Expected " << arg1 << " +/- " << eps << ", but was " << arg2 << std::endl;

#define RUN_TEST(TestSuite, TestName)                                      \
{                                                                          \
   auto startTime = std::chrono::system_clock::now();\
   bool ret = true;\
    std::cout << std::left << std::setfill('-')                             \
   << std::setw(50) << #TestSuite " --> " #TestName " started" << std::endl; \
   try                                                                     \
   {                                                                        \
    ret = Test##TestSuite::test__##TestSuite##__##TestName();          \
   }                                                                        \
   catch (std::exception& e)                                         \
   {\
        std::cout << "Exception has been thrown. Message: " << e.what() << std::endl;\
        ret = false;\
   }\
   auto elapsed = std::chrono::system_clock::now() - startTime;            \
   std::cout << std::left << std::setfill('-')                             \
   << std::setw(50) << #TestSuite " --> " #TestName " " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() \
    << "s " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() % 1000 << "ms ";                   \
                                                                           \
   if(ret)                                                                 \
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
} /* Coloring valid for *nix systems. */
#endif // TEST_H
