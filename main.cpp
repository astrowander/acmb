#include "Tests/TestRunner.h"

int main()
{
    auto& testRunner = TestRunner::GetInstance();
    testRunner.RunAllTestsInSuite("Converter");
    return 0;
}
