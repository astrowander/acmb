#include "../Tests/TestRunner.h"

int main()
{
    return acmb::tests::TestRunner::RunAllTests() ? 0 : 1;
}
