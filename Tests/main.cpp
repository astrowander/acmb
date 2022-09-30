#include "TestRunner.h"

int main()
{
    return !acmb::tests::TestRunner::RunSuite( "CliParser" );
}
