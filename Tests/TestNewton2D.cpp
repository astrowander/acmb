#include "test.h"
//#include "testtools.h"
#include "./../Tools/Newton2D.h"

BEGIN_SUITE(Newton2D)

BEGIN_TEST(Newton2D, TestNumericalJacobian)

auto res = Newton2D::Solve
(
	{
		[](auto x) {return 3 * x[0] * x[0] + 2 * x[1] * x[1] - 25; },
		[](auto x) {return 2 * x[0] * x[0] - x[1] - 15; }
	},	
	{
		1.0,
		1.0
	}
	);

EXPECT_NEAR(2.81, res[0], 0.01);
EXPECT_NEAR(0.81, res[1], 0.01);

END_TEST

BEGIN_TEST(Newton2D, TestAnaliticalJacobian)

auto res = Newton2D::Solve
(
	{
		[](auto x) {return 3 * x[0] * x[0] + 2 * x[1] * x[1] - 25; },
		[](auto x) {return 2 * x[0] * x[0] - x[1] - 15; }
	},
	
	{		
			[](auto x) {return 6 * x[0]; },
			[](auto x) {return 4 * x[1]; },		
			[](auto x) {return 4 * x[0]; },
            [](auto) {return -1; }
	},
	{
		1.0,
		1.0
	}
);

EXPECT_NEAR(2.81, res[0], 0.01);
EXPECT_NEAR(0.81, res[1], 0.01);

END_TEST

END_SUITE (Newton2D)
