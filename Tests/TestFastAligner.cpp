#include "test.h"
#include "testtools.h"
#include "../Registrator/FastAligner.h"

BEGIN_SUITE(FastAligner)

BEGIN_TEST(FastAligner, BasicTest)

std::vector<Star> refStars(5);
refStars[0].center = { 7.0, 2.0 };
refStars[1].center = { 5.5, 3.5 };
refStars[2].center = { 4.0, 2.0 };
refStars[3].center = { 3.0, 3.0 };
refStars[4].center = { 2.0, 2.0 };

std::vector<Star> targetStars(6);

targetStars[0].center = { 1.0, 1.5 };
targetStars[1].center = { 3.0, 2.5 };
targetStars[2].center = { 2.0, 3.5 };
targetStars[3].center = { 3.0, 4.5 };
targetStars[4].center = { 1.5, 6.0 };
targetStars[5].center = { 3.0, 7.5 };

FastAligner fastAligner(refStars);
fastAligner.SetEps(0.1);
fastAligner.Align(targetStars);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(5, matches.size());
EXPECT_EQ(0, matches.at(5));
EXPECT_EQ(1, matches.at(4));
EXPECT_EQ(2, matches.at(3));
EXPECT_EQ(3, matches.at(2));
EXPECT_EQ(4, matches.at(1));
END_TEST

BEGIN_TEST(FastAligner, InverseMatching)

std::vector<Star> refStars(6);
refStars[0].center = { 1.0, 1.5 };
refStars[1].center = { 3.0, 2.5 };
refStars[2].center = { 2.0, 3.5 };
refStars[3].center = { 3.0, 4.5 };
refStars[4].center = { 1.5, 6.0 };
refStars[5].center = { 3.0, 7.5 };

std::vector<Star> targetStars(5);
targetStars[0].center = { 7.0, 2.0 };
targetStars[1].center = { 5.5, 3.5 };
targetStars[2].center = { 4.0, 2.0 };
targetStars[3].center = { 3.0, 3.0 };
targetStars[4].center = { 2.0, 2.0 };


FastAligner fastAligner(refStars);
fastAligner.SetEps(0.1);
fastAligner.Align(targetStars);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(5, matches.size());
EXPECT_EQ(5, matches.at(0));
EXPECT_EQ(4, matches.at(1));
EXPECT_EQ(2, matches.at(3));
EXPECT_EQ(3, matches.at(2));
EXPECT_EQ(1, matches.at(4));
END_TEST



END_SUITE