#include "test.h"
#include "testtools.h"
#include "../Registrator/FastAligner.h"
#include "../Registrator/registrator.h"

BEGIN_SUITE(FastAligner)

BEGIN_TEST(FastAligner, BasicTest)

std::vector<Star> refStars(5);
refStars[0].center = { 70, 20 };
refStars[0].rect = { 70, 20, 1, 1 };

refStars[1].center = { 55, 35 };
refStars[1].rect = { 55, 35, 1, 1 };

refStars[2].center = { 40, 20 };
refStars[2].rect = { 40, 20, 1, 1 };

refStars[3].center = { 30, 30 };
refStars[3].rect = { 30, 30, 1, 1 };

refStars[4].center = { 20, 20 };
refStars[4].rect = { 20, 20, 1, 1 };

std::vector<Star> targetStars(6);

targetStars[0].center = { 10, 15 };
targetStars[0].rect = { 10, 15, 1, 1 };

targetStars[1].center = { 30, 25 };
targetStars[1].rect = { 30, 25, 1, 1 };

targetStars[2].center = { 20, 35 };
targetStars[2].rect = { 20, 35, 1, 1 };

targetStars[3].center = { 30, 45 };
targetStars[3].rect = { 30, 45, 1, 1 };

targetStars[4].center = { 15, 60 };
targetStars[4].rect = { 15, 60, 1, 1 };

targetStars[5].center = { 30, 75 };
targetStars[5].rect = { 30, 75, 1, 1 };

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
refStars[0].center = { 10, 15 };
refStars[0].rect = { 10, 15, 1, 1 };

refStars[1].center = { 30, 25 };
refStars[1].rect = { 30, 25, 1, 1 };

refStars[2].center = { 20, 35 };
refStars[2].rect = { 20, 35, 1, 1 };

refStars[3].center = { 30, 45 };
refStars[3].rect = { 30, 45, 1, 1 };

refStars[4].center = { 15, 60 };
refStars[4].rect = { 10, 60, 1, 1 };

refStars[5].center = { 30, 75 };
refStars[5].rect = { 30, 70, 1, 1 };

std::vector<Star> targetStars(5);
targetStars[0].center = { 70, 20 };
targetStars[0].rect = { 70, 20, 1, 1 };

targetStars[1].center = { 55, 35 };
targetStars[1].rect = { 55, 35, 1, 1 };

targetStars[2].center = { 40, 20 };
targetStars[3].rect = { 40, 20, 1, 1 };

targetStars[3].center = { 30, 30 };
targetStars[3].rect = { 30, 30, 1, 1 };

targetStars[4].center = { 20, 20 };
targetStars[4].rect = { 20, 20, 1, 1 };

FastAligner fastAligner(refStars);
fastAligner.Align(targetStars);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(5, matches.size());
EXPECT_EQ(5, matches.at(0));
EXPECT_EQ(4, matches.at(1));
EXPECT_EQ(2, matches.at(3));
EXPECT_EQ(3, matches.at(2));
EXPECT_EQ(1, matches.at(4));
END_TEST

BEGIN_TEST(FastAligner, RealPhotoTest)

auto pRefBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2"));
auto pRegistrator = std::make_unique<Registrator>(70);
auto refStars = pRegistrator->Registrate(pRefBitmap);
pRefBitmap.reset();

auto pTargetBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2"));
auto targetStars = pRegistrator->Registrate(pTargetBitmap);
pTargetBitmap.reset();

FastAligner fastAligner(refStars);
fastAligner.Align(targetStars);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(308, matches.size());


END_TEST



END_SUITE