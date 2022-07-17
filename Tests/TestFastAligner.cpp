#include "test.h"
#include "testtools.h"
#include "../Registrator/FastAligner.h"
#include "../Registrator/registrator.h"

BEGIN_SUITE(FastAligner)

BEGIN_TEST(FastAligner, BasicTest)

PointF refPoints[5] = { {70, 20}, {55, 35}, {40, 20}, {30, 30}, {20,20} };
PointF targetPoints[6] = { {10, 15}, {30, 25}, {20, 35}, {30, 45}, {15, 60}, {30, 75} };

std::vector<Star> refStars(5);
refStars[0].center = refPoints[0];
refStars[0].rect = { 70, 20, 1, 1 };

refStars[1].center = refPoints[1];
refStars[1].rect = { 55, 35, 1, 1 };

refStars[2].center = refPoints[2];
refStars[2].rect = { 40, 20, 1, 1 };

refStars[3].center = refPoints[3];
refStars[3].rect = { 30, 30, 1, 1 };

refStars[4].center = refPoints[4];
refStars[4].rect = { 20, 20, 1, 1 };

std::vector<Star> targetStars(6);

targetStars[0].center = targetPoints[0];
targetStars[0].rect = { 10, 15, 1, 1 };

targetStars[1].center = targetPoints[1];
targetStars[1].rect = { 30, 25, 1, 1 };

targetStars[2].center = targetPoints[2];
targetStars[2].rect = { 20, 35, 1, 1 };

targetStars[3].center = targetPoints[3];
targetStars[3].rect = { 30, 45, 1, 1 };

targetStars[4].center = targetPoints[4];
targetStars[4].rect = { 15, 60, 1, 1 };

targetStars[5].center = targetPoints[5];
targetStars[5].rect = { 30, 75, 1, 1 };

FastAligner fastAligner(refStars);
fastAligner.SetEps(0.1);
fastAligner.Align(targetStars);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(5, matches.size());
EXPECT_EQ(refPoints[0], matches.at(targetPoints[5]));
EXPECT_EQ(refPoints[1], matches.at(targetPoints[4]));
EXPECT_EQ(refPoints[2], matches.at(targetPoints[3]));
EXPECT_EQ(refPoints[3], matches.at(targetPoints[2]));
EXPECT_EQ(refPoints[4], matches.at(targetPoints[1]));
END_TEST

BEGIN_TEST(FastAligner, InverseMatching)

PointF targetPoints[5] = { {70, 20}, {55, 35}, {40, 20}, {30, 30}, {20,20} };
PointF refPoints[6] = { {10, 15}, {30, 25}, {20, 35}, {30, 45}, {15, 60}, {30, 75} };

std::vector<Star> refStars(6);
refStars[0].center = refPoints[0];
refStars[0].rect = { 10, 15, 1, 1 };

refStars[1].center = refPoints[1];
refStars[1].rect = { 30, 25, 1, 1 };

refStars[2].center = refPoints[2];
refStars[2].rect = { 20, 35, 1, 1 };

refStars[3].center = refPoints[3];
refStars[3].rect = { 30, 45, 1, 1 };

refStars[4].center = refPoints[4];
refStars[4].rect = { 10, 60, 1, 1 };

refStars[5].center = refPoints[5];
refStars[5].rect = { 30, 70, 1, 1 };

std::vector<Star> targetStars(5);
targetStars[0].center = targetPoints[0];
targetStars[0].rect = { 70, 20, 1, 1 };

targetStars[1].center = targetPoints[1];
targetStars[1].rect = { 55, 35, 1, 1 };

targetStars[2].center = targetPoints[2];
targetStars[3].rect = { 40, 20, 1, 1 };

targetStars[3].center = targetPoints[3];
targetStars[3].rect = { 30, 30, 1, 1 };

targetStars[4].center = targetPoints[4];
targetStars[4].rect = { 20, 20, 1, 1 };

FastAligner fastAligner(refStars);
fastAligner.Align(targetStars, 1.0);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(5, matches.size());
EXPECT_EQ(refPoints[5], matches.at(targetPoints[0]));
EXPECT_EQ(refPoints[4], matches.at(targetPoints[1]));
EXPECT_EQ(refPoints[3], matches.at(targetPoints[2]));
EXPECT_EQ(refPoints[2], matches.at(targetPoints[3]));
EXPECT_EQ(refPoints[1], matches.at(targetPoints[4]));
END_TEST

BEGIN_TEST(FastAligner, RealPhotoTest)

auto pRefBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2"));
auto pRegistrator = std::make_unique<Registrator>(70);
pRegistrator->Registrate(pRefBitmap);
auto refStars = pRegistrator->GetStars();
pRefBitmap.reset();

auto pTargetBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2"));
pRegistrator->Registrate(pTargetBitmap);
auto targetStars = pRegistrator->GetStars();
pTargetBitmap.reset();

FastAligner fastAligner(refStars[0]);
fastAligner.Align(targetStars[0], 1.0);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(39, matches.size());


END_TEST

BEGIN_TEST(FastAligner, TestLargeIntervalPhotos)

auto pRefBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2"));
auto pRegistrator = std::make_unique<Registrator>(40);
pRegistrator->Registrate(pRefBitmap);
auto refStars = pRegistrator->GetStars();
pRefBitmap.reset();

auto pTargetBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8970.CR2"));
pRegistrator->Registrate(pTargetBitmap);
auto targetStars = pRegistrator->GetStars();
pTargetBitmap.reset();

FastAligner fastAligner(refStars[0]);
fastAligner.Align(targetStars[0]);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(18, matches.size());


END_TEST

BEGIN_TEST(FastAligner, TestThreshold60)

auto pRefBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8944.CR2"));
auto pRegistrator = std::make_unique<Registrator>(60);
pRegistrator->Registrate(pRefBitmap);
auto refStars = pRegistrator->GetStars();
pRefBitmap.reset();

auto pTargetBitmap = IBitmap::Create(GetPathToTestFile("RAW/MilkyWayCR2/IMG_8945.CR2"));
pRegistrator->Registrate(pTargetBitmap);
auto targetStars = pRegistrator->GetStars();
pTargetBitmap.reset();

FastAligner fastAligner(refStars[0]);
fastAligner.Align(targetStars[0]);
const auto& matches = fastAligner.GetMatches();
EXPECT_EQ(61, matches.size());

END_TEST



END_SUITE (FastAligner)