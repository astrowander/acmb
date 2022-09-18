#pragma once
#include "./../Core/macros.h"
#include <memory>
#include <string>

ACMB_NAMESPACE_BEGIN
class IBitmap;
ACMB_NAMESPACE_END

ACMB_TESTS_NAMESPACE_BEGIN

bool BitmapsAreEqual(std::shared_ptr<IBitmap> lhs, std::shared_ptr<IBitmap> rhs);
bool BitmapsAreEqual(const std::string& fileName, std::shared_ptr<IBitmap> rhs);

std::string GetPathToTestFile(const std::string& fileName);
std::string GetPathToPattern(const std::string& fileName);

ACMB_TESTS_NAMESPACE_END