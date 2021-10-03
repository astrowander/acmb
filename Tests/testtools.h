#ifndef TESTTOOLS_H
#define TESTTOOLS_H

#include <memory>
#include <string>

class IBitmap;
bool BitmapsAreEqual(std::shared_ptr<IBitmap> lhs, std::shared_ptr<IBitmap> rhs);
bool BitmapsAreEqual(const std::string& fileName, std::shared_ptr<IBitmap> rhs);

const std::string GetPathToTestFile(const std::string& fileName);
const std::string GetPathToPattern(const std::string& fileName);



#endif // TESTTOOLS_H
