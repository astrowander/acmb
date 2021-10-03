#ifndef TESTTOOLS_H
#define TESTTOOLS_H

#include <memory>

class IBitmap;
bool BitmapsAreEqual(std::shared_ptr<IBitmap> lhs, std::shared_ptr<IBitmap> rhs);
bool BitmapsAreEqual(const std::string& fileName, std::shared_ptr<IBitmap> rhs);

#endif // TESTTOOLS_H
