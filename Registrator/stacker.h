#ifndef STACKER_H
#define STACKER_H
#include <memory>
#include <vector>
#include <string>
class ImageDecoder;

class Stacker
{
    std::vector<std::shared_ptr<ImageDecoder>> _decoders;
public:
    Stacker(const std::vector<std::string>& fileNames);

};

#endif // STACKER_H
