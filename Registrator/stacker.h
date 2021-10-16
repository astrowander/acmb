#ifndef STACKER_H
#define STACKER_H
#include <memory>
#include <vector>
#include <string>

class ImageDecoder;
class IBitmap;
struct AlignmentDataset;

class Stacker
{
    std::vector<std::shared_ptr<ImageDecoder>> _decoders;
    std::vector<std::shared_ptr<AlignmentDataset>> _datasets;

public:
    Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders);
    std::shared_ptr<IBitmap> Stack();

};

#endif // STACKER_H
