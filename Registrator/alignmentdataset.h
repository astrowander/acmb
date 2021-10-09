#ifndef ALIGNMENTDATASET_H
#define ALIGNMENTDATASET_H
#include "AGG/agg_trans_affine.h"
#include <memory>
#include <vector>

#include "star.h"

class ImageDecoder;

struct AlignmentDataset
{
    std::shared_ptr<ImageDecoder> pDecoder;
    std::vector<Star> stars;
    agg::trans_affine transform;
    uint32_t valuableStarCount = 20;

    AlignmentDataset(std::shared_ptr<ImageDecoder> pDecoder)
    :pDecoder(pDecoder)
    {}
};

#endif // ALIGNMENTDATASET_H
