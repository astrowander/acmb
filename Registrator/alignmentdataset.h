#ifndef ALIGNMENTDATASET_H
#define ALIGNMENTDATASET_H
#include "../AGG/agg_trans_affine.h"
#include <memory>
#include <vector>

#include "star.h"

class ImageDecoder;

struct AlignmentDataset
{
    std::vector<Star> stars;
    agg::trans_affine transform;
    uint32_t starCount = 0;
};

#endif // ALIGNMENTDATASET_H
