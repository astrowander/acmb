#ifndef ALIGNMENTDATASET_H
#define ALIGNMENTDATASET_H
#include "AGG/agg_trans_affine.h"
#include <memory>
#include <vector>

#include "star.h"

class IBitmap;

struct AlignmentDataset
{
    std::shared_ptr<IBitmap> pBitmap;
    std::vector<Star> stars;
    agg::trans_affine transform;
    uint32_t valuableStarCount = 20;

    AlignmentDataset(std::shared_ptr<IBitmap> pBitmap)
    :pBitmap(pBitmap)
    {}
};

#endif // ALIGNMENTDATASET_H
