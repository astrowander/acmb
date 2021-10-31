#ifndef ALIGNER_H
#define ALIGNER_H
#include <memory>
#include <vector>
#include "star.h"
#include "../AGG/agg_trans_affine.h"


class IBitmap;
struct AlignmentDataset;

using StarPair = std::pair<Star, Star>;
using PointFPair = std::pair<PointF, PointF>;

class Aligner
{
    std::shared_ptr<AlignmentDataset> _pRefDataset;
    std::shared_ptr<AlignmentDataset> _pTargetDataset;

    Aligner(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset);

    void Align();
    bool CheckTransform();
    bool TryRefPair(const std::pair<Star, Star>& refPair);
    agg::trans_affine CalculateTransform(PointFPair& refPoints,PointFPair& targetPoints);

public:
    static void Align(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset);
};

#endif // ALIGNER_H
