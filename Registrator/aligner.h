#ifndef ALIGNER_H
#define ALIGNER_H
#include <memory>
#include <vector>
#include "star.h"
#include "AGG/agg_trans_affine.h"


class ImageDecoder;
struct AlignmentDataset;

using StarPair = std::pair<Star, Star>;
using PointFPair = std::pair<PointF, PointF>;

class Aligner
{
    double _threshold = 10;
    uint32_t _minStarSize = 5;
    uint32_t _maxStarSize = 25;

    std::vector<std::shared_ptr<ImageDecoder>> _decoders;


    Aligner(std::vector<std::shared_ptr<ImageDecoder>> decoders);

    std::vector<std::shared_ptr<AlignmentDataset>> Align();

    void ProcessPairOfDatasets(std::shared_ptr<AlignmentDataset> ref, std::shared_ptr<AlignmentDataset> target);
    bool CheckPairOfDatasets(std::shared_ptr<AlignmentDataset> ref, std::shared_ptr<AlignmentDataset> target);
    agg::trans_affine CalculateTransform(PointFPair& refPoints,PointFPair& targetPoints);

public:
    static std::vector<std::shared_ptr<AlignmentDataset>> Align(std::vector<std::shared_ptr<ImageDecoder>> decoders);
};

#endif // ALIGNER_H
