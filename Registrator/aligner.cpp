#include "aligner.h"
#include "registrator.h"

Aligner::Aligner(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset)
: _pRefDataset(pRefDataset)
, _pTargetDataset(pTargetDataset)
{}

void Aligner::Align()
{
    std::pair<Star, Star> refPair { _pRefDataset->stars[0], _pRefDataset->stars[1] };
    std::pair<PointF, PointF> refPoints{refPair.first.center, refPair.second.center};
    auto refDist = refPoints.first.Distance(refPoints.second);

    std::pair<Star, Star> targetPair { Star{}, Star{} };
    std::pair<PointF, PointF> targetPoints{ PointF{}, PointF{} };
    for (uint32_t i = 0; i < _pTargetDataset->valuableStarCount - 1; ++i)
    {
        targetPair.first = _pTargetDataset->stars[i];
        targetPoints.first = targetPair.first.center;

        for (uint32_t j = i + 1; j < _pTargetDataset->valuableStarCount; ++j)
        {
            targetPair.second = _pTargetDataset->stars[j];
            targetPoints.second = targetPair.second.center;
            auto targetDist = targetPoints.first.Distance(targetPoints.second);

            //BUGBUG magic number
            if (fabs(targetDist - refDist) > 5)
                continue;

            _pTargetDataset->transform = CalculateTransform(refPoints, targetPoints);

            if (CheckTransform())
                return;

            std::swap(targetPoints.first, targetPoints.second);
            _pTargetDataset->transform = CalculateTransform(refPoints, targetPoints);

            if (CheckTransform())
                return;
        }
    }
}

bool Aligner::CheckTransform()
{
    uint32_t matches = 0;

    for (uint32_t i = 0; i < _pRefDataset->valuableStarCount; ++i)
    {
        auto refPoint = _pRefDataset->stars[i].center;
        _pTargetDataset->transform.transform(&refPoint.x, &refPoint.y);

        for (uint32_t j = 0; j < _pTargetDataset->valuableStarCount; ++j)
        {
            auto targetPoint = _pTargetDataset->stars[j].center;
            auto dist = refPoint.Distance(targetPoint);
            //BUGBUG magic number
            if (dist < 5)
            {
                ++matches;
                break;
            }
        }
    }

    return (matches > 2);
}

agg::trans_affine Aligner::CalculateTransform(PointFPair &refPoints, PointFPair &targetPoints)
{
    auto dx = refPoints.first.x - refPoints.second.x;
    auto dy = refPoints.first.y - refPoints.second.y;
    double det = dx * dx + dy * dy;

    auto dx1 = targetPoints.first.x - targetPoints.second.x;
    auto dy1 = targetPoints.first.y - targetPoints.second.y;

    double cosa = (dx1 * dx + dy1 * dy) / det;
    double sina = (dx1 * dy1 - dx1 * dy) / det;

    return agg::trans_affine (cosa, sina, -sina, cosa, targetPoints.first.x + sina * refPoints.first.y - cosa * refPoints.first.x, targetPoints.first.y - cosa * refPoints.first.y - sina * refPoints.first.x);
}

void Aligner::Align(std::shared_ptr<AlignmentDataset> pRefDataset, std::shared_ptr<AlignmentDataset> pTargetDataset)
{
    Aligner aligner(pRefDataset, pTargetDataset);
    return aligner.Align();
}
