#include "aligner.h"
#include "registrator.h"

Aligner::Aligner(std::vector<std::shared_ptr<IBitmap> > bitmaps)
    :_bitmaps(bitmaps)
{}

std::vector<std::shared_ptr<AlignmentDataset>> Aligner::Align()
{
    std::vector<std::shared_ptr<AlignmentDataset>> datasets;
    for (auto& bitmap : _bitmaps)
    {
        datasets.push_back(Registrator::Registrate(bitmap, 50, 5, 25));
        auto & dataset = datasets.back();
        if (dataset->stars.size() < dataset->valuableStarCount)
        {
            dataset->valuableStarCount = dataset->stars.size();
        }
        else
        {
            const Star upperVal {Rect {}, PointF{}, 0.5, 0};
            auto brightCount = std::upper_bound(dataset->stars.begin(), dataset->stars.end(), upperVal, [](const Star& a, const Star& b) { return a.luminance > b.luminance; }) - std::begin(dataset->stars);
            if (brightCount > dataset->valuableStarCount)
                dataset->valuableStarCount = brightCount;
        }
    }

    if (datasets.size() < 2)
        return datasets;

    for (uint32_t i = 1; i < datasets.size(); ++i)
    {
        ProcessPairOfDatasets(datasets[0], datasets[i]);
    }

    return datasets;
}

void Aligner::ProcessPairOfDatasets(std::shared_ptr<AlignmentDataset> ref, std::shared_ptr<AlignmentDataset> target)
{
    std::pair<Star, Star> refPair { ref->stars[0], ref->stars[1] };
    std::pair<PointF, PointF> refPoints{refPair.first.center, refPair.second.center};
    auto refDist = refPoints.first.Distance(refPoints.second);

    std::pair<Star, Star> targetPair { Star{}, Star{} };
    std::pair<PointF, PointF> targetPoints{ PointF{}, PointF{} };
    for (uint32_t i = 0; i < target->valuableStarCount - 1; ++i)
    {
        targetPair.first = target->stars[i];
        targetPoints.first = targetPair.first.center;

        for (uint32_t j = i + 1; j < target->valuableStarCount; ++j)
        {
            targetPair.second = target->stars[j];
            targetPoints.second = targetPair.second.center;
            auto targetDist = targetPoints.first.Distance(targetPoints.second);

            if (fabs(targetDist - refDist) > _maxStarSize / 2)
                continue;

            target->transform = CalculateTransform(refPoints, targetPoints);

            if (CheckPairOfDatasets(ref, target))
                return;

            std::swap(targetPoints.first, targetPoints.second);

            if (CheckPairOfDatasets(ref, target))
                return;
        }
    }
}

bool Aligner::CheckPairOfDatasets(std::shared_ptr<AlignmentDataset> ref, std::shared_ptr<AlignmentDataset> target)
{
    uint32_t matches = 0;

    for (uint32_t i = 0; i < ref->valuableStarCount; ++i)
    {
        auto refPoint = ref->stars[i].center;
        target->transform.transform(&refPoint.x, &refPoint.y);

        for (uint32_t j = 0; j < target->valuableStarCount; ++j)
        {
            auto targetPoint = target->stars[j].center;
            auto dist = refPoint.Distance(targetPoint);
            if (dist < _maxStarSize / 2)
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

std::vector<std::shared_ptr<AlignmentDataset> > Aligner::Align(std::vector<std::shared_ptr<IBitmap> > bitmaps)
{
    Aligner aligner(bitmaps);
    return aligner.Align();
}
