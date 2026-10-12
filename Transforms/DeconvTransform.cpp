#include "DeconvTransform.h"
#include "converter.h"
#include "../Registrator/registrator.h"
#include "../Geometry/delaunator.hpp"
#include "../Geometry/triangle.h"
#include "../Transforms/HistogramBuilder.h"

#include <optional>
#include <algorithm>
#include <unordered_map>


ACMB_NAMESPACE_BEGIN

static constexpr double epsilon = 1e-6;

struct StarMoments
{
    double theta;
    double L;
    double sigma;

    auto operator<=>(const StarMoments& rhs) const = default;
};

struct StarMomentsHasher
{
    std::size_t operator()(const StarMoments& moments) const
    {
        std::size_t h1 = std::hash<double>{}(moments.theta);
        std::size_t h2 = std::hash<double>{}(moments.L);
        std::size_t h3 = std::hash<double>{}(moments.sigma);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

static std::optional<StarMoments> CalculateStarMoments(const Star::MomentMatrix& m)
{
    const double trace = m.xx + m.yy;
    const double d = sqrt((m.xx - m.yy) * (m.xx - m.yy) + 4 * m.xy * m.xy);
    const double lambdaMax = (trace + d) / 2.0;
    const double lambdaMin = (trace - d) / 2.0;

    if ( lambdaMin <= 0 || lambdaMax <= 0 )
        return std::nullopt;

    double L2 = 12.0 * (lambdaMax - lambdaMin);
    if ( L2 < 0.0 )
        return std::nullopt;    

    StarMoments res { .theta = 0.5 * atan2(2 * m.xy, m.xx - m.yy),
                      .L = sqrt(L2),
                      .sigma = sqrt(lambdaMin) };

    res.L = round(res.L * 100.0) / 100.0;
    res.sigma = round(res.sigma * 10.0) / 10.0;
    res.theta = round(res.theta * 10.0) / 10.0;

    return res;
}

int CalcKernelSize(double sigma, double L)
{
    const double k = 4.0;

    int radius = static_cast<int>(
        std::ceil(0.5 * L + k * sigma)
        );

    return 2 * radius + 1;
}

int calcMotionBlurKernelSize(double L)
{
    int radius =
        static_cast<int>(std::ceil(0.5 * L)) + 1;

    return 2 * radius + 1;
}

static std::vector<Star> FindStars(std::shared_ptr<IBitmap> pBitmap, double thresholdPercents, int minStarSize, int maxStarSize)
{
    Registrator registrator(thresholdPercents, minStarSize, maxStarSize);
    registrator.Registrate(pBitmap);
    std::vector<Star> stars;
    for (const auto& tileStars : registrator.GetStars())
    {
        stars.insert(stars.end(), tileStars.begin(), tileStars.end());
    }
    return stars;
}

static void FilterClippedStars(std::vector<Star>& stars, int width, int height)
{
    // Remove clipped stars
    std::erase_if(stars, [](const Star& star)
    {
        if ( star.isClipped )
            return true;

        return false;
    });

    // Remove stars that are too close to the image borders
    std::erase_if(stars, [width, height](const Star& star)
    {
        if ( star.rect.x <= 0 || star.rect.y <= 0 || star.rect.x + star.rect.width >= int(width) || star.rect.y + star.rect.height >= int(height) )
            return true;
        return false;
    });
}

static void FilterStarsWithInvalidMoments(std::vector<Star>& stars, std::unordered_map<PointD, std::pair<Star, StarMoments>, PointDHasher>& starMomentsMap)
{
    // Remove stars with invalid moments
    std::erase_if(stars, [&starMomentsMap](const Star& star)
    {
        auto moments = CalculateStarMoments(star.m);
        if ( moments.has_value() )
        {
            starMomentsMap[star.center.Rounded(epsilon)] = std::make_pair(star, moments.value());
            return false;
        }
        return true;
    });
}

static void FilterStarsTooFarFromMedian(std::vector<Star>& stars, std::unordered_map<PointD, std::pair<Star, StarMoments>, PointDHasher>& starMomentsMap)
{
    // find median L and sigma
    std::vector<double> LValues;
    std::vector<double> sigmaValues;
    for (const auto& star : stars)
    {
        const auto& moments = starMomentsMap[star.center.Rounded(epsilon)];
        LValues.push_back(moments.second.L);
        sigmaValues.push_back(moments.second.sigma);
    }
    std::sort(LValues.begin(), LValues.end());
    std::sort(sigmaValues.begin(), sigmaValues.end());
    const double medianL = LValues[LValues.size() / 2];
    const double medianSigma = sigmaValues[sigmaValues.size() / 2];
    // Remove stars that are too far from the median L and sigma
    std::erase_if(stars, [medianL, medianSigma, &starMomentsMap](const Star& star)
    {
        auto it = starMomentsMap.find(star.center.Rounded(epsilon));
        if ( it != starMomentsMap.end() && (std::abs(it->second.second.L - medianL) > 0.5 * medianL || std::abs(it->second.second.sigma - medianSigma) > 0.5 * medianSigma) )
        {
            starMomentsMap.erase(it);
            return true;
        }
        return false;
    });
}

static void TriangulateStars(const std::vector<Star>& stars, std::vector<Triangle>& triangles, std::vector<PointD>& hullPoints)
{
    std::vector<double> coords;
    for (const auto& star : stars)
    {
        coords.push_back(star.center.x);
        coords.push_back(star.center.y);
    }
    delaunator::Delaunator d(coords);
    size_t e = d.hull_start;
    do
    {
        hullPoints.push_back(PointD{d.coords[2 * e], d.coords[2 * e + 1]});
        e = d.hull_next[e];
    } while ( e != d.hull_start );
    triangles.reserve(d.triangles.size() / 3);
    for (int i = 0; i < int(d.triangles.size()); i += 3)
    {
        triangles.push_back(Triangle{
            PointD{d.coords[2 * d.triangles[i]], d.coords[2 * d.triangles[i] + 1]},
            PointD{d.coords[2 * d.triangles[i + 1]], d.coords[2 * d.triangles[i + 1] + 1]},
            PointD{d.coords[2 * d.triangles[i + 2]], d.coords[2 * d.triangles[i + 2] + 1]}
        });
    }
}

static void RasterizeTriangles(const std::vector<Triangle>& triangles, int width, int height, std::vector<int>& triangleIndicesMap)
{
    triangleIndicesMap.resize(width * height, -1);
    for (int i = 0; i < int(triangles.size()); ++i)
    {
        const Triangle& tri = triangles[i];
        const auto bbox = tri.GetBoundingBox();
        for (int y = std::max(0, int(bbox.y)); y < std::min(int(height), int(bbox.y + bbox.height)); ++y)
        {
            for (int x = std::max(0, int(bbox.x)); x < std::min(int(width), int(bbox.x + bbox.width)); ++x)
            {
                if (tri.IsPointInsideUnchecked(PointD{double(x + 0.5), double(y + 0.5)}))
                {
                    triangleIndicesMap[y * width + x] = i;
                }
            }
        }
    }
}

struct ProjectionToHullResult
{
    PointD p1;
    PointD p2;
    double param;
};

static ProjectionToHullResult ProjectPointToHull(const PointD& point, const std::vector<PointD>& hullPoints)
{
    double nearestEdgeDistSq = std::numeric_limits<double>::max();
    double nearestParam = 0.0;
    PointD nearestP1;
    PointD nearestP2;
    for (int i = 0; i < int(hullPoints.size()); ++i)
    {
        const PointD& p1 = hullPoints[i];
        const PointD& p2 = hullPoints[(i + 1) % hullPoints.size()];
        auto projectionResult = PointD::ProjectPointOntoSegment(point, p1, p2);
        if (projectionResult.distanceSq < nearestEdgeDistSq)
        {
            nearestEdgeDistSq = projectionResult.distanceSq;
            nearestParam = projectionResult.param;
            nearestP1 = p1;
            nearestP2 = p2;
        }
    }
    return {nearestP1, nearestP2, nearestParam};
}

static Star::MomentMatrix InterpolateMoments(const Star::MomentMatrix& m1, const Star::MomentMatrix& m2, double param)
{
    Star::MomentMatrix m;
    m.xx = (1 - param) * m1.xx + param * m2.xx;
    m.yy = (1 - param) * m1.yy + param * m2.yy;
    m.xy = (1 - param) * m1.xy + param * m2.xy;
    return m;
}

static Star::MomentMatrix InterpolateMoments(const Star::MomentMatrix& m1, const Star::MomentMatrix& m2, const Star::MomentMatrix& m3,
                                          std::array<double, 3> baryCoords)
{
    return Star::MomentMatrix
    {
        .xx = baryCoords[0] * m1.xx + baryCoords[1] * m2.xx + baryCoords[2] * m3.xx,
        .yy = baryCoords[0] * m1.yy + baryCoords[1] * m2.yy + baryCoords[2] * m3.yy,
        .xy = baryCoords[0] * m1.xy + baryCoords[1] * m2.xy + baryCoords[2] * m3.xy
    };
}

struct PSF
{
    int size = 0;
    std::vector<double> data;

    double& operator()(int y, int x)
    {
        return data[y * size + x];
    }

    double operator()(int y, int x) const
    {
        return data[y * size + x];
    }

    std::shared_ptr<Bitmap<PixelFormat::Gray16>> GetAsBitmap(const std::string& filename) const
    {
        std::vector<uint16_t> bitmapData(size * size);
        double maxValue = *std::max_element(data.begin(), data.end());
        for (int i = 0; i < size * size; ++i)
        {
            bitmapData[i] = static_cast<uint16_t>(std::round((data[i] / maxValue) * 65535.0));
        }
        auto pBitmap = IBitmap::Create(size, size, PixelFormat::Gray16);
        for (int y = 0; y < size; ++y)
        {
            memcpy(pBitmap->GetPlanarScanline(y), &bitmapData[y * size], size * sizeof(uint16_t));
        }

        return std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pBitmap);
    }
};

PSF MakePSF(int size, double L, double sigma, double theta)
{
    constexpr double PI = 3.14159265358979323846;
    constexpr double SQRT2 = 1.4142135623730950488;

    sigma = std::max(sigma, 1e-6);
    L = std::max(L, 0.0);

    PSF psf;
    psf.size = size;
    psf.data.assign(size * size, 0.0);

    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    double sum = 0.0;
    const int halfSize = size / 2;

    for ( int ky = 0; ky < size; ++ky )
    {
        for ( int kx = 0; kx < size; ++kx )
        {
            const double x = kx - halfSize;
            const double y = ky - halfSize;

            const double u = x * ct + y * st;
            const double v = -x * st + y * ct;

            double value;

            if ( L < 1e-6 )
            {
                const double r2 = x * x + y * y;

                value =
                    std::exp(-r2 / (2.0 * sigma * sigma)) /
                    (2.0 * PI * sigma * sigma);
            }
            else
            {
                const double a =
                    (u + 0.5 * L) /
                    (SQRT2 * sigma);

                const double b =
                    (u - 0.5 * L) /
                    (SQRT2 * sigma);

                const double longitudinal =
                    std::erf(a) - std::erf(b);

                const double transverse =
                    std::exp(
                        -(v * v) /
                        (2.0 * sigma * sigma)
                    );

                value =
                    transverse * longitudinal /
                    (
                        2.0 *
                        L *
                        std::sqrt(2.0 * PI) *
                        sigma
                        );
            }

            psf(ky, kx) = value;
            sum += value;
        }
    }

    if ( sum > 0.0 )
    {
        for ( double& v : psf.data )
            v /= sum;
    }

    return psf;
}

PSF MakeMotionPSF(int size, double L, double theta)
{
    L = std::max(L, 0.0);

    PSF psf;
    psf.size = size;
    psf.data.assign(size * size, 0.0);

    const int halfSize = size / 2;

    // Если смаза почти нет — delta kernel.
    if ( L < 1e-6 )
    {
        psf(halfSize, halfSize) = 1.0;
        return psf;
    }

    const double ct = std::cos(theta);
    const double st = std::sin(theta);

    // Количество сэмплов вдоль линии.
    // 8 сэмплов на пиксель длины обычно хватает.
    const int samples = std::max(
        16,
        static_cast<int>(std::ceil(L * 8.0))
    );

    for ( int i = 0; i < samples; ++i )
    {

        const double t =
            L * (
                (static_cast<double>(i) + 0.5) /
                static_cast<double>(samples)
                - 0.5
                );

        // Координаты sample в системе kernel
        const double x =
            static_cast<double>(halfSize) + t * ct;

        const double y =
            static_cast<double>(halfSize) + t * st;

        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));

        const double fx = x - x0;
        const double fy = y - y0;

        auto Add = [&](int px, int py, double w)
        {
            if ( px >= 0 && px < size &&
                py >= 0 && py < size )
            {
                psf(py, px) += w;
            }
        };

        // Bilinear splatting
        Add(x0, y0, (1.0 - fx) * (1.0 - fy));
        Add(x0 + 1, y0, fx * (1.0 - fy));
        Add(x0, y0 + 1, (1.0 - fx) * fy);
        Add(x0 + 1, y0 + 1, fx * fy);
    }

    // Нормализация
    double sum = 0.0;
    for ( double v : psf.data )
        sum += v;

    if ( sum > 0.0 )
    {
        for ( double& v : psf.data )
            v /= sum;
    }

    return psf;
}

using PositionToStarAndMomentsMap = std::unordered_map <PointD, std::pair<Star, StarMoments>, PointDHasher>;

class PSFCache
{
    std::unordered_map<StarMoments, int, StarMomentsHasher> _psfIdxCache;
    std::vector<PSF> _psfCache;
    std::vector<int> _psfIndices;
    
public:
    PSFCache(int width, int height)
        : _psfIndices(width * height, -1)
    {
    }

    PSF GetPSF(int px) const
    {
        int psfIdx = _psfIndices[px];
        if ( psfIdx == -1 )
            return PSF{};

        return _psfCache[psfIdx];
    }

    void Clear()
    {
        _psfIdxCache.clear();
        _psfCache.clear();
        std::fill(_psfIndices.begin(), _psfIndices.end(), -1);
    }

    void GenerateFromOneStar(const Star& star, const StarMoments& moments, int width, int height)
    {
        Clear();
        PSF psf = MakePSF(CalcKernelSize(moments.sigma, moments.L), moments.L, moments.sigma, moments.theta);

        /*Star::MomentMatrix psfMatrix;
        double sum = 0.0;
        for (int y = 0; y < psf.size; ++y)
        {
            for (int x = 0; x < psf.size; ++x)
            {
                double value = psf(y, x);
                double dx = x - psf.size / 2;
                double dy = y - psf.size / 2;
                psfMatrix.xx += value * dx * dx;
                psfMatrix.yy += value * dy * dy;
                psfMatrix.xy += value * dx * dy;

                sum += value;
            }
        }

        psfMatrix.xx /= sum;
        psfMatrix.yy /= sum;
        psfMatrix.xy /= sum;

        auto psfMomentsOpt = CalculateStarMoments(psfMatrix);

        auto psfBitmap = psf.GetAsBitmap("");
        psfBitmap->Save(psfBitmap, "F:\\Projects\\AstroCombine\\Tests\\TestFiles\\TIFF\\psf.tif");*/

        _psfCache.push_back(std::move(psf));
        _psfIdxCache.insert({ moments, 0 });
        _psfIndices.assign(width * height, 0);
    }

    void Generate(const std::vector<Triangle>& triangles, const std::vector<PointD>& hullPoints, const std::vector<int>& triangleIndices, const PositionToStarAndMomentsMap& starMomentsMap, int width, int height)
    {
        Clear();

        for ( int y = 0; y < int(height); ++y )
        {
            for ( int x = 0; x < int(width); ++x )
            {
                Star::MomentMatrix m;

                int triIndex = triangleIndices[y * width + x];
                if ( triIndex == -1 )
                {
                    ProjectionToHullResult projection = ProjectPointToHull(PointD{ x + 0.5, y + 0.5 }, hullPoints);

                    // Interpolate moments from the two nearest hull points
                    auto it1 = starMomentsMap.find(projection.p1.Rounded(epsilon));
                    auto it2 = starMomentsMap.find(projection.p2.Rounded(epsilon));

                    if ( it1 != starMomentsMap.end() && it2 != starMomentsMap.end() )
                    {
                        m = InterpolateMoments(it1->second.first.m, it2->second.first.m, projection.param);
                    }
                }
                else
                {
                    const Triangle& tri = triangles[triIndex];
                    auto baryCoordsOpt = tri.GetBarycentricCoords(PointD{ x + 0.5, y + 0.5 });
                    if ( !baryCoordsOpt.has_value() )
                        continue;

                    auto it1 = starMomentsMap.find(tri.vertices[0].Rounded(epsilon));
                    auto it2 = starMomentsMap.find(tri.vertices[1].Rounded(epsilon));
                    auto it3 = starMomentsMap.find(tri.vertices[2].Rounded(epsilon));

                    if ( it1 == starMomentsMap.end() || it2 == starMomentsMap.end() || it3 == starMomentsMap.end() )
                        continue;

                    m = InterpolateMoments(
                        it1->second.first.m,
                        it2->second.first.m,
                        it3->second.first.m,
                        baryCoordsOpt.value()
                    );
                }

                std::optional<StarMoments> momentsOpt = CalculateStarMoments(m);
                if ( !momentsOpt.has_value() )
                    continue;

                const StarMoments& moments = momentsOpt.value();
                auto psfIt = _psfIdxCache.find(moments);
                int psfIdx = -1;

                if ( psfIt == _psfIdxCache.end() )
                {
                    int kernelSize = CalcKernelSize(moments.sigma, moments.L);
                    PSF psf = MakePSF(kernelSize, moments.L, moments.sigma, moments.theta);

                    psfIdx = static_cast<int>(_psfCache.size());
                    _psfCache.push_back(std::move(psf));
                    _psfIdxCache.insert({ moments, psfIdx });
                }
                else
                {
                    psfIdx = psfIt->second;
                }

                _psfIndices[y * width + x] = psfIdx;
            }
        }
    }
};

template<PixelFormat pixelFormat>
class DeconvTransformImpl : public DeconvTransform
{
    using ChannelType = typename PixelFormatTraits<pixelFormat>::ChannelType;
    static constexpr auto channelCount = PixelFormatTraits<pixelFormat>::channelCount;

    using FPImage = std::vector<double>;

    static FPImage ConvertBitmapToFPImage(std::shared_ptr<Bitmap<pixelFormat>> pBitmap, const std::array<double, channelCount>& bgValues)
    {
        const auto width = pBitmap->GetWidth();
        const auto height = pBitmap->GetHeight();
        FPImage image(width * height * channelCount);
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                for (uint32_t c = 0; c < channelCount; ++c)
                {
                    image[(y * width + x) * channelCount + c] = std::max(0.0, double(pBitmap->GetChannel(x, y, c)) - bgValues[c]);
                }
            }
        }
        return image;
    }

    static std::shared_ptr<Bitmap<pixelFormat>> ConvertFPImageToBitmap(const FPImage& image, uint32_t width, uint32_t height, const std::array<double, channelCount>& bgValues)
    {
        auto pBitmap = std::make_shared<Bitmap<pixelFormat>>(width, height);
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                for (uint32_t c = 0; c < channelCount; ++c)
                {
                    double value = image[(y * width + x) * channelCount + c] + bgValues[c] + 0.5;
                    value = std::clamp(value, 0.0, static_cast<double>(std::numeric_limits<ChannelType>::max()));
                    pBitmap->SetChannel(x, y, c, static_cast<ChannelType>(value));
                }
            }
        }
        return pBitmap;
    }

    static void ApplyH(
        const FPImage& input,
        FPImage& output,
        int width,
        int height,
        const PSFCache& cache)
    {
        output.assign(width * height * channelCount, 0.0);
        Rect bounds(0, 0, width - 1, height - 1);

        for ( int y = 0; y < height; ++y )
        {
            for ( int x = 0; x < width; ++x )
            {
                PSF psf = cache.GetPSF(y * width + x);

                const int r = psf.size / 2;

                for ( int py = 0; py < psf.size; ++py )
                {
                    for ( int px = 0; px < psf.size; ++px )
                    {
                        int xx = x + px - r;
                        int yy = y + py - r;

                        if ( !bounds.IsPointInside(Point(xx, yy)) )
                            continue;

                        for ( uint32_t c = 0; c < channelCount; ++c )
                        {
                            output[(yy * width + xx) * channelCount + c] +=
                                input[(y * width + x) * channelCount + c] * psf(py, px);
                        }
                    }
                }
            }
        }
    }

    static void ApplyHT(
        const FPImage& input,
        FPImage& output,
        int width,
        int height,
        const PSFCache& cache)
    {
        output.assign(width * height * channelCount, 0.0);
        Rect bounds(0, 0, width - 1, height-1);

        for ( int y = 0; y < height; ++y )
        {
            for ( int x = 0; x < width; ++x )
            {
                PSF psf = cache.GetPSF(y * width + x);
                const int r = psf.size / 2;

                std::array<double, channelCount> sum = {};

                for ( int py = 0; py < psf.size; ++py )
                {
                    for ( int px = 0; px < psf.size; ++px )
                    {
                        int xx = x + px - r;
                        int yy = y + py - r;

                        if ( !bounds.IsPointInside(Point(xx, yy)) )
                            continue;

                        for ( uint32_t c = 0; c < channelCount; ++c )
                        { 
                            sum[c] +=
                                input[(yy * width + xx) * channelCount + c] *
                                psf(py, px);
                        }
                    }
                }

                for ( uint32_t c = 0; c < channelCount; ++c )
                {
                    output[(y * width + x) * channelCount + c] = sum[c];
                }
            }
        }
    }

    static std::shared_ptr<Bitmap<pixelFormat>> LucyRichardson(std::shared_ptr<Bitmap<pixelFormat>> pObserved, const PSFCache& cache, int iterations)
    {
        const int width = int(pObserved->GetWidth());
        const int height = int(pObserved->GetHeight());

        auto pHistogramBuilder = HistogramBuilder::Create(pObserved);
        pHistogramBuilder->BuildHistogram();

        std::array<double, channelCount> bgValues = {};
        for ( uint32_t c = 0; c < channelCount; ++c )
        {
            const auto& stats = pHistogramBuilder->GetChannelStatistics(c);
            const double median = stats.median;
            bgValues[c] = median;
        }

        FPImage observed = ConvertBitmapToFPImage(pObserved, bgValues);
        FPImage estimate = observed;
        FPImage predicted;
        FPImage correction;
        FPImage ratio(width * height * channelCount, 0.0);
        

        FPImage ones(width * height * channelCount, 1.0); 
        FPImage normalization;
        ApplyHT(ones, normalization, width, height, cache);

        for ( int iteration = 0; iteration < iterations; ++iteration )
        {
            ApplyH(estimate, predicted, width, height, cache);

            for ( int y = 0; y < height; ++y )
            {
                for ( int x = 0; x < width; ++x )
                {
                    int pixelIndex = (y * width + x) * channelCount;
                    for ( uint32_t c = 0; c < channelCount; ++c )
                    {
                        ratio[pixelIndex + c] = observed[pixelIndex + c] / (predicted[pixelIndex + c] + epsilon);
                    }
                }
            }

            ApplyHT(ratio, correction, width, height, cache);

            for ( int y = 0; y < height; ++y )
            {
                for ( int x = 0; x < width; ++x )
                {
                    int pixelIndex = (y * width + x) * channelCount;

                    for ( uint32_t c = 0; c < channelCount; ++c )
                    {
                        estimate[pixelIndex + c] *=
                            correction[pixelIndex + c] /
                            (normalization[pixelIndex + c] + epsilon);

                        estimate[pixelIndex + c] = std::max(0.0, estimate[pixelIndex + c]);
                    }
                }
            }
        }

        const double bg = channelCount == 1 ? bgValues[0] : (bgValues[0] + bgValues[1] + bgValues[2]) / 3.0;

        // stretch the estimate to fit into the channel range
        const double actualMax = bg + *std::max_element(estimate.begin(), estimate.end());
        constexpr double desiredMax = 0.9 * std::numeric_limits<ChannelType>::max();
        const double scale = desiredMax / actualMax;

        if ( actualMax > 0.0 )
        {
            for ( double& v : estimate )
                v *= scale;

            for ( auto& bgValue : bgValues )
                bgValue *= scale;
        }

        auto res =  ConvertFPImageToBitmap(estimate, width, height, bgValues);

        for ( uint32_t i = 0; i < height; ++i )
        {
            ChannelType* pSrcScanline = pObserved->GetScanline(i);
            ChannelType* pDstScanline = res->GetScanline(i);
            for ( uint32_t j = 0; j < width; ++j )
            {
                for ( uint32_t c = 0; c < channelCount; ++c )
                {
                    if ( estimate[(i * width + j) * channelCount + c] < epsilon )
                        pDstScanline[j * channelCount + c] = ChannelType(std::clamp(pSrcScanline[j * channelCount + c] * scale + 0.5, 0.0, double(std::numeric_limits<ChannelType>::max())));
                }
            }
        }

        return res;
    }

public:
    DeconvTransformImpl(std::shared_ptr<Bitmap<pixelFormat>> pSrcBitmap, const Settings& settings)
        : DeconvTransform(pSrcBitmap, settings)
    {
    }

    virtual void Run() override
    {
        const auto width = _pSrcBitmap->GetWidth();
        const auto height = _pSrcBitmap->GetHeight();

        auto pSrcBitmap = std::static_pointer_cast<Bitmap<pixelFormat>>(_pSrcBitmap);
        auto pDstBitmap = pSrcBitmap->Clone();

        constexpr PixelFormat grayFormat = ConstructPixelFormat(PixelFormatTraits<pixelFormat>::bitsPerChannel, 1);
        auto pGraySrcBitmap = std::static_pointer_cast<Bitmap<grayFormat>>(Converter::Convert(pSrcBitmap, grayFormat));
        auto pGrayDstBitmap = std::static_pointer_cast<Bitmap<grayFormat>>(Converter::Convert(pDstBitmap, grayFormat));

        std::vector<Star> stars = FindStars(pGraySrcBitmap, _settings._thresholdPercents, _settings._minStarSize, _settings._maxStarSize);
        FilterClippedStars(stars, width, height);


        PositionToStarAndMomentsMap starMomentsMap;
        FilterStarsWithInvalidMoments(stars, starMomentsMap);

        FilterStarsTooFarFromMedian(stars, starMomentsMap);

        if ( stars.empty() )
            return;

        PSFCache psfCache(width, height);

        if ( stars.size() < 3 )
        {
            // Not enough stars to triangulate, use the first star's moments for the whole image
            const auto& moments = starMomentsMap[stars[0].center.Rounded(epsilon)].second;
            psfCache.GenerateFromOneStar(stars[0], moments, width, height);
        }
        else
        {
            std::vector<Triangle> triangles;
            std::vector<PointD> hullPoints;
            TriangulateStars(stars, triangles, hullPoints);

            std::vector<int> triangleIndices;
            RasterizeTriangles(triangles, width, height, triangleIndices);

            psfCache.Generate(triangles, hullPoints, triangleIndices, starMomentsMap, width, height);

        }

        _pDstBitmap = LucyRichardson(pSrcBitmap, psfCache, _settings._iterations);
    }

    virtual void ValidateSettings() override
    {
    }
};


DeconvTransform::DeconvTransform(IBitmapPtr pSrcBitmap, const Settings& settings)
    : BaseTransform(pSrcBitmap)
    , _settings(settings)
{
}

std::shared_ptr<DeconvTransform> DeconvTransform::Create(IBitmapPtr pSrcBitmap, const Settings& settings)
{
    switch ( pSrcBitmap->GetPixelFormat() )
    {
        case PixelFormat::RGB24:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB24>>(std::static_pointer_cast<Bitmap<PixelFormat::RGB24>>(pSrcBitmap), settings);
        case PixelFormat::RGB48:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB48>>(std::static_pointer_cast<Bitmap<PixelFormat::RGB48>>(pSrcBitmap), settings);
        case PixelFormat::Gray8:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray8>>(std::static_pointer_cast<Bitmap<PixelFormat::Gray8>>(pSrcBitmap), settings);
        case PixelFormat::Gray16:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray16>>(std::static_pointer_cast<Bitmap<PixelFormat::Gray16>>(pSrcBitmap), settings);

        default:
            throw std::invalid_argument("DeconvTransform: Unsupported pixel format");
    }
}

std::shared_ptr<DeconvTransform> DeconvTransform::Create(PixelFormat pixelFormat, const Settings& settings)
{
    switch ( pixelFormat )
    {
        case PixelFormat::RGB24:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB24>>(nullptr, settings);
        case PixelFormat::RGB48:
            return std::make_shared<DeconvTransformImpl<PixelFormat::RGB48>>(nullptr, settings);
        case PixelFormat::Gray8:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray8>>(nullptr, settings);
        case PixelFormat::Gray16:
            return std::make_shared<DeconvTransformImpl<PixelFormat::Gray16>>(nullptr, settings);

        default:
            throw std::invalid_argument("DeconvTransform: Unsupported pixel format");
    }
}

IBitmapPtr DeconvTransform::ApplyTransform(IBitmapPtr pSrcBitmap, const Settings& settings)
{
    auto pDeconvTransform = Create(pSrcBitmap, settings);
    return pDeconvTransform->RunAndGetBitmap();
}

DeconvTransform::Settings DeconvTransform::Interpolate(const Settings& a, const Settings& b, double t)
{
    Settings result;
    result._intensity = a._intensity + (b._intensity - a._intensity) * t;
    return result;
}

ACMB_NAMESPACE_END