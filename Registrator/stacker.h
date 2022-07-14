#ifndef STACKER_H
#define STACKER_H
#include <memory>
#include <vector>
#include <string>

#include "../Core/bitmap.h"
#include "../Geometry/triangle.h"
#include "../AGG/agg_trans_affine.h"
#include "../Tests/test.h"
#include "registrator.h"
#include <array>
#include "FastAligner.h"

class ImageDecoder;

struct StackingDatum
{
    std::shared_ptr<ImageDecoder> pDecoder;
    std::vector<std::vector<Star>> stars;
    uint32_t totalStarCount;
};

class Stacker
{
    template <PixelFormat pixelFormat> friend class AddingBitmapHelper;
    template <PixelFormat pixelFormat> friend class AddingBitmapWithAlignmentHelper;
    template <PixelFormat pixelFormat> friend class GeneratingResultHelper;
    friend class AlignmentHelper;

    using TriangleTransformPair = std::pair<Triangle, agg::trans_affine>;
    using GridCell = std::vector<TriangleTransformPair>;
    using Grid = std::vector<GridCell>;

    Grid _grid;
    std::vector<StackingDatum> _stackingData;

    std::vector<float> _means;
    std::vector<float> _devs;
    std::vector<uint16_t> _counts;

    uint32_t _hTileCount = 0;
    uint32_t _vTileCount = 0;

    uint32_t _width = 0;
    uint32_t _height = 0;

    static const size_t gridSize = 100;
    uint32_t _gridWidth = 0;
    uint32_t _gridHeight = 0;

    double _alignmentError = 2.0;

    bool _enableDeaberration;

    std::vector<std::shared_ptr<FastAligner>> _aligners;
    MatchMap _matches;


    void ChooseTriangle(PointF p, std::pair<Triangle, agg::trans_affine>& lastPair, const GridCell& trianglePairs);
    void StackWithAlignment(IBitmapPtr pRefBitmap, IBitmapPtr pTargetBitmap, uint32_t i);    

public:

    Stacker(std::vector<std::shared_ptr<ImageDecoder>> decoders, bool enableDeaberration = false);

    void Registrate(uint32_t hTileCount = 1, uint32_t vTileCount = 1, double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25);
    std::shared_ptr<IBitmap>  RegistrateAndStack(uint32_t hTileCount = 1, uint32_t vTileCount = 1, double threshold = 40, uint32_t minStarSize = 5, uint32_t maxStarSize = 25);
    std::shared_ptr<IBitmap> Stack(bool doAlignment);   

    TEST_ACCESS(Stacker);

};

#endif // STACKER_H
