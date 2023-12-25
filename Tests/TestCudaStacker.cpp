#include "test.h"
#include "testtools.h"
#include "../Core/bitmap.h"
#include "../Core/pipeline.h"
#include "../Cuda/CudaStacker.h"
#include "../Registrator/stacker.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( CudaStacker )

BEGIN_TEST( TwoBitmaps )

std::vector<Pipeline> pipelines;
auto pRedBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>(10,10, MakeRGB24(255,0,0));
auto pBlueBitmap = std::make_shared<Bitmap<PixelFormat::RGB24>>(10,10, MakeRGB24(0,0,255));
pipelines.emplace_back( pRedBitmap );
pipelines.emplace_back( pBlueBitmap );

Stacker stacker( pipelines, StackMode::DarkOrFlat );
cuda::Stacker cudaStacker( pipelines, StackMode::DarkOrFlat );
EXPECT_TRUE( BitmapsAreEqual( stacker.Stack(), cudaStacker.Stack() ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
