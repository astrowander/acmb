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
pipelines.emplace_back( std::make_shared<Bitmap<PixelFormat::RGB24>>( 10, 10, MakeRGB24( 254, 0, 0 ) ) );
pipelines.emplace_back( std::make_shared<Bitmap<PixelFormat::RGB24>>( 10, 10, MakeRGB24( 0, 0, 254 ) ) );

Stacker stacker( pipelines, StackMode::DarkOrFlat );
cuda::Stacker cudaStacker( pipelines, StackMode::DarkOrFlat );
EXPECT_TRUE( BitmapsAreEqual( stacker.Stack(), cudaStacker.Stack() ) );

END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
