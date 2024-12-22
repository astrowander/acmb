#include "test.h"
#include "testtools.h"
#include "../Core/bitmap.h"
#include "../Core/color.h"
#include "../Core/pipeline.h"
#include "../Cuda/CudaStacker.h"
#include "../Registrator/stacker.h"

ACMB_TESTS_NAMESPACE_BEGIN

BEGIN_SUITE( CudaStacker )

BEGIN_TEST( TwoBitmaps )

std::vector<Pipeline> pipelines;
pipelines.emplace_back( std::make_shared<Bitmap<PixelFormat::RGB24>>( 10, 10, IColor::MakeRGB24( 254, 0, 0 ) ) );
pipelines.emplace_back( std::make_shared<Bitmap<PixelFormat::RGB24>>( 10, 10, IColor::MakeRGB24( 0, 0, 254 ) ) );

const auto & params = *pipelines[0].GetFinalParams();
Stacker stacker( params, StackMode::DarkOrFlat);
stacker.AddBitmaps( pipelines );
cuda::Stacker cudaStacker( params, StackMode::DarkOrFlat );
cudaStacker.AddBitmaps( pipelines );

EXPECT_TRUE( BitmapsAreEqual( stacker.GetResult(), cudaStacker.GetResult()));

END_TEST

BEGIN_TEST( StarTrails )
std::vector<Pipeline> pipelines;
pipelines.emplace_back( IBitmap::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8944.CR2" ) ) );
pipelines.emplace_back( IBitmap::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8945.CR2" ) ) );
pipelines.emplace_back( IBitmap::Create( GetPathToTestFile( "RAW/MilkyWayCR2/IMG_8946.CR2" ) ) );

const auto& params = *pipelines[0].GetFinalParams();
Stacker stacker( params, StackMode::StarTrails );
stacker.AddBitmaps( pipelines );
cuda::Stacker cudaStacker( params, StackMode::StarTrails );
cudaStacker.AddBitmaps( pipelines );

EXPECT_TRUE( BitmapsAreEqual( stacker.GetResult(), cudaStacker.GetResult()));
END_TEST

END_SUITE

ACMB_TESTS_NAMESPACE_END
