#include "test.h"
#include "testtools.h"
#include "../Geometry/startrektransform.h"
#include "../Core/camerasettings.h"
#include <fstream>

BEGIN_SUITE(StarTrekTransform)

BEGIN_TEST(StarTrekTransform, CalculateOffsetField)

const PointF centralPoint { 2736.0, 1824.0 };
const double radiansPerPixel = 2 * atan(12.0 / 24.0) / (2 * centralPoint.y);
const agg::trans_affine affineMatrix = agg::trans_affine_translation(-centralPoint.x, -centralPoint.y) * agg::trans_affine_scaling(radiansPerPixel, radiansPerPixel);

for (double decl0 = 0; decl0 < 80.0; decl0 += 5.0)
{
	const double timeSpan = 600;

	auto pStarTrekTransform = std::make_shared<StarTrekTransform>(affineMatrix, decl0 * 3.1416 / 180, timeSpan);

	std::ofstream output(std::string("./Scripts/calculated") + std::to_string(decl0) + std::string(".csv"));
	const double step = 50;

	PointF src{ centralPoint.x, centralPoint.y };
	PointF dst = pStarTrekTransform->Transform(src);
	Vector2 baseTranslate{ dst.x - src.x, dst.y - src.y };


	for (double dx = -500; dx < 500 + step / 2; dx += step)
		for (double dy = -500; dy < 500 + step / 2; dy += step)
		{
			PointF src{ centralPoint.x + dx, centralPoint.y + dy };
			PointF dst = pStarTrekTransform->Transform(src);

			output << dx << ";" << dy << ";" << dst.x - src.x - baseTranslate[0] << ";" << dst.y - src.y - baseTranslate[1] << std::endl;
		}
}
/*double dx = 500;
double dy = -1;

src = { centralPoint.x + dx, centralPoint.y + dy };
dst = pStarTrekTransform->Transform(src);

output << dx << ";" << dy << ";" << dst.x - src.x << ";" << (dst.y - src.y) << std::endl;

dy = 0;

src = { centralPoint.x + dx, centralPoint.y + dy };
dst = pStarTrekTransform->Transform(src);

output << dx << ";" << dy << ";" << dst.x - src.x<< ";" << (dst.y - src.y) << std::endl;*/
END_TEST

END_SUITE