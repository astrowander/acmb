#ifndef MATHTOOLS_H
#define MATHTOOLS_H

float QuadraticInterpolation(float t, float t0, float t1, float t2);

template<typename T>
T FastRound(double x)
{
    return static_cast<T>(x > 0 ? x + 0.5 : x - 0.5);
}

#endif
