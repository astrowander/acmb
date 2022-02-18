#ifndef NEWTON2D_H
#define NEWTON2D

#include "math.h"
#include <functional>
#include <array>
#include <stdexcept>
#include <limits>
#include <optional>

using Vector2 = std::array<double, 2>;
using Func2D = std::function<double(Vector2)>;
using FuncVector2 = std::array<Func2D, 2>;
using Jacobian = std::array< Func2D, 2 * 2>;
using JacobianValues = std::array< double, 2 * 2>;

class Newton2D
{
	FuncVector2 _f;
	
	//if the jacobian is not initialized, we will compute it numerically
	std::optional<Jacobian> _jacobian;
	JacobianValues _jacobianValues;

	const double _eps = 1e-6;
	const double _step = 1e-9;
	const int _maxIterations = 10;

	Newton2D(const std::array<Func2D, 2>& f);
	Newton2D(const std::array<Func2D, 2>& f, const Jacobian& jacobian);

	Vector2 Solve(Vector2 x);

public:

	static Vector2 Solve(const std::array<Func2D, 2>& f, Vector2 x);
	static Vector2 Solve(const std::array<Func2D, 2>& f, const Jacobian& jacobian, Vector2 x);
};
#endif
