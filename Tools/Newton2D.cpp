#include "Newton2D.h"

Newton2D::Newton2D(const std::array<Func2D, 2>& f)
: _f(f)
, _jacobianValues()
{

}

Newton2D::Newton2D(const std::array<Func2D, 2>& f, const Jacobian& jacobian)
: _f(f)
, _jacobian(jacobian)
, _jacobianValues()
{

}

Vector2 Newton2D::Solve(Vector2 x)
{
	Vector2 v = {};
	Vector2 dx = { std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };

	int iterations = 0;

	while ((iterations++ < _maxIterations) && (dx[0] > _eps || dx[1] > _eps))
	{
		v[0] = _f[0](x);
		v[1] = _f[1](x);

		//write components of the jacobian like it is already inverted
		if (_jacobian)
		{
			_jacobianValues[1 * 2 + 1] = (*_jacobian)[0 * 2 + 0](x);
			_jacobianValues[0 * 2 + 1] = -(*_jacobian)[0 * 2 + 1](x);
			_jacobianValues[1 * 2 + 0] = -(*_jacobian)[1 * 2 + 0](x);
			_jacobianValues[0 * 2 + 0] = (*_jacobian)[1 * 2 + 1](x);
		}
		else
		{
			_jacobianValues[1 * 2 + 1] = (_f[0]({ x[0] + _step, x[1] }) - v[0]) / _step;
			_jacobianValues[0 * 2 + 1] = -(_f[0]({ x[0], x[1] + _step }) - v[0]) / _step;
			_jacobianValues[1 * 2 + 0] = -(_f[1]({ x[0] + _step, x[1] }) - v[1]) / _step;
			_jacobianValues[0 * 2 + 0] = (_f[1]({ x[0], x[1] + _step }) - v[1]) / _step;
		}

		double mult = (_jacobianValues[0 * 2 + 0] * _jacobianValues[1 * 2 + 1] - _jacobianValues[0 * 2 + 1] * _jacobianValues[1 * 2 + 0]);
		if (mult == 0.0)
			throw std::runtime_error("unable to solve the system of the equations");

		dx[0] = (_jacobianValues[0 * 2 + 0] * v[0] + _jacobianValues[0 * 2 + 1] * v[1]) / mult;
		dx[1] = (_jacobianValues[1 * 2 + 0] * v[0] + _jacobianValues[1 * 2 + 1] * v[1]) / mult;

		x[0] -= dx[0];
		x[1] -= dx[1];
	}

	return x;
}

std::array<double, 2> Newton2D::Solve(const std::array<Func2D, 2>& f, std::array<double, 2> x)
{
	Newton2D solver(f);
	return solver.Solve(x);
}

std::array<double, 2> Newton2D::Solve(const std::array<Func2D, 2>& f, const Jacobian& jacobian, std::array<double, 2> x)
{
	Newton2D solver(f, jacobian);
	return solver.Solve(x);
}
