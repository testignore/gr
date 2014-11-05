
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include "FunctionMinimization.h"
#include "FunctionMinimizationProblem.h"

#define BOOST_TEST_MODULE(FunctionMinimizationTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

class Rosenbrock2DProblem : public Grante::FunctionMinimizationProblem {
public:
	virtual double Eval(const std::vector<double>& x,
		std::vector<double>& grad) {
		assert(x.size() == 2);
		assert(grad.size() == 2);

		double fobj = (1.0 - x[0])*(1.0 - x[0]) +
			100.0*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);

		grad[0] = 2.0*x[0] - 2.0 - 400.0*(x[0]*x[1]) + 400.0*x[0]*x[0]*x[0];
		grad[1] = 200.0*(x[1] - x[0]*x[0]);

		return (fobj);
	}

	virtual unsigned int Dimensions() const {
		return (2);
	}

	virtual void ProvideStartingPoint(std::vector<double>& x0) const {
		assert(x0.size() == 2);
		x0[0] = 0.03;
		x0[1] = 0.1;
	}
};

class Quadratic2DProblem : public Grante::FunctionMinimizationProblem {
public:
	virtual double Eval(const std::vector<double>& x,
		std::vector<double>& grad) {
		assert(x.size() == 2);
		assert(grad.size() == 2);

		// x'Ax - b'x,
		// A = [0.8, 0.4; 0.4, 0.5], b = [-0.3; 0.6].
		double fobj = (0.8*x[0] + 0.4*x[1])*x[0]
			+ (0.4*x[0] + 0.5*x[1])*x[1]
			- (-0.3)*x[0] - 0.6*x[1];

		grad[0] = 2.0*(0.8*x[0] + 0.4*x[1]) - (-0.3);
		grad[1] = 2.0*(0.4*x[0] + 0.5*x[1]) - 0.6;

		return (fobj);
	}

	virtual unsigned int Dimensions() const {
		return (2);
	}

	virtual void ProvideStartingPoint(std::vector<double>& x0) const {
		assert(x0.size() == 2);
		x0[0] = 0.0;
		x0[1] = 0.0;
	}
};

class Convex2DProblem : public Grante::FunctionMinimizationProblem {
public:
	virtual double Eval(const std::vector<double>& x,
		std::vector<double>& grad) {
		assert(x.size() == 2);
		assert(grad.size() == 2);

		double fobj = std::log(std::exp(2.0*x[0]*x[0] + 0.1*x[1])
			+ std::exp(0.1*x[0] + 1.4*x[1]*x[1]));
		assert(fobj >= 0.0);

		grad[0] = std::exp(-fobj)*(std::exp(2.0*x[0]*x[0] + 0.1*x[1])*(4.0*x[0])
			+ std::exp(0.1*x[0] + 1.4*x[1]*x[1])*0.1);
		grad[1] = std::exp(-fobj)*(std::exp(2.0*x[0]*x[0] + 0.1*x[1])*0.1
			+ std::exp(0.1*x[0] + 1.4*x[1]*x[1])*(2.8*x[1]));

		return (fobj);
	}

	virtual unsigned int Dimensions() const {
		return (2);
	}

	virtual void ProvideStartingPoint(std::vector<double>& x0) const {
		assert(x0.size() == 2);
		x0[0] = 3.7;
		x0[1] = -4.5;
	}
};

BOOST_AUTO_TEST_CASE(QuadraticBB)
{
	Quadratic2DProblem quad;

	BOOST_CHECK(Grante::FunctionMinimization::CheckDerivative(
		quad, 10.0, 100, 1e-8, 1e-5));

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::FunctionMinimization::BarzilaiBorweinMinimize(
		quad, x_opt, 1e-6, 0, false);

	BOOST_CHECK_CLOSE_ABS(fobj, -0.49687, 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(x_opt[0], -0.81250, 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(x_opt[1], 1.2500, 1.0e-5);
}

BOOST_AUTO_TEST_CASE(ConvexBB)
{
	Convex2DProblem bb;

	BOOST_CHECK(Grante::FunctionMinimization::CheckDerivative(
		bb, 10.0, 100, 1e-8, 1e-5));

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::FunctionMinimization::BarzilaiBorweinMinimize(
		bb, x_opt, 1e-13, 0, false);

	BOOST_CHECK(std::log(2.0) >= fobj);
	BOOST_CHECK_SMALL(x_opt[0], 5e-2);
	BOOST_CHECK_SMALL(x_opt[1], 5e-2);
}

BOOST_AUTO_TEST_CASE(ConvexGradientMethod)
{
	Convex2DProblem bb;

	BOOST_CHECK(Grante::FunctionMinimization::CheckDerivative(
		bb, 10.0, 100, 1e-8, 1e-5));

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::FunctionMinimization::GradientMethodMinimize(
		bb, x_opt, 1e-4, 0, true);

	BOOST_CHECK(std::log(2.0) >= fobj);
	BOOST_CHECK_SMALL(x_opt[0], 5e-2);
	BOOST_CHECK_SMALL(x_opt[1], 5e-2);
}

BOOST_AUTO_TEST_CASE(QuadraticSubgradientMethod)
{
	Quadratic2DProblem bb;

	BOOST_CHECK(Grante::FunctionMinimization::CheckDerivative(
		bb, 10.0, 100, 1e-8, 1e-5));

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::FunctionMinimization::SubgradientMethodMinimize(
		bb, x_opt, 1e-4, 0, false);

	BOOST_CHECK_CLOSE_ABS(fobj, -0.49687, 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(x_opt[0], -0.81250, 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(x_opt[1], 1.2500, 1.0e-2);
}

BOOST_AUTO_TEST_CASE(QuadraticGradientMethod)
{
	Quadratic2DProblem bb;

	BOOST_CHECK(Grante::FunctionMinimization::CheckDerivative(
		bb, 10.0, 100, 1e-8, 1e-5));

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::FunctionMinimization::GradientMethodMinimize(
		bb, x_opt, 1e-4, 0, false);

	BOOST_CHECK_CLOSE_ABS(fobj, -0.49687, 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(x_opt[0], -0.81250, 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(x_opt[1], 1.2500, 1.0e-2);
}

BOOST_AUTO_TEST_CASE(RosenbrockLBFGS)
{
//	Quadratic2DProblem bb;
//	Convex2DProblem bb;
	Rosenbrock2DProblem bb;

	BOOST_CHECK(Grante::FunctionMinimization::CheckDerivative(
		bb, 1.0, 100, 1e-8, 1e-1));

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::FunctionMinimization::LimitedMemoryBFGSMinimize(
		bb, x_opt, 1e-8, 0, true);

	BOOST_CHECK_CLOSE_ABS(0.0, fobj, 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(1.0, x_opt[0], 1e-5);
	BOOST_CHECK_CLOSE_ABS(1.0, x_opt[1], 1e-5);
}
