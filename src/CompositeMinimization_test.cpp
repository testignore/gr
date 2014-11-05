
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include "CompositeMinimization.h"
#include "CompositeMinimizationProblem.h"
#include "FunctionMinimization.h"

#define BOOST_TEST_MODULE(CompositeMinimizationTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

class QuadraticProblem : public Grante::CompositeMinimizationProblem {
public:
	virtual ~QuadraticProblem() {
	}

	virtual unsigned int Dimensions() const {
		return (2);
	}

	virtual void ProvideStartingPoint(std::vector<double>& x0) const {
		x0[0] = 0.0;
		x0[1] = 0.0;
	}

	virtual double EvalF(const std::vector<double>& x,
		std::vector<double>& grad) {
		double obj = 0.0;
		obj += 0.6*x[0] - 0.2*x[1];
		obj += 0.5 * (1.5*x[0]*x[0] + 2.0*0.4*x[0]*x[1] + 0.5*x[1]*x[1]);

		if (grad.empty() == false) {
			grad[0] = 0.6 + 1.5*x[0] + 0.4*x[1];
			grad[1] = -0.2 + 0.4*x[0] + 0.5*x[1];
		}

		return (obj);
	}
};

class QuadraticL1Problem : public QuadraticProblem {
public:
	virtual ~QuadraticL1Problem() {
	}

	virtual double EvalG(const std::vector<double>& x,
		std::vector<double>& grad) {
		assert(grad.empty());
		return (0.35*(std::fabs(x[0]) + std::fabs(x[1])));
	}

	virtual void EvalGProximalOperator(const std::vector<double>& u,
		double L, std::vector<double>& wprox) const {
		wprox[0] = std::max(0.0, std::fabs(u[0]) - 0.35/L);
		wprox[1] = std::max(0.0, std::fabs(u[1]) - 0.35/L);
		wprox[0] *= (u[0] < 0.0) ? -1.0 : 1.0;
		wprox[1] *= (u[1] < 0.0) ? -1.0 : 1.0;
	}
};

class QuadraticL2Problem : public QuadraticProblem {
public:
	virtual ~QuadraticL2Problem() {
	}

	virtual double EvalG(const std::vector<double>& x,
		std::vector<double>& grad) {
		if (grad.empty() == false) {
			grad[0] += 2.0*0.35*x[0];
			grad[1] += 2.0*0.35*x[1];
		}
		return (0.35*(x[0]*x[0] + x[1]*x[1]));
	}

	virtual void EvalGProximalOperator(const std::vector<double>& u,
		double L, std::vector<double>& wprox) const {
		double sfac = L / (2.0*0.35 + L);
		wprox[0] = sfac*u[0];
		wprox[1] = sfac*u[1];
	}
};

BOOST_AUTO_TEST_CASE(QuadraticL1Test)
{
	QuadraticL1Problem qpl1;

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::CompositeMinimization::FISTAMinimize(
		qpl1, x_opt, 1.0e-8, 0, true);

	BOOST_CHECK_CLOSE_ABS(fobj, -0.020833333, 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(x_opt[0], -0.1666666, 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(x_opt[1], 0.0, 1.0e-4);
}

BOOST_AUTO_TEST_CASE(QuadraticL2Test)
{
	QuadraticL2Problem qpl2;

	std::vector<double> x_opt(2, 0.0);
	double fobj = Grante::CompositeMinimization::FISTAMinimize(
		qpl2, x_opt, 1.0e-8, 0, true);

	BOOST_CHECK_CLOSE_ABS(fobj, -0.1242, 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(x_opt[0], -0.32258062, 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(x_opt[1], 0.2742, 1.0e-4);

	double bb_obj = Grante::FunctionMinimization::BarzilaiBorweinMinimize(
		qpl2, x_opt, 1.0e-10);
	BOOST_CHECK_CLOSE_ABS(bb_obj, -0.1242, 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(x_opt[0], -0.32258062, 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(x_opt[1], 0.2742, 1.0e-4);
}

