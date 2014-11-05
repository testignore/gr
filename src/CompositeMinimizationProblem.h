
#ifndef GRANTE_COMPOSITEMINIMIZATIONPROBLEM_H
#define GRANTE_COMPOSITEMINIMIZATIONPROBLEM_H

#include <vector>

#include "FunctionMinimizationProblem.h"

namespace Grante {

// F(x) = f(x) + g(x), where
//
// f: smooth convex, continuously differentiable,
// g: continuous convex, possibly non-smooth.
class CompositeMinimizationProblem : public FunctionMinimizationProblem {
public:
	virtual ~CompositeMinimizationProblem();

	// Return the number of dimensions of the problem.
	virtual unsigned int Dimensions() const = 0;

	// Return in 'x0' a starting point for optimization.
	// The provided vector must already have the correct dimension, i.e.
	// x0.size() == Dimensions().
	virtual void ProvideStartingPoint(std::vector<double>& x0) const = 0;

	// Evaluate the function at a given query point and return the function
	// objective and a (sub-)gradient vector.
	//
	// x: The query point.
	// grad: (output) the (sub-)gradient vector at x.
	//
	// The return value is equal to f(x)+g(x).
	virtual double Eval(const std::vector<double>& x,
		std::vector<double>& grad);

	// EvalF takes two roles, depending on whether a batch or stochastic
	// optimization algorithm is used to minimize the composite function.
	// In the batch case, EvalF evaluates f(x) exactly and returns the value
	// f(x) and additionally the gradient in 'grad'.
	// In the stochastic case, EvalF returns an unbiased estimate of both
	// quantities, the objective and gradient.
	// If grad.empty()==true, no gradient is computed.
	virtual double EvalF(const std::vector<double>& x,
		std::vector<double>& grad) = 0;

	// Evaluate the function value of g(x) exactly.
	// If subgrad.empty() == false, then a subgradient is returned.
	virtual double EvalG(const std::vector<double>& x,
		std::vector<double>& subgrad) = 0;

	// Evaluate the proximal operator associated to g, that is, solve
	//     argmin_w g(w) + (1/2)*L*|w-u|^2.
	//
	// The solution must be exact, ideally up to machine precision.  The
	// result is returned in wprox.
	virtual void EvalGProximalOperator(const std::vector<double>& u,
		double L, std::vector<double>& wprox) const = 0;
};

}

#endif

