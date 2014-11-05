
#ifndef GRANTE_FUNCTIONMINIMIZATIONPROBLEM_H
#define GRANTE_FUNCTIONMINIMIZATIONPROBLEM_H

#include <vector>

namespace Grante {

class FunctionMinimizationProblem {
public:
	virtual ~FunctionMinimizationProblem();

	// Evaluate the function at a given query point and return the function
	// objective and gradient vector.
	//
	// x: The query point.
	// grad: (output) the gradient vector at x.
	//
	// The return value is equal to the function value f(x).
	virtual double Eval(const std::vector<double>& x,
		std::vector<double>& grad) = 0;

	// Return the number of dimensions of the problem.
	virtual unsigned int Dimensions() const = 0;

	// Return in 'x0' a starting point for optimization.
	// The provided vector must already have the correct dimension, i.e.
	// x0.size() == Dimensions().
	virtual void ProvideStartingPoint(std::vector<double>& x0) const = 0;

	// Allow user-specified convergence criterion.
	//
	// x: current iterate,
	// grad: gradient vector at the current iterate,
	// conv_tol: user-provided convergence tolerance parameter.
	//
	// The default test is: |grad| < conv_tol.
	//
	// Return true in case the iterate has converged to the desired tolerance.
	virtual bool HasConverged(const std::vector<double>& x,
		const std::vector<double>& grad, double conv_tol) const;
};

}

#endif

