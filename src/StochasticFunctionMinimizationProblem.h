
#ifndef GRANTE_STOCHASTICFUNCMINPROBLEM_H
#define GRANTE_STOCHASTICFUNCMINPROBLEM_H

#include <vector>
#include <cstddef>

namespace Grante {

/* Definition of a stochastic unconstrained minimization problem.  The
 * function is assumed to decompose additively in a fixed number of
 * 'elements' such that the objective and gradient of each element can be
 * evaluated separately for each element.
 * In the learning setting the elements are usually sample instances and the
 * functions they decompose into are loss functions over iid samples.
 */
class StochasticFunctionMinimizationProblem {
public:
	virtual ~StochasticFunctionMinimizationProblem();

	// Evaluate the function at a given query point for a given sample and
	// return the function objective and gradient vector with respect to that
	// sample.
	//
	// sample_id: The sample instance id (between 0 and NumberOfElements()-1).
	// x: The query point.
	// grad: (output) the gradient vector at x.
	//
	// The return value is equal to the function value f(x).
	virtual double Eval(unsigned int sample_id, const std::vector<double>& x,
		std::vector<double>& grad) = 0;

	// Return the number of dimensions of the problem.
	virtual unsigned int Dimensions() const = 0;

	// Number of sample elements
	virtual size_t NumberOfElements() const = 0;

	// Return in 'x0' a starting point for optimization.
	// The provided vector must already have the correct dimension, i.e.
	// x0.size() == Dimensions().
	virtual void ProvideStartingPoint(std::vector<double>& x0) const = 0;
};

}

#endif

