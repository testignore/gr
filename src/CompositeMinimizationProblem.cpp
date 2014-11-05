
#include "CompositeMinimizationProblem.h"

namespace Grante {

CompositeMinimizationProblem::~CompositeMinimizationProblem() {
}

double CompositeMinimizationProblem::Eval(const std::vector<double>& x,
	std::vector<double>& grad) {
	// Set gradient to zero
	std::fill(grad.begin(), grad.end(), 0.0);

	double obj = EvalF(x, grad);
	obj += EvalG(x, grad);

	return (obj);
}

}

