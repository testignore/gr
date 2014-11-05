
#include <numeric>
#include <cmath>

#include "FunctionMinimizationProblem.h"

namespace Grante {

FunctionMinimizationProblem::~FunctionMinimizationProblem() {
}

bool FunctionMinimizationProblem::HasConverged(const std::vector<double>& x,
	const std::vector<double>& grad, double conv_tol) const {
	double l2 = std::inner_product(grad.begin(), grad.end(), grad.begin(), 0.0);
	return (std::sqrt(l2) < conv_tol);
}

}

