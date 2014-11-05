
#ifndef GRANTE_STOCHASTICFUNCMINIMIZATION_H
#define GRANTE_STOCHASTICFUNCMINIMIZATION_H

#include <vector>

#include "StochasticFunctionMinimizationProblem.h"

namespace Grante {

class StochasticFunctionMinimization {
public:
	// Minimize a continous (not necessarily differentiable) unconstrained
	// function by means of the stochastic subgradient method.
	//
	// prob: The stochastic minimization problem, see
	//    StochasticFunctionMinimizationProblem.h.
	// x_opt: The resulting approximately optimal solution vector.  Does not
	//    have to be initialized.
	// conv_tol: The convergence tolerance as measured by the averaged
	//    subgradient Euclidean norm.
	// max_epochs: The maximum number of epochs.  If zero (default), there
	//    is no limit.
	// verbose: If true some statistics are printed during optimization.
	//
	// The return value is the estimated objective function value.
	static double StochasticSubgradientMethodMinimize(
		StochasticFunctionMinimizationProblem& prob,
		std::vector<double>& x_opt, double conv_tol,
		unsigned int max_epochs = 0, bool verbose = true);
};

}

#endif

