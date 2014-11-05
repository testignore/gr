
#ifndef GRANTE_COMPOSITE_MINIMIZATION_H
#define GRANTE_COMPOSITE_MINIMIZATION_H

#include <vector>

#include "CompositeMinimizationProblem.h"

namespace Grante {

class CompositeMinimization {
public:
	// FISTA method of (Beck, Teboulle, "A Fast Iterative
	// Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM
	// Journal of Imaging Sciences, Vol. 2, No. 1, pp. 183-202, 2009)
	//
	// This is the backtracking variant.
	static double FISTAMinimize(CompositeMinimizationProblem& prob,
		std::vector<double>& x_opt, double conv_tol,
		unsigned int max_iter = 0, bool verbose = true);
};

}

#endif

