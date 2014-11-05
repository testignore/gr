
#ifndef GRANTE_LOGSUMEXP_H
#define GRANTE_LOGSUMEXP_H

#include <vector>

namespace Grante {

class LogSumExp {
public:
	// Compute the expression
	//    log sum_i exp(x_i),
	// in a numerically stable way.  Moreover, -inf elements are treated
	// correctly.
	static double Compute(const std::vector<double>& x);
	//    log sum_i exp(-x_i)
	static double ComputeNeg(const std::vector<double>& x);
};

}

#endif

