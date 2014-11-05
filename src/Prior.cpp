
#include <limits>
#include <cassert>

#include "Prior.h"

namespace Grante {

Prior::~Prior() {
}

void Prior::EvaluateProximalOperator(const std::vector<double>& u,
	double L, std::vector<double>& wprox) const {
	assert(0);
}

double Prior::EvaluateFenchelDual(const std::vector<double>& u,
	std::vector<double>& w_out) const {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}

