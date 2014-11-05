
#include <cmath>
#include <cassert>

#include "LaplacePrior.h"

namespace Grante {

LaplacePrior::LaplacePrior(double sigma, unsigned int dim)
	: sigma(sigma) {
	logp_constant = 2.0 * static_cast<double>(dim) * std::log(4.0*sigma);
}

LaplacePrior::~LaplacePrior() {
}

double LaplacePrior::EvaluateNegLogP(const std::vector<double>& w,
	std::vector<double>& grad, double scale) const {
	double mconst = 1.0 / (2.0 * sigma * sigma);
	double nlogp = 0.0;
	if (grad.empty() == false) {
		assert(w.size() == grad.size());
	}
	for (unsigned int d = 0; d < w.size(); ++d) {
		nlogp += std::fabs(w[d]);
		if (grad.empty() == false) {
			grad[d] += w[d] >= 0.0 ? (scale*mconst) : (-scale*mconst);
		}
	}
	nlogp = scale * (mconst*nlogp + logp_constant);

	return (nlogp);
}

void LaplacePrior::EvaluateProximalOperator(const std::vector<double>& u,
	double L, std::vector<double>& wprox) const {
	// The problem argmin_w -log p(w) + (1/2)*L*|w-u|^2 for the Laplacian case
	// has a well-known closed form solution.
	assert(u.size() == wprox.size());
	double sub = 1.0 / (L * 2.0 * sigma * sigma);
	for (unsigned int d = 0; d < u.size(); ++d) {
		double p = std::fabs(u[d]) - sub;
		if (p < 0.0) {
			p = 0.0;
		} else if (u[d] < 0.0) {
			p = -p;
		}
		wprox[d] = p;
	}
}

}

