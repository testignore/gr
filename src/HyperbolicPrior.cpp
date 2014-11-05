
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tr1/cmath>

#include "HyperbolicPrior.h"

namespace Grante {

const double HyperbolicPrior::pi = 3.14159265358979323846;

HyperbolicPrior::HyperbolicPrior(unsigned int dim, double alpha,
	double delta)
	: logp_constant(0.0), alpha(alpha), delta(delta) {
	double d = dim;
	// These are the constant parts of -log p(x), that do not depend on x.
	// Because they involve a large number of logarithms and the Bessel
	// function, we precompute them once.
	logp_constant += std::log(alpha)*(1.0 - 0.5*(d+1.0));
	logp_constant += std::log(delta)*(0.5*(d+1.0));
	logp_constant += std::log(
		std::tr1::cyl_bessel_k(0.5*(d+1.0), delta*alpha));
	logp_constant += 0.5*(d-1.0)*std::log(2.0*pi) + std::log(2.0);
}

HyperbolicPrior::~HyperbolicPrior() {
}

double HyperbolicPrior::EvaluateNegLogP(const std::vector<double>& w,
	std::vector<double>& grad, double scale) const {
	// Compute -log p(w)
	double nlogp = logp_constant;
	double ww = std::inner_product(w.begin(), w.end(), w.begin(), 0.0);
	double sq_part = std::sqrt(delta*delta + ww);
	nlogp += alpha * sq_part;

	// Compute gradient, if needed
	if (grad.empty() == false) {
		double gr_scale = (scale * 2.0 * alpha) / sq_part;
		for (unsigned int wi = 0; wi < w.size(); ++wi)
			grad[wi] += gr_scale * w[wi];
	}

	return (scale * nlogp);
}

}

