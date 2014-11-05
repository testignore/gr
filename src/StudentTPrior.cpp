
#include <cmath>
#include <cassert>
#include <boost/math/special_functions/gamma.hpp>

#include "StudentTPrior.h"

namespace Grante {

const double StudentTPrior::pi = 3.14159265358979323846;

StudentTPrior::StudentTPrior(double dof, double sigma, unsigned int dim)
	: dof(dof), sigma(sigma) {
	// Precompute two constants
	double d = static_cast<double>(dim);
	logp_constant1 = -boost::math::lgamma(0.5*(dof+d))
		+ boost::math::lgamma(0.5*dof) + 0.5*d*std::log(dof*pi)
		+ d*log(sigma);
	logp_constant2 = dof + d;
}

StudentTPrior::~StudentTPrior() {
}

// -log p(w) = -log Gamma((dof + dim)/2) + log Gamma(dof/2)
//             + (dim/2)*log(dof * pi) + dim*log(sigma)
//             + ((dof + dim)/2)*log(1 + (1/(dof*sigma^2))*w'*w)
// \nabla_w -log p(w) = ((dof+dim)/(dof*sigma^2))
//             * (1/(1+(1/(dof*sigma^2))*w'*w)) * w.
double StudentTPrior::EvaluateNegLogP(const std::vector<double>& w,
	std::vector<double>& grad, double scale) const {
	if (grad.empty() == false) {
		assert(w.size() == grad.size());
	}
	double xnorm = 0.0;
	for (unsigned int d = 0; d < w.size(); ++d)
		xnorm += w[d]*w[d];

	double nlogp = 1.0 + xnorm/(dof*sigma*sigma);
	double scale2 = (logp_constant2 / (dof*sigma*sigma)) / nlogp;
	if (grad.empty() == false) {
		for (unsigned int d = 0; d < w.size(); ++d)
			grad[d] += scale * scale2 * w[d];
	}

	double res = 0.5*logp_constant2*std::log(nlogp) + logp_constant1;
	return (scale * res);
}

}

