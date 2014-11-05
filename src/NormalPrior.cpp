
#include <algorithm>
#include <cmath>
#include <cassert>

#include "NormalPrior.h"

namespace Grante {

const double NormalPrior::pi = 3.14159265358979323846;

NormalPrior::NormalPrior(double sigma, unsigned int dim)
	: sigma(sigma) {
	logp_constant = static_cast<double>(dim) * std::log(sigma * sqrt(2.0*pi));
}

NormalPrior::~NormalPrior() {
}

// -log p(w) = 0.5 sigma^-2 * w'w + dim*log(sigma sqrt(2 pi))
// \nabla_w -log p(w) = sigma^-2 w
double NormalPrior::EvaluateNegLogP(const std::vector<double>& w,
	std::vector<double>& grad, double scale) const {
	if (grad.empty() == false) {
		assert(w.size() == grad.size());
	}
	double mconst = 1.0 / (sigma*sigma);
	double nlogp = 0.0;
	for (unsigned int d = 0; d < w.size(); ++d) {
		nlogp += w[d]*w[d];
		if (grad.empty() == false)
			grad[d] += scale * mconst * w[d];
	}
	nlogp = scale * (0.5*mconst*nlogp + logp_constant);

	return (nlogp);
}

// wprox = (L/(sigma^-2 + L)) u.
void NormalPrior::EvaluateProximalOperator(const std::vector<double>& u,
	double L, std::vector<double>& wprox) const {
	double sfac = L / (1.0/(sigma*sigma) + L);
	std::transform(u.begin(), u.end(), wprox.begin(),
		[sfac](double ue) -> double { return (sfac*ue); });
}

// Omega^*(u) = 0.5 sigma^2 u'u - dim*log(sigma sqrt(2 pi))
//       w(u) = sigma^2 u  (not returned)
// \nabla_u Omega^*(u) = sigma^2 u
double NormalPrior::EvaluateFenchelDual(const std::vector<double>& u,
	std::vector<double>& grad) const {
	assert(grad.size() == u.size());
	double sigma_squared = sigma*sigma;
	std::transform(u.begin(), u.end(), grad.begin(),
		[sigma_squared](double ue) -> double { return (sigma_squared*ue); });

	double obj = 0.0;
	for (unsigned int d = 0; d < u.size(); ++d)
		obj += u[d] * u[d];

	return (0.5*sigma_squared*obj - logp_constant);
}

}

