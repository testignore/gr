
#ifndef GRANTE_NORMALPRIOR_H
#define GRANTE_NORMALPRIOR_H

#include <vector>

#include "Prior.h"

namespace Grante {

class NormalPrior : public Prior {
public:
	// Create a multivariate Normal N(0; sigma I)
	//
	// sigma: standard deviation, isotropic.
	// dim: number of dimensions.
	NormalPrior(double sigma, unsigned int dim);

	virtual ~NormalPrior();

	// -log p(w) = 0.5 sigma^-2 * w'w + dim*log(sigma sqrt(2 pi))
	// \nabla_w -log p(w) = sigma^-2 w
	virtual double EvaluateNegLogP(const std::vector<double>& w,
		std::vector<double>& grad, double scale = 1.0) const;

	// wprox = (L/(sigma^-2 + L)) u.
	virtual void EvaluateProximalOperator(const std::vector<double>& u,
		double L, std::vector<double>& wprox) const;

	// w_out = 1/(sigma^-2) u
	// value: ( 1/(sigma^-2) - 1/(sigma^-4) ) * (u'*u)
	virtual double EvaluateFenchelDual(const std::vector<double>& u,
		std::vector<double>& w_out) const;

private:
	double sigma;
	double logp_constant;	// dim*log(sigma sqrt(2 pi))

	// C/C++ do not have a standard definition of \pi
	static const double pi;
};

}

#endif

