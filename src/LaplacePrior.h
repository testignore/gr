
#ifndef GRANTE_LAPLACEPRIOR_H
#define GRANTE_LAPLACEPRIOR_H

#include "Prior.h"

namespace Grante {

/* Multivariate Laplace distribution, a special case of the multivariate
 * p-generalized Normal distribution.  See,
 *
 * Fabian Sinz, Sebastian Gerwinn, and Matthias Bethge,
 * "Characterization of the p-generalized Normal distribution",
 * Multivariate Analysis, Vol. 100, No. 5, pp 817-820, May 2009.
 */
class LaplacePrior : public Prior {
public:
	LaplacePrior(double sigma, unsigned int dim);
	virtual ~LaplacePrior();

	virtual double EvaluateNegLogP(const std::vector<double>& w,
		std::vector<double>& grad, double scale = 1.0) const;

	virtual void EvaluateProximalOperator(const std::vector<double>& u,
		double L, std::vector<double>& wprox) const;

private:
	double sigma;
	double logp_constant;	// 2*dim*log(4*sigma)
};

}

#endif

