
#ifndef GRANTE_STUDENTTPRIOR_H
#define GRANTE_STUDENTTPRIOR_H

#include <vector>

#include "Prior.h"

namespace Grante {

/* Multivariate isotropic student-t distribution.
 */
class StudentTPrior : public Prior {
public:
	/* p(w) = (Gamma((dof + dim) / 2) /
	 *           (Gamma(dof/2) (dof pi)^(dim/2) det(Sigma)^(1/2))) *
	 *        (1 + (1/dof) w' Sigma^-1 w)^(-(dof+dim)/2),
	 *
	 * where dof > 0 are the degrees-of-freedom of the student-t distribution
	 * (for dof->inf we recover the Normal, whereas dof->0 becomes the Delta
	 * distribution), and we use an isotropic inverse covariance matrix:
	 * Sigma^-1 = (1/(sigma*sigma))*I.
	 */
	StudentTPrior(double dof, double sigma, unsigned int dim);

	virtual ~StudentTPrior();

	virtual double EvaluateNegLogP(const std::vector<double>& w,
		std::vector<double>& grad, double scale = 1.0) const;

private:
	double dof;
	double sigma;
	double logp_constant1;
	double logp_constant2;

	static const double pi;
};

}

#endif

