
#ifndef GRANTE_HYPERBOLICPRIOR_H
#define GRANTE_HYPERBOLICPRIOR_H

#include "Prior.h"

namespace Grante {

/* k-dimensional hyperbolic distribution, special case (no skew, zero mean),
 *
 * p(w;alpha,delta) =
 *     ( ((alpha/delta)^((d+1)/2)) /
 *        ((2*pi)^((d-1)/2)*2*alpha*K_{(d+1)/2}(delta*alpha)) ) *
 *     exp{ -alpha * sqrt(delta^2 + (w'*w)) },
 *
 * where K_d(b) denotes the modified Bessel function of the second kind.
 */
class HyperbolicPrior : public Prior {
public:
	HyperbolicPrior(unsigned int dim, double alpha = 1.0,
		double delta = 1.0);
	virtual ~HyperbolicPrior();

	virtual double EvaluateNegLogP(const std::vector<double>& w,
		std::vector<double>& grad, double scale = 1.0) const;

private:
	double logp_constant;

	double alpha;
	double delta;

	static const double pi;
};

}

#endif

