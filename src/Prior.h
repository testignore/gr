
#ifndef GRANTE_PRIOR_H
#define GRANTE_PRIOR_H

#include <vector>

namespace Grante {

class Prior {
public:
	virtual ~Prior();

	// Evaluate -log p(w) and \nabla_w -log p(w).
	//
	// w: The input point.
	// grad: (output) The gradient at w will be added to this vector.  If
	//    grad.empty() is true, no gradient computation is done.
	// scale: Both the returned value and the gradient are scaled with this
	//    multiplier.
	//
	// Return -log p(w).
	virtual double EvaluateNegLogP(const std::vector<double>& w,
		std::vector<double>& grad, double scale = 1.0) const = 0;

	// Evaluate the Euclidean proximal operator
	//    p_L(w) = argmin_w -log p(w) + (1/2)*L*|w-u|^2
	//
	// u: the proximal point, u.size()==w.size().
	// L: >0, scaling for the proximal term.
	// wprox: the unique minimizing point.
	//
	// This function does not need to be implemented, but is useful for fancy
	// sparsity-inducing priors, when used with the proximal optimization
	// methods.
	virtual void EvaluateProximalOperator(const std::vector<double>& u,
		double L, std::vector<double>& wprox) const;

	// Solve
	//    sup_{w} <w,u> + log p(w)    (1)
	//
	// u: the argument, u.size() == w.size().
	// grad: \nabla_u (1), must be properly sized (grad.size()==u.size()).
	//    The gradient will be added to grad.
	//
	// Return the value of the supremum (1).
	virtual double EvaluateFenchelDual(const std::vector<double>& u,
		std::vector<double>& grad) const;
};

}

#endif

