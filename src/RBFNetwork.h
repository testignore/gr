
#ifndef GRANTE_RBFNETWORK_H
#define GRANTE_RBFNETWORK_H

#include <vector>

namespace Grante {

/* Simple Radial Basis Function Network for regression.  The network is of the
 * form
 *    f(x) = \sum_{n=1}^N \alpha_n exp(-exp(\beta) |x-c_n|_2^2),
 * where
 *    \beta \in \R is a global bandwidth parameters,
 *    \alpha_n \in \R are signed mixing weights, and
 *    c_n \in \R^d are radial basis function centers in the input space.
 */
class RBFNetwork {
public:
	// Learnable prototypes
	RBFNetwork(unsigned int N, unsigned int d);
	// Fixed prototypes
	RBFNetwork(const std::vector<std::vector<double> >& prototypes);

	// Fix beta to a value and remove it from the parameter vector.
	// Must be called directly after constructor
	void FixBeta(double log_beta);

	// Number of total scalar parameters of this network
	size_t ParameterDimension() const;
	bool HasFixedPrototypes() const;
	bool HasFixedBeta() const;

	// Evaluate real-valued response f(x)
	double Evaluate(const std::vector<double>& x,
		const std::vector<double>& param, size_t param_base = 0) const;

	// Return f(x) and return scale*\nabla_param f(x) in grad.  Hence
	// grad.size() must be equal to param.size().
	double EvaluateGradient(const std::vector<double>& x,
		const std::vector<double>& param, std::vector<double>& grad,
		size_t param_base = 0, double scale = 1.0) const;

private:
	size_t N;
	size_t d;

	// Only used when param does not contain the center locations
	std::vector<std::vector<double> > prototypes;
	bool has_proto;

	double beta;
	bool has_beta;

	// Parameter ordering for 'param' is [beta, alpha, c_1, c_2, ..., c_N].
	double EvaluateRBFFunction(const std::vector<double>& x,
		const std::vector<double>& param, unsigned int n,
		size_t param_base = 0) const;
	double EvaluateL2(const std::vector<double>& x,
		const std::vector<double>& param, unsigned int n,
		size_t param_base = 0) const;
};

}

#endif

