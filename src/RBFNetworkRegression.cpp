
#include <algorithm>
#include <cassert>

#include "FunctionMinimization.h"
#include "RBFNetworkRegression.h"

namespace Grante {

RBFNetworkRegression::RBFNetworkRegression(unsigned int N, unsigned int d)
	: N(N), d(d), rbfnet(N, d) {
	param.resize(rbfnet.ParameterDimension());
	std::fill(param.begin(), param.end(), 0.0);
}

RBFNetworkRegression::RBFNetworkRegression(
	const std::vector<std::vector<double> >& prototypes)
	: N(prototypes.size()), d(prototypes[0].size()), rbfnet(prototypes) {
	param.resize(rbfnet.ParameterDimension());
	std::fill(param.begin(), param.end(), 0.0);
}

void RBFNetworkRegression::FixBeta(double beta) {
	rbfnet.FixBeta(beta);
}

double RBFNetworkRegression::Fit(const std::vector<std::vector<double> >& X,
	const std::vector<double>& Y, double conv_tol, unsigned int max_iter) {
	assert(X.size() >= N);
	assert(X.size() == Y.size());
	sample_X = &X;
	sample_Y = &Y;

	RBFL2Problem reg_prob(this);
	// FIXME: remove derivative check
	FunctionMinimization::CheckDerivative(reg_prob, 1.0, 200, 1.0e-6, 1.0e-4);
	double l2_err = FunctionMinimization::LimitedMemoryBFGSMinimize(
		reg_prob, param, conv_tol, max_iter, false, 50);

	return (l2_err);
}

double RBFNetworkRegression::Evaluate(const std::vector<double>& x) const {
	return (rbfnet.Evaluate(x, param));
}

RBFNetworkRegression::RBFL2Problem::RBFL2Problem(RBFNetworkRegression* reg_base)
	: reg_base(reg_base) {
}

RBFNetworkRegression::RBFL2Problem::~RBFL2Problem() {
}

double RBFNetworkRegression::RBFL2Problem::Eval(const std::vector<double>& x,
	std::vector<double>& grad) {
	double res = 0.0;
	size_t sample_count = reg_base->sample_X->size();

	std::fill(grad.begin(), grad.end(), 0.0);
	for (size_t i = 0; i < sample_count; ++i) {
		double y_pred = reg_base->rbfnet.Evaluate((*reg_base->sample_X)[i], x);
		double y_truth = (*reg_base->sample_Y)[i];

		reg_base->rbfnet.EvaluateGradient((*reg_base->sample_X)[i], x,
			grad, 0, -2.0*(y_truth - y_pred));
		res += (y_pred-y_truth) * (y_pred-y_truth);
	}
	return (res);
}

unsigned int RBFNetworkRegression::RBFL2Problem::Dimensions() const {
	return (static_cast<unsigned int>(reg_base->rbfnet.ParameterDimension()));
}

void RBFNetworkRegression::RBFL2Problem::ProvideStartingPoint(
	std::vector<double>& x0) const {
	assert(x0.size() == Dimensions());
	bool has_beta = reg_base->rbfnet.HasFixedBeta();
	size_t b_base = 0;
	if (has_beta == false) {
		x0[0] = 0.0;	// exp(beta) = 1
		b_base = 1;
	}

	for (size_t n = 0; n < reg_base->N; ++n) {
		x0[b_base+n] = 1.0;	// alpha
		if (reg_base->rbfnet.HasFixedPrototypes() == false) {
			// Use first samples as centers
			size_t c_n_start = b_base + reg_base->N + n*reg_base->d;
			for (size_t dp = 0; dp < reg_base->d; ++dp)
				x0[c_n_start+dp] = (*reg_base->sample_X)[n][dp];
		}
	}
}

}

