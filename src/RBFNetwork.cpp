
#include <numeric>
#include <iostream>
#include <cmath>
#include <cassert>

#include "RBFNetwork.h"

namespace Grante {

RBFNetwork::RBFNetwork(unsigned int N, unsigned int d)
	: N(N), d(d), has_proto(false), has_beta(false) {
}

RBFNetwork::RBFNetwork(const std::vector<std::vector<double> >& prototypes)
	: prototypes(prototypes), has_proto(true), has_beta(false) {
	assert(prototypes.size() > 0);
	d = prototypes[0].size();
	N = prototypes.size();
	for (unsigned int pi = 0; pi < prototypes.size(); ++pi) {
		assert(prototypes[pi].size() == d);
	}
}

void RBFNetwork::FixBeta(double log_beta) {
	this->beta = log_beta;
	has_beta = true;
}

size_t RBFNetwork::ParameterDimension() const {
	return ((has_proto ? (N) : (N + N*d)) + (has_beta ? 0 : 1));
}

bool RBFNetwork::HasFixedPrototypes() const {
	return (has_proto);
}

bool RBFNetwork::HasFixedBeta() const {
	return (has_beta);
}

double RBFNetwork::Evaluate(const std::vector<double>& x,
	const std::vector<double>& param, size_t param_base) const {
	double res = 0.0;
	for (unsigned int n = 0; n < N; ++n) {
		// += alpha_n * rbf(x)
		res += param[param_base+(has_beta ? 0 : 1)+n] *
			EvaluateRBFFunction(x, param, n, param_base);
	}
	return (res);
}

double RBFNetwork::EvaluateGradient(const std::vector<double>& x,
	const std::vector<double>& param, std::vector<double>& grad,
	size_t param_base, double scale) const {
	assert(param.size() == grad.size());
	assert(param_base < param.size());
	double res = 0.0;
	double exp_beta = has_beta ?
		std::exp(beta) : std::exp(param[param_base+0]);
	unsigned int b_base = has_beta ? 0 : 1;
	for (unsigned int n = 0; n < N; ++n) {
		double l2_resp = EvaluateL2(x, param, n, param_base);
		double cur_rbf_resp = std::exp(-exp_beta*l2_resp);
		double alpha_n = param[param_base+b_base+n];

		// 0. RBF response
		res += alpha_n * cur_rbf_resp;

		// 1. \nabla_{alpha_n} f(x) = rbf_resp[n]
		grad[param_base+b_base+n] += scale * cur_rbf_resp;

		// 2. \nabla_beta
		if (has_beta == false) {
			grad[param_base+0] += scale * -alpha_n * cur_rbf_resp * l2_resp *
				exp_beta;
		}

		if (has_proto == false) {
			// 3. \nabla_{c_n}
			size_t c_n_start = param_base + b_base + N + n*d;
			for (size_t dp = 0; dp < d; ++dp) {
				grad[c_n_start+dp] += scale * -2.0 * exp_beta *
					alpha_n * cur_rbf_resp * (param[c_n_start+dp] - x[dp]);
			}
		}
	}
	return (res);
}

double RBFNetwork::EvaluateRBFFunction(const std::vector<double>& x,
	const std::vector<double>& param, unsigned int n,
	size_t param_base) const {
	double exp_beta = has_beta ?
		std::exp(beta) : std::exp(param[param_base+0]);

	double l2 = EvaluateL2(x, param, n, param_base);
	double res = std::exp(-exp_beta*l2);
#if 0
	std::cout << "res " << res << std::endl;
	std::cout << "  exp_beta * L2 = " << exp_beta << " * " << l2 << std::endl;
#endif
	return (res);
}

double RBFNetwork::EvaluateL2(const std::vector<double>& x,
	const std::vector<double>& param, unsigned int n,
	size_t param_base) const {
	double xcn_diff = 0.0;
	if (has_proto) {
		for (unsigned int dp = 0; dp < d; ++dp)
			xcn_diff += (x[dp]-prototypes[n][dp]) * (x[dp]-prototypes[n][dp]);
	} else {
		size_t c_n_start = (has_beta ? 0 : 1) + N + n*d;
		for (size_t dp = 0; dp < d; ++dp) {
			xcn_diff += (x[dp]-param[param_base+c_n_start+dp]) *
				(x[dp]-param[param_base+c_n_start+dp]);
		}
	}
	return (xcn_diff);
}

}

