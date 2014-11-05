
#include <vector>
#include <limits>
#include <cmath>
#include <cassert>

#include "LogSumExp.h"
#include "AISInference.h"

namespace Grante {

AISInference::AISInference(const FactorGraph* fg)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::signaling_NaN()),
		gibbs(fg), K(80), gibbs_sweeps(1), sample_count(1000) {
}

AISInference::~AISInference() {
}

InferenceMethod* AISInference::Produce(const FactorGraph* fg) const {
	return (new AISInference(fg));
}

void AISInference::SetSamplingParameters(unsigned int anneal_k,
	unsigned int gibbs_sweeps, unsigned int sample_count) {
	assert(anneal_k >= 2);
	this->K = anneal_k;
	assert(gibbs_sweeps >= 1);
	this->gibbs_sweeps = gibbs_sweeps;
	assert(sample_count >= 1);
	this->sample_count = sample_count;
}

void AISInference::PerformInference() {
	// The first non-zero beta,
	// 0 = beta_0 < beta_1 < beta_2 < ... < beta_K
	double beta_1 = 1.0e-4;
	// beta_k = gamma^(k-1)*beta_1
	double gamma = std::pow(1.0 / beta_1, 1.0 / static_cast<double>(K-1));

	std::vector<double> logw(sample_count);
	std::vector<std::vector<unsigned int> > samples(sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		// Compute annealed samples
		double logw_cur = 0.0;
		double prev_beta = 0.0;	// beta_0
		double cur_beta = 0.0;
		for (unsigned int k = 0; k < K; ++k) {
			gibbs.SetInverseTemperature(cur_beta);
			gibbs.Sweep(gibbs_sweeps);

			prev_beta = cur_beta;
			cur_beta = std::pow(gamma, static_cast<double>(k)) * beta_1;

			// Add (log p_k(v_k) - log p_{k-1}(v_k))
			double cur_energy = fg->EvaluateEnergy(gibbs.State());
			logw_cur += (prev_beta-cur_beta)*cur_energy;
		}
		logw[si] = logw_cur;
		//std::cout << "logw[" << si << "]: " << logw[si] << std::endl;

		// Save state
		gibbs.SetInverseTemperature(1.0);
		gibbs.Sweep(gibbs_sweeps);
		samples[si] = gibbs.State();
	}
	double log_ZA = 0.0;
	const std::vector<unsigned int>& card = fg->Cardinalities();
	for (unsigned int vi = 0; vi < card.size(); ++vi)
		log_ZA += std::log(static_cast<double>(card[vi]));

	// Compute AIS approximation to log_z
	double logw_lse = LogSumExp::Compute(logw);
	log_z = -std::log(static_cast<double>(sample_count)) + log_ZA + logw_lse;

	// TODO: optionally compute empirical variance (how to do this in a stable
	// way?)  Neal's paper has normality results for logw, maybe std for logw
	// gives us the value
	double logw_mean = 0.0;
	double logw_var = 0.0;
	for (unsigned int si = 0; si < sample_count; ++si) {
		double ld = logw[si] - logw_mean;
		logw_mean += ld / static_cast<double>(si + 1);
		logw_var += ld*(logw[si] - logw_mean);
	}
	// logw_var: the population variance of the logw's.  According to Neal
	// the logw will be asymptotically Normal.
	logw_var /= static_cast<double>(sample_count - 1);

	// Setup marginals
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}
	// Compute marginals from weighted samples
	double total = 0.0;
	for (unsigned int si = 0; si < sample_count; ++si) {
		double sample_contribution = std::exp(logw[si] - logw_lse);
		total += sample_contribution;

		// Add to marginals
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			marginals[fi][factors[fi]->ComputeAbsoluteIndex(samples[si])] +=
				sample_contribution;
		}
	}
	assert(std::fabs(total - 1.0) <= 1.0e-6);
}

void AISInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& AISInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >& AISInference::Marginals() const {
	return (marginals);
}

double AISInference::LogPartitionFunction() const {
	return (log_z);
}

void AISInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	// Not supported
	assert(0);
}

double AISInference::MinimizeEnergy(std::vector<unsigned int>& state) {
	assert(0);
	// Not supported
	return (std::numeric_limits<double>::signaling_NaN());
}

}

