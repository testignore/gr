
#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <cassert>

#include "MultichainGibbsInference.h"
#include "LogSumExp.h"

namespace Grante {

MultichainGibbsInference::MultichainGibbsInference(const FactorGraph* fg)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::signaling_NaN()),
		number_of_chains(5), accept_psrf(1.1), spacing_sweeps(1),
		sample_count(10000) {
}

MultichainGibbsInference::~MultichainGibbsInference() {
}

InferenceMethod* MultichainGibbsInference::Produce(const FactorGraph* fg) const {
	MultichainGibbsInference* ginf_new = new MultichainGibbsInference(fg);

	ginf_new->SetSamplingParameters(number_of_chains, accept_psrf,
		spacing_sweeps, sample_count);

	return (ginf_new);
}

void MultichainGibbsInference::SetSamplingParameters(
	unsigned int number_of_chains, double accept_psrf,
	unsigned int spacing_sweeps, unsigned int sample_count) {
	assert(sample_count > 0);

	this->number_of_chains = number_of_chains;
	this->accept_psrf = accept_psrf;
	this->spacing_sweeps = spacing_sweeps;
	this->sample_count = sample_count;
}

void MultichainGibbsInference::SetupChains(void) {
	chain_mean.clear();
	chain_mean.resize(number_of_chains);
	chain_varm.clear();
	chain_varm.resize(number_of_chains);

	chain_gibbs.clear();
	chain_gibbs.reserve(number_of_chains);

	// Setup marginal means and variance vectors
	const std::vector<Factor*>& factors = fg->Factors();
	size_t factor_count = factors.size();
	for (unsigned int ci = 0; ci < number_of_chains; ++ci) {
		chain_mean[ci].resize(factor_count);
		chain_varm[ci].resize(factor_count);
		for (size_t fi = 0; fi < factor_count; ++fi) {
			chain_mean[ci][fi].resize(factors[fi]->Type()->ProdCardinalities());
			std::fill(chain_mean[ci][fi].begin(), chain_mean[ci][fi].end(), 0.0);

			chain_varm[ci][fi].resize(factors[fi]->Type()->ProdCardinalities());
			std::fill(chain_varm[ci][fi].begin(), chain_varm[ci][fi].end(), 0.0);
		}
		chain_gibbs.push_back(GibbsSampler(fg));
	}

	// Final inference result marginals
	marginals.resize(factor_count);
	for (unsigned int fi = 0; fi < factor_count; ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}
}

void MultichainGibbsInference::PerformInference() {
	std::cout << "SETUP" << std::endl;
	SetupChains();

	// Run chains until we are sure they converged to the stationary
	// distribution
	std::cout << "BURNIN" << std::endl;
	PerformBurninPhase();

	// Produce approximate samples, using all chains
	std::cout << "SAMPLE" << std::endl;
	unsigned int chain_i = 0;
	const std::vector<Factor*>& factors = fg->Factors();

	double sample_contribution = 1.0 / static_cast<double>(sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		chain_gibbs[chain_i].Sweep(1 + spacing_sweeps);
		const std::vector<unsigned int>& sample =
			chain_gibbs[chain_i].State();

		// Add to marginals
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			marginals[fi][factors[fi]->ComputeAbsoluteIndex(sample)] +=
				sample_contribution;
		}

		chain_i += 1;
		chain_i %= number_of_chains;
	}
}

void MultichainGibbsInference::PerformBurninPhase() {
	// Overdispersed initialization with the distribution of maximum entropy
	for (unsigned int ci = 0; ci < number_of_chains; ++ci)
		chain_gibbs[ci].SetStateUniformRandom();

	// Run chains until maximum per-dimension PSRF is below threshold
	unsigned int sweep_steps = 10;
	unsigned int total_sample_count = 0;
	while (true) {
		// Update all chains
		#pragma omp parallel for schedule(dynamic)
		for (int ci = 0; ci < static_cast<int>(number_of_chains); ++ci) {
			// Run a number of sweeps on this chain
			for (unsigned int swi = 0; swi < sweep_steps; ++swi) {
				chain_gibbs[ci].Sweep(1);

				UpdateMeanVariance(chain_gibbs[ci], chain_mean[ci],
					chain_varm[ci], total_sample_count+swi+1);
			}
		}
		total_sample_count += sweep_steps;

		if (total_sample_count >= 2) {
			double current_psrf = ComputePSRF(total_sample_count);
			std::cout << number_of_chains << " chains, "
				<< total_sample_count << " samples each, "
				<< "maxPSRF " << current_psrf << std::endl;

			if (current_psrf <= accept_psrf)
				break;
		}
	}
}

// n: this is the n'th sample, where n >= 1.
void MultichainGibbsInference::UpdateMeanVariance(const GibbsSampler& gibbs,
	marginals_t& mean, marginals_t& varm, unsigned int n) {
	// Current sampler state
	const std::vector<unsigned int>& state = gibbs.State();

	const std::vector<Factor*>& factors = fg->Factors();
	assert(mean.size() == factors.size());
	assert(varm.size() == factors.size());

	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// Compute the indicator index of the current sample for this factor,
		// this is an observation=1, where all other indices are
		// observation=0.
		unsigned int ai = factors[fi]->ComputeAbsoluteIndex(state);

		for (unsigned int ei = 0; ei < mean[fi].size(); ++ei) {
			// Online mean/variance update algorithm
			double x = (ei == ai) ? 1.0 : 0.0;
			double diff = x - mean[fi][ei];
			mean[fi][ei] += diff / static_cast<double>(n);
			varm[fi][ei] += diff*(x - mean[fi][ei]);
		}
	}
}

double MultichainGibbsInference::ComputePSRF(unsigned int n) const {
	// We compute the maximum PSRF over all marginal dimensions
	double max_psrf = -std::numeric_limits<double>::infinity();

	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		size_t msize = factors[fi]->Type()->ProdCardinalities();
		for (size_t ei = 0; ei < msize; ++ei) {
			double bar_t = 0.0;
			for (unsigned int ci = 0; ci < number_of_chains; ++ci)
				bar_t += chain_mean[ci][fi][ei];
			bar_t /= static_cast<double>(number_of_chains);

			// Estimate between-chain variance B and within-chain variance W
			double B = 0.0;
			double W = 0.0;
			for (unsigned int ci = 0; ci < number_of_chains; ++ci) {
				B += std::pow(chain_mean[ci][fi][ei] - bar_t, 2.0);
				W += chain_varm[ci][fi][ei] / static_cast<double>(n - 1);
			}

			B *= static_cast<double>(n);
			B /= static_cast<double>(number_of_chains - 1);
			W /= static_cast<double>(number_of_chains);
#ifdef DEBUG
			std::cout << "B " << B << "  W " << W << std::endl;
#endif

			// Posterior marginal variance estimate
			double V_hat =
				(static_cast<double>(n-1) / static_cast<double>(n)) * W
				+ (static_cast<double>(number_of_chains + 1) /
					static_cast<double>(n * number_of_chains)) * B;
			assert(V_hat >= 0.0);

			double psrf = 0.0;
			if (W >= 1.0e-12)
				psrf = std::sqrt(V_hat / W);

			max_psrf = std::max(psrf, max_psrf);
		}
	}
	return (max_psrf);
}

void MultichainGibbsInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& MultichainGibbsInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >& MultichainGibbsInference::Marginals() const {
	return (marginals);
}

double MultichainGibbsInference::LogPartitionFunction() const {
	return (std::numeric_limits<double>::signaling_NaN());
}

void MultichainGibbsInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	SetupChains();
	PerformBurninPhase();

	states.resize(sample_count);
	unsigned int chain_i = 0;
	for (unsigned int si = 0; si < sample_count; ++si) {
		chain_gibbs[chain_i].Sweep(1 + spacing_sweeps);
		states[si] = chain_gibbs[chain_i].State();

		chain_i += 1;
		chain_i %= number_of_chains;
	}
}

double MultichainGibbsInference::MinimizeEnergy(std::vector<unsigned int>& state) {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}

