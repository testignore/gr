
#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <cassert>

#include "GibbsInference.h"
#include "LogSumExp.h"

namespace Grante {

GibbsInference::GibbsInference(const FactorGraph* fg)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::signaling_NaN()),
		gibbs(fg), burnin_sweeps(100), spacing_sweeps(0), sample_count(10000) {
}

GibbsInference::~GibbsInference() {
}

InferenceMethod* GibbsInference::Produce(const FactorGraph* fg) const {
	GibbsInference* ginf_new = new GibbsInference(fg);
	ginf_new->SetSamplingParameters(burnin_sweeps, spacing_sweeps,
		sample_count);

	return (ginf_new);
}

void GibbsInference::SetSamplingParameters(unsigned int burnin_sweeps,
	unsigned int spacing_sweeps, unsigned int sample_count) {
	assert(sample_count > 0);
	this->burnin_sweeps = burnin_sweeps;
	this->spacing_sweeps = spacing_sweeps;
	this->sample_count = sample_count;
}

void GibbsInference::PerformInference() {
	// 1. Setup marginals
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}

	PerformBurninPhase();

	// 3. Produce approximate samples
	double sample_contribution = 1.0 / static_cast<double>(sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		gibbs.Sweep(1 + spacing_sweeps);
		const std::vector<unsigned int>& sample = gibbs.State();

		// Add to marginals
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			marginals[fi][factors[fi]->ComputeAbsoluteIndex(sample)] +=
				sample_contribution;
		}
	}
}

void GibbsInference::PerformBurninPhase() {
	// 2. Burnin heuristic is based on annealed sampling
	//      i) random initialization,
	//     ii) annealing run for burnin_sweeps/2 sweeps,
	//    iii) burnin_sweeps/2 regular Gibbs sweeps.
	gibbs.SetStateUniformRandom();
	if (burnin_sweeps > 0) {
		unsigned int burnin_anneal = burnin_sweeps / 2;
		if (burnin_anneal > 1) {
			// The first non-zero beta,
			// 0 = beta_0 < beta_1 < beta_2 < ... < beta_K
			// beta_k = gamma^(k-1)*beta_1
			double beta_1 = 1.0e-4;
			double gamma = std::pow(1.0 / beta_1,
				1.0 / static_cast<double>(burnin_anneal-1));

			// Compute a single annealed sample
			double cur_beta = 0.0;
			for (unsigned int k = 0; k < burnin_anneal; ++k) {
				gibbs.SetInverseTemperature(cur_beta);
				gibbs.Sweep(1);

				cur_beta = std::pow(gamma, static_cast<double>(k)) * beta_1;
			}
			gibbs.SetInverseTemperature(1.0);
			gibbs.Sweep(1);
		}
		// Normal Gibbs sweeps
		gibbs.Sweep(burnin_sweeps - burnin_anneal);
	}
}

void GibbsInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& GibbsInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >& GibbsInference::Marginals() const {
	return (marginals);
}

double GibbsInference::LogPartitionFunction() const {
	return (std::numeric_limits<double>::signaling_NaN());
}

void GibbsInference::Sample(std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	PerformBurninPhase();

	states.resize(sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		gibbs.Sweep(1 + spacing_sweeps);
		states[si] = gibbs.State();
	}
}

double GibbsInference::MinimizeEnergy(std::vector<unsigned int>& state) {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}

