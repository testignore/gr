
#include <algorithm>
#include <limits>
#include <iostream>
#include <cmath>
#include <cassert>

#include "RandomSource.h"
#include "GibbsSampler.h"
#include "SAMCInference.h"

namespace Grante {

SAMCInference::SAMCInference(const FactorGraph* fg)
	: InferenceMethod(fg), rgen(RandomSource::GetGlobalRandomSeed()),
		randu(rgen, rdestu), levels(20), high_temp(20.0),
		swap_probability(0.5), burnin_sweeps(1000), sample_count(1000) {
}

SAMCInference::~SAMCInference() {
}

InferenceMethod* SAMCInference::Produce(
	const FactorGraph* fg) const {
	SAMCInference* pt = new SAMCInference(fg);
	pt->SetSamplingParameters(levels, high_temp, swap_probability,
		burnin_sweeps, sample_count);

	return (pt);
}

void SAMCInference::SetSamplingParameters(unsigned int levels,
	double high_temp, double swap_probability, unsigned int burnin_sweeps,
	unsigned int sample_count) {
	this->levels = levels;
	this->high_temp = high_temp;
	this->swap_probability = swap_probability;
	this->burnin_sweeps = burnin_sweeps;
	this->sample_count = sample_count;
}

void SAMCInference::PerformInference() {
	PerformInference(false, sample_count);
}

void SAMCInference::PerformInference(bool keep_samples,
	unsigned int sample_count) {
	// 1. Setup marginals
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}

	// 2. Setup temperature ladder
	InitializeLadder(fg, levels, high_temp);

	// Initialize sampler
	GibbsSampler gibbs(fg);
	unsigned int cli = levels - 1;	// current temperature level
	gibbs.SetInverseTemperature(1.0 / temperatures[cli]);
	gibbs.SetStateUniformRandom();

	// SAMC weight update parameters
	double gf_t0 = 100.0;
	double gf_m = 1.0;
	double gf_alpha = 5.0;
	double gf_xi = 0.7;

	// 3. SAMC runs
	unsigned int si = 0;
	double sample_contribution = 1.0 / static_cast<double>(sample_count);
	unsigned int ksi = 0;
	while (si < (burnin_sweeps + sample_count)) {
#ifdef DEBUG
		for (unsigned int li = 0; li < cli; ++li)
			std::cout << ".";
		std::cout << "*";
		for (unsigned int li = cli+1; li < levels; ++li)
			std::cout << ".";
		std::cout << std::endl;
#endif

		if (randu() >= swap_probability) {
			// In-temperature transition
			gibbs.Sweep(1);
		} else {
			// Attempt temperature transition
			unsigned int tli = static_cast<unsigned int>(
				static_cast<double>(levels) * randu());
			assert(tli < levels);

			double state_energy = fg->EvaluateEnergy(gibbs.State());
			double cur_energy = state_energy / temperatures[cli];
			double target_energy = state_energy / temperatures[tli];

			// Acceptance probability
			double r = std::exp(cur_energy + theta[cli]
				- (target_energy + theta[tli]));
			r = std::min(1.0, r);

			if (randu() <= r) {
				// Accept temperature transition
				cli = tli;
				gibbs.SetInverseTemperature(1.0 / temperatures[cli]);
			}
		}

		// Weight updating
		double gamma_t = (gf_alpha*(gf_m + gf_t0)) /
			(gf_m + std::max(gf_t0, std::pow(static_cast<double>(si), gf_xi)));
		for (unsigned int li = 0; li < levels; ++li) {
			double th_b = -1.0/static_cast<double>(levels);
			if (li == cli)
				th_b += 1.0;

			theta[li] += gamma_t * th_b;
		}
		histogram[cli] += 1;

		if (cli == 0)	// target temperature
			si += 1;	// Take a sample, count as sweep

		// Still in burn-in phase -> skip
		if (si <= burnin_sweeps)
			continue;

		if (cli != 0)
			continue;

		if (keep_samples) {
			assert(ksi < samples.size());
			samples[ksi] = gibbs.State();
			ksi += 1;
		}

		// Add current sample of temperature one chain to marginals
		const std::vector<unsigned int>& sample = gibbs.State();
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			marginals[fi][factors[fi]->ComputeAbsoluteIndex(sample)] +=
				sample_contribution;
		}
	}
	if (keep_samples) {
		assert(ksi == samples.size());
	}
}

void SAMCInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	samples.clear();
	samples.resize(sample_count);

	PerformInference(true, sample_count);
	states = samples;
}

void SAMCInference::InitializeLadder(const FactorGraph* fg,
	unsigned int levels, double high_temp) {
	assert(high_temp > 1.0);
	assert(levels >= 2);

	// Calculate geometric temperature ladder from high_temp to 1.0.
	double alpha = std::exp(std::log(1.0 / high_temp) /
		static_cast<double>(levels - 1));
	double temp = high_temp;

	// Create ladder: TEMP[0]=1.0, TEMP[levels-1]=high_temp
	temperatures.resize(levels);
	for (int li = levels - 1; li >= 0; --li) {
		temperatures[li] = temp;
		temp *= alpha;	// Decrease temperature
	}
	temperatures[0] = 1.0;

	// Initialize temperature histogram and log partition function estimates
	histogram.resize(levels);
	std::fill(histogram.begin(), histogram.end(), 0);
	theta.resize(levels);
	std::fill(theta.begin(), theta.end(), 0.0);

}

void SAMCInference::DestroyLadder(void) {
	temperatures.clear();
}

// SUPPORT CODE
void SAMCInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<unsigned int>&
SAMCInference::TemperatureHistogram(void) const {
	return (histogram);
}

const std::vector<double>&
SAMCInference::LogPartitionEstimates(void) const {
	return (theta);
}

const std::vector<double>& SAMCInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >& SAMCInference::Marginals() const {
	return (marginals);
}

double SAMCInference::LogPartitionFunction() const {
	return (std::numeric_limits<double>::signaling_NaN());
}

double SAMCInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}


