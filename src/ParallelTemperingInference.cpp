
#include <algorithm>
#include <limits>
#include <iostream>
#include <cmath>
#include <cassert>

#include "RandomSource.h"
#include "ParallelTemperingInference.h"

namespace Grante {

ParallelTemperingInference::ParallelTemperingInference(const FactorGraph* fg)
	: InferenceMethod(fg), rgen(RandomSource::GetGlobalRandomSeed()),
		randu(rgen, rdestu), levels(20), high_temp(20.0),
		swap_probability(0.5), burnin_sweeps(1000), sample_count(1000) {
}

ParallelTemperingInference::~ParallelTemperingInference() {
}

InferenceMethod* ParallelTemperingInference::Produce(
	const FactorGraph* fg) const {
	ParallelTemperingInference* pt = new ParallelTemperingInference(fg);
	pt->SetSamplingParameters(levels, high_temp, swap_probability,
		burnin_sweeps, sample_count);

	return (pt);
}

void ParallelTemperingInference::SetSamplingParameters(unsigned int levels,
	double high_temp, double swap_probability, unsigned int burnin_sweeps,
	unsigned int sample_count) {
	this->levels = levels;
	this->high_temp = high_temp;
	this->swap_probability = swap_probability;
	this->burnin_sweeps = burnin_sweeps;
	this->sample_count = sample_count;
}

void ParallelTemperingInference::PerformInference() {
	// 1. Setup marginals
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}

	// 2. Setup temperature ladder
	InitializeLadder(fg, levels, high_temp);
	accept_prob.resize(levels - 1);
	std::fill(accept_prob.begin(), accept_prob.end(), 0.0);
	std::vector<double> accept_total(levels - 1, 0.0);

	// 3. Parallel tempering
	double sample_contribution = 1.0 / static_cast<double>(sample_count);
	unsigned int si = 0;
	while (si < (burnin_sweeps + sample_count)) {
		if (randu() >= swap_probability) {
			// Parallel step
			#pragma omp parallel for
			for (int li = 0; li < static_cast<int>(levels); ++li) {
				ladder[li]->Sweep(1);
			}
		} else {
			// Swapping step
			unsigned int li = static_cast<unsigned int>(
				static_cast<double>(levels - 1) * randu());
			assert(li < (levels - 1));

			// Attempt swap between swap_li and swap_li+1
			double accept_swap_prob = std::exp(
				(ladder[li]->InverseTemperature() -
					ladder[li+1]->InverseTemperature()) *
				(fg->EvaluateEnergy(ladder[li]->State()) -
					fg->EvaluateEnergy(ladder[li+1]->State())));
			accept_swap_prob = std::min(1.0, accept_swap_prob);

			accept_total[li] += 1.0;
			if (randu() >= accept_swap_prob)
				continue;	// reject swap

			// Keep statistics of average acceptance rates for each temperature level
			accept_prob[li] += 1.0;

			// Swap has been accepted
			std::vector<unsigned int> state = ladder[li]->State();
			ladder[li]->SetState(ladder[li+1]->State());
			ladder[li+1]->SetState(state);

			// Do not count swapping steps as sweeps, i.e. continue
			continue;
		}
		si += 1;	// Take a sample, count as sweep

		// Still in burn-in phase -> skip
		if (si <= burnin_sweeps)
			continue;

		// Add current sample of temperature one chain to marginals
		const std::vector<unsigned int>& sample = ladder[0]->State();
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			marginals[fi][factors[fi]->ComputeAbsoluteIndex(sample)] +=
				sample_contribution;
		}
	}
	for (unsigned int li = 0; li < accept_prob.size(); ++li) {
		accept_prob[li] /= accept_total[li];
		std::cout << "   li " << li << ", temp "
			<< (1.0 / ladder[li]->InverseTemperature())
			<< ", accept prob " << accept_prob[li] << std::endl;
	}
}

void ParallelTemperingInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(0);
	// TODO
}

void ParallelTemperingInference::InitializeLadder(const FactorGraph* fg,
	unsigned int levels, double high_temp) {
	assert(high_temp > 1.0);
	assert(levels >= 2);

	// Calculate geometric temperature ladder from high_temp to 1.0.
	double alpha = std::exp(std::log(1.0 / high_temp) /
		static_cast<double>(levels - 1));
	double temp = high_temp;

	// Create ladder: TEMP[0]=1.0, TEMP[levels-1]=high_temp
	ladder.resize(levels);
	for (int li = levels - 1; li >= 0; --li) {
		ladder[li] = new GibbsSampler(fg);
		ladder[li]->SetInverseTemperature(1.0 / temp);
		ladder[li]->SetStateUniformRandom();

		temp *= alpha;	// Decrease temperature
	}
}

void ParallelTemperingInference::DestroyLadder(void) {
	for (unsigned int li = 0; li < ladder.size(); ++li)
		delete (ladder[li]);

	ladder.clear();
}

// SUPPORT CODE
void ParallelTemperingInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>&
ParallelTemperingInference::AcceptanceProbabilities(void) const {
	return (accept_prob);
}

const std::vector<double>& ParallelTemperingInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >&
ParallelTemperingInference::Marginals() const {
	return (marginals);
}

double ParallelTemperingInference::LogPartitionFunction() const {
	return (std::numeric_limits<double>::signaling_NaN());
}

double ParallelTemperingInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}

