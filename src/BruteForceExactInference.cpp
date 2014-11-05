
#include <algorithm>
#include <limits>
#include <cassert>

#include "Factor.h"
#include "LogSumExp.h"
#include "BruteForceExactInference.h"

namespace Grante {

BruteForceExactInference::BruteForceExactInference(const FactorGraph* fg)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::signaling_NaN())
{
}

BruteForceExactInference::~BruteForceExactInference() {
}

InferenceMethod* BruteForceExactInference::Produce(const FactorGraph* fg) const
{
	return (new BruteForceExactInference(fg));
}

void BruteForceExactInference::PerformInference() {
	// 1st pass: compute log_z
	std::vector<double> eval(StateCount());
	std::vector<unsigned int> cur_state;
	InitializeState(cur_state);
	unsigned int si = 0;
	do {
		eval[si] = -fg->EvaluateEnergy(cur_state);
		si += 1;
	} while (AdvanceState(cur_state));
	log_z = LogSumExp::Compute(eval);

	// Allocate marginals
	const std::vector<Factor*>& factors = fg->Factors();
	size_t fac_count = factors.size();
	marginals.resize(fac_count);
	for (size_t fi = 0; fi < fac_count; ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}

	// 2nd pass: compute expectations
	InitializeState(cur_state);
	si = 0;
	do {
		for (size_t fi = 0; fi < fac_count; ++fi) {
			unsigned int ei = factors[fi]->Type()->ComputeAbsoluteIndex(
				factors[fi], cur_state);
			marginals[fi][ei] += std::exp(eval[si] - log_z);
		}
		si += 1;
	} while (AdvanceState(cur_state));
}

void BruteForceExactInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& BruteForceExactInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >&
BruteForceExactInference::Marginals() const {
	return (marginals);
}

double BruteForceExactInference::LogPartitionFunction() const {
	return (log_z);
}

// TODO: not implemented yet
void BruteForceExactInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(0);
}

double BruteForceExactInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	double best_energy = std::numeric_limits<double>::infinity();
	std::vector<unsigned int> cur_state;
	InitializeState(cur_state);
	do {
		double cur_energy = fg->EvaluateEnergy(cur_state);
		if (cur_energy < best_energy) {
			best_energy = cur_energy;
			state = cur_state;
		}
	} while (AdvanceState(cur_state));

	return (best_energy);
}

unsigned int BruteForceExactInference::StateCount() const {
	unsigned int scount = 1;
	const std::vector<unsigned int>& card = fg->Cardinalities();
	for (unsigned int vi = 0; vi < card.size(); ++vi)
		scount *= card[vi];

	return (scount);
}

void BruteForceExactInference::InitializeState(
	std::vector<unsigned int>& state) {
	state.resize(fg->Cardinalities().size());
	std::fill(state.begin(), state.end(), 0);
}

bool BruteForceExactInference::AdvanceState(
	std::vector<unsigned int>& state) {
	const std::vector<unsigned int>& card = fg->Cardinalities();
	assert(state.size() == card.size());
	for (unsigned int si = 0; si < card.size(); ++si) {
		// Increase possible?
		if (state[si] < (card[si] - 1)) {
			state[si] += 1;
			return (true);
		}
		// Last state reached
		if (si == (card.size() - 1))
			return (false);

		state[si] = 0;
	}
	return (false);
}

}

