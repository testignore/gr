
#include <vector>
#include <cassert>

#include "StructuredHammingLoss.h"

namespace Grante {

StructuredHammingLoss::StructuredHammingLoss(
	const FactorGraphObservation* y_truth,
	const std::vector<double>& penalty_weights)
	: StructuredLossFunction(y_truth), penalty_weights(penalty_weights) {
	assert(y_truth->Type() == FactorGraphObservation::DiscreteLabelingType);
	assert(y_truth->State().size() == penalty_weights.size());
}

StructuredHammingLoss::StructuredHammingLoss(
	const FactorGraphObservation* y_truth)
	: StructuredLossFunction(y_truth) {
	assert(y_truth->Type() == FactorGraphObservation::DiscreteLabelingType);

	size_t var_count = y_truth->State().size();
	penalty_weights.resize(var_count, 1.0);
}

StructuredHammingLoss::~StructuredHammingLoss() {
}

double StructuredHammingLoss::Eval(
	const std::vector<unsigned int>& y1_state) const {
	const std::vector<unsigned int>& y_truth_state = y_truth->State();
	assert(y1_state.size() == y_truth_state.size());

	double loss = 0.0;
	for (unsigned int n = 0; n < y1_state.size(); ++n) {
		if (y1_state[n] == y_truth_state[n])
			continue;
		loss += penalty_weights[n];
	}
	return (loss);
}

void StructuredHammingLoss::PerformLossAugmentation(FactorGraph* fg,
	double scale) const {
	const std::vector<unsigned int>& y_truth_state = y_truth->State();
	size_t var_count = y_truth_state.size();
	std::vector<int> var_is_done(var_count, 0);

	// Loss augment the first factor that contains the variable.  As there
	// must be at least one factor for each variable, this augments all
	// variables.
	const std::vector<Factor*>& factors = fg->Factors();
	for (std::vector<Factor*>::const_iterator faci = factors.begin();
		faci != factors.end(); ++faci) {
		const std::vector<unsigned int>& fac_vars = (*faci)->Variables();
		for (size_t vi = 0; vi < fac_vars.size(); ++vi) {
			unsigned int cur_var = fac_vars[vi];
			assert(cur_var < var_count);
			if (var_is_done[cur_var] != 0)
				continue;

			// Augment factor energies
			std::vector<double>& fac_energies = (*faci)->Energies();
			for (size_t ei = 0; ei < fac_energies.size(); ++ei) {
				if ((*faci)->ComputeVariableState(ei, vi) ==
					y_truth_state[cur_var]) {
					continue;	// no penalty for the true state
				}
				fac_energies[ei] += scale * penalty_weights[cur_var];
			}
			var_is_done[cur_var] = 1;
		}
	}
	for (size_t vi = 0; vi < var_count; ++vi) {
		assert(var_is_done[vi] != 0);
	}
}

}

