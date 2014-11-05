
#include <cassert>
#include <cmath>

#include "FactorGraphUtility.h"

namespace Grante {

FactorGraphUtility::FactorGraphUtility(const FactorGraph* fg)
	: fg(fg) {
	// Precompute var_to_factorset
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const Factor* factor = factors[fi];
		const std::vector<unsigned int>& vars = factor->Variables();
		for (std::vector<unsigned int>::const_iterator vi = vars.begin();
			vi != vars.end(); ++vi) {
			var_to_factorset[*vi].insert(fi);
		}
	}
}

double FactorGraphUtility::ComputeConditionalSiteDistribution(
	std::vector<unsigned int>& test_state,
	unsigned int var_index, std::vector<double>& cond_dist_unnorm,
	double temp) const {
	unsigned int var_card = fg->Cardinalities()[var_index];
	cond_dist_unnorm.resize(var_card);
	std::fill(cond_dist_unnorm.begin(), cond_dist_unnorm.end(), 0.0);

	const std::set<unsigned int>& factors_b = AdjacentFactors(var_index);
	const std::vector<Factor*>& factors = fg->Factors();
	unsigned int old_var_state = test_state[var_index];
	for (unsigned int var_state = 0; var_state < var_card; ++var_state) {
		test_state[var_index] = var_state;

		for (std::set<unsigned int>::const_iterator fbi = factors_b.begin();
			fbi != factors_b.end(); ++fbi) {
			// Factor information
			const Factor* factor = factors[*fbi];

			// Add factor energy (TODO: this could be made more efficient)
			cond_dist_unnorm[var_state] +=
				temp*factor->EvaluateEnergy(test_state);
		}
	}
	test_state[var_index] = old_var_state;

	// Conditional distribution
	double Z = 0.0;
	for (unsigned int vi = 0; vi < var_card; ++vi) {
		cond_dist_unnorm[vi] = std::exp(-cond_dist_unnorm[vi]);	// exp(-E)
		Z += cond_dist_unnorm[vi];
	}

	return (Z);
}

double FactorGraphUtility::ComputeEnergyChange(
	std::vector<unsigned int>& state, unsigned int var_index,
	unsigned int old_state, unsigned int new_state) const {
	assert(state.size() == fg->Cardinalities().size());
	assert(var_index < state.size());
	assert(old_state < fg->Cardinalities()[var_index]);
	assert(new_state < fg->Cardinalities()[var_index]);

	unsigned int save_state = state[var_index];

	// Compute energy difference
	double delta = 0.0;
	const std::set<unsigned int>& factors_b = AdjacentFactors(var_index);
	const std::vector<Factor*>& factors = fg->Factors();

	for (std::set<unsigned int>::const_iterator fbi = factors_b.begin();
		fbi != factors_b.end(); ++fbi) {
		const Factor* factor = factors[*fbi];

		state[var_index] = new_state;
		delta += factor->EvaluateEnergy(state);
		state[var_index] = old_state;
		delta -= factor->EvaluateEnergy(state);
	}
	state[var_index] = save_state;

	return (delta);
}

const std::set<unsigned int>& FactorGraphUtility::AdjacentFactors(
	unsigned int var_index) const {
	std::tr1::unordered_map<unsigned int,
		std::set<unsigned int> >::const_iterator vfi =
		var_to_factorset.find(var_index);
	assert(vfi != var_to_factorset.end());

	return (vfi->second);
}

}

