
#include <algorithm>
#include <iostream>
#include <utility>
#include <cassert>

#include "FactorConditioningTable.h"

namespace Grante {

FactorConditioningTable::FactorConditioningTable() {
}

FactorConditioningTable::~FactorConditioningTable() {
	// Remove conditioned factor type in condfac_table!
	for (condfac_table_t::iterator cti = condfac_table.begin();
		cti != condfac_table.end(); ++cti) {
		delete (*cti);
	}
}

Factor* FactorConditioningTable::ConditionAndAddFactor(Factor* full_factor,
	const std::vector<unsigned int>& condition_var_set,
	const std::vector<unsigned int>& condition_var_state,
	const std::vector<unsigned int>& var_index) {
	// Check cardinalities
	assert(condition_var_set.size() == condition_var_state.size());
	const std::vector<unsigned int>& fac_card = full_factor->Cardinalities();
	for (unsigned int cvi = 0; cvi < condition_var_set.size(); ++cvi) {
		assert(condition_var_set[cvi] < fac_card.size());
		assert(condition_var_state[cvi] < fac_card[condition_var_set[cvi]]);
	}

	// Construct the conditioned factor type
	ConditionedFactorType* cft = AddCFT(new ConditionedFactorType(
		full_factor->Type(), condition_var_set, this));

	// Construct conditioned factor
	std::vector<double> data_dummy;
	Factor* fac = new Factor(cft, var_index, data_dummy);

	// Insert factor mapping and conditioning data (states)
	new_to_original_factormap[fac] = full_factor;
	new_to_condstate.insert(std::pair<const Factor*, std::vector<unsigned int> >
		(fac, condition_var_state));

	return (fac);
}

Factor* FactorConditioningTable::ConditionAndAddFactor(Factor* full_factor,
	const std::vector<unsigned int>& condition_var_set,
	const std::vector<double>& condition_var_expectations,
	const std::vector<unsigned int>& var_index) {
	// Construct the conditioned factor type
	ConditionedFactorType* cft = AddCFT(new ConditionedFactorType(
		full_factor->Type(), condition_var_set, this));

	// Construct conditioned factor
	std::vector<double> data_dummy;
	Factor* fac = new Factor(cft, var_index, data_dummy);

	// Insert factor mapping and conditioning data (expectations)
	new_to_original_factormap[fac] = full_factor;
	new_to_condexpect.insert(std::pair<const Factor*, std::vector<double> >
		(fac, condition_var_expectations));

	return (fac);
}

Factor* FactorConditioningTable::OriginalFactor(
	const Factor* new_factor) const {
	std::tr1::unordered_map<const Factor*, Factor*>::const_iterator
		ofi = new_to_original_factormap.find(new_factor);
	assert(ofi != new_to_original_factormap.end());

	return (ofi->second);
}

void FactorConditioningTable::UpdateConditioningInformation(
	const Factor* new_factor,
	const std::vector<unsigned int>& condition_var_state) {
	// Find the old conditioning information (state)
	new_to_condstate_t::iterator cvi = new_to_condstate.find(new_factor);
	assert(cvi != new_to_condstate.end());
	assert(condition_var_state.size() == cvi->second.size());

	// Update
	std::copy(condition_var_state.begin(), condition_var_state.end(),
		cvi->second.begin());
}

void FactorConditioningTable::UpdateConditioningInformation(
	const Factor* new_factor,
	const std::vector<double>& condition_var_expectations) {
	// Find the old conditioning information (expectation)
	new_to_condexpect_t::iterator cei = new_to_condexpect.find(new_factor);
	assert(cei != new_to_condexpect.end());
	assert(condition_var_expectations.size() == cei->second.size());

	// Update
	std::copy(condition_var_expectations.begin(),
		condition_var_expectations.end(), cei->second.begin());
}

void FactorConditioningTable::ConditionEnergies(const Factor* new_factor,
	const std::vector<double>& orig_energies,
	std::vector<double>& new_energies) const {
	// Obtain variable indices conditioned on
	const ConditionedFactorType* cft =
		dynamic_cast<const ConditionedFactorType*>(new_factor->Type());
	// TODO: this is not elegant and might be inefficient.  We should either
	// move the condition function into the ConditionedFactorType
	// TODO: can safely use static_cast here
	assert(cft != 0);
	const std::vector<unsigned int>& cv_index =
		cft->ConditionedVariableIndices();

	// Obtain states of variables conditioned on
	new_to_condstate_t::const_iterator cdi_s = new_to_condstate.find(new_factor);
	new_to_condexpect_t::const_iterator cdi_e =
		new_to_condexpect.find(new_factor);
	assert(cdi_s == new_to_condstate.end() || cdi_e == new_to_condexpect.end());
	if (cdi_s != new_to_condstate.end()) {
		// Conditioned by state
		const std::vector<unsigned int>& cv_data = cdi_s->second;
		assert(cv_index.size() == cv_data.size());

		// Condition energy table
		unsigned int nei = 0;
		const FactorType* base_ft = cft->BaseType();
		for (unsigned int oei = 0; oei < orig_energies.size(); ++oei) {
			// Test whether oei is an index that corresponds to the conditioned
			// variables
			if (IsIndexMatchingConditioning(base_ft, cv_index, cv_data, oei)
				== false)
				continue;

			// Does match -> copy energy
			assert(nei < new_energies.size());
			new_energies[nei] = orig_energies[oei];
			nei += 1;
		}
		assert(nei == new_energies.size());
	} else if (cdi_e != new_to_condexpect.end()) {
		// Conditioned by expectation
		const std::vector<double>& cv_expect = cdi_e->second;

		// Weighted summation
		std::fill(new_energies.begin(), new_energies.end(), 0.0);
		const FactorType* base_ft = cft->BaseType();
		for (unsigned int oei = 0; oei < orig_energies.size(); ++oei) {
			// Map oei into conditioned and unconditioned index parts
			unsigned int cei = IndexMapConditioned(base_ft, cv_index, oei);
			unsigned int nei = IndexMapUnconditioned(base_ft, cv_index, oei);
			assert(cei < cv_expect.size());
			assert(nei < new_energies.size());
			new_energies[nei] += cv_expect[cei] * orig_energies[oei];
		}
	} else {
		// This should not happen
		assert(0);
	}
}

void FactorConditioningTable::ExtendMarginals(const Factor* new_factor,
	const std::vector<double>& marginals,
	std::vector<double>& ext_marginals, bool replicate) const {
	// Clear extended marginals
	std::fill(ext_marginals.begin(), ext_marginals.end(), 0.0);

	// Obtain variable indices conditioned on (TODO, see above)
	const ConditionedFactorType* cft =
		dynamic_cast<const ConditionedFactorType*>(new_factor->Type());
	assert(cft != 0);
	assert(marginals.size() == cft->ProdCardinalities());
	const std::vector<unsigned int>& cv_index =
		cft->ConditionedVariableIndices();

	// Obtain states/expectations of variables conditioned on
	new_to_condstate_t::const_iterator cdi_s = new_to_condstate.find(new_factor);
	new_to_condexpect_t::const_iterator cdi_e =
		new_to_condexpect.find(new_factor);
	assert(cdi_s == new_to_condstate.end() || cdi_e == new_to_condexpect.end());
	const FactorType* base_ft = cft->BaseType();
	assert(ext_marginals.size() == base_ft->ProdCardinalities());
	if (cdi_s != new_to_condstate.end()) {
		// Conditioning on state
		const std::vector<unsigned int>& cv_data = cdi_s->second;
		assert(cv_index.size() == cv_data.size());
		assert(replicate == false);

		// Extend marginals
		unsigned int nei = 0;
		for (unsigned int oei = 0; oei < ext_marginals.size(); ++oei) {
			// Test whether oei is an index that corresponds to the conditioned
			// variables
			if (IsIndexMatchingConditioning(base_ft, cv_index, cv_data, oei) == false)
				continue;

			// Does match -> copy marginal
			assert(nei < marginals.size());
			ext_marginals[oei] = marginals[nei];
			nei += 1;
		}
		assert(nei == marginals.size());
	} else if (cdi_e != new_to_condexpect.end()) {
		// Conditioned by expectation
		const std::vector<double>& cv_expect = cdi_e->second;
		for (unsigned int oei = 0; oei < ext_marginals.size(); ++oei) {
			// Map oei into conditioned and unconditioned index parts
			unsigned int nei = IndexMapUnconditioned(base_ft, cv_index, oei);
			assert(nei < marginals.size());
			if (replicate) {
				ext_marginals[oei] = marginals[nei];
			} else {
				unsigned int cei = IndexMapConditioned(base_ft, cv_index, oei);
				assert(cei < cv_expect.size());
				ext_marginals[oei] = cv_expect[cei] * marginals[nei];
			}
		}
	} else {
		// This should not happen
		assert(0);
	}
}

void FactorConditioningTable::ProjectExtendedMarginalsCond(
	const Factor* new_factor, const std::vector<double>& ext_marginals,
	std::vector<double>& cond_var_expect) const {
	const ConditionedFactorType* cft =
		dynamic_cast<const ConditionedFactorType*>(new_factor->Type());
	const FactorType* base_ft = cft->BaseType();
	assert(cond_var_expect.size() ==
		(base_ft->ProdCardinalities() / cft->ProdCardinalities()));
	assert(ext_marginals.size() == base_ft->ProdCardinalities());
	const std::vector<unsigned int>& cv_index =
		cft->ConditionedVariableIndices();

	// Copy a subset of the full marginals
	// TODO: this should be more efficient, especially for two-factors
	std::fill(cond_var_expect.begin(), cond_var_expect.end(), 0.0);
	for (unsigned int oei = 0; oei < ext_marginals.size(); ++oei) {
		unsigned int cei = IndexMapConditioned(base_ft, cv_index, oei);
		// Will be written multiple times but all marginals should be equal
		cond_var_expect[cei] = ext_marginals[oei];
	}
}

bool FactorConditioningTable::IsIndexMatchingConditioning(
	const FactorType* base_ft, const std::vector<unsigned int>& cv_index,
	const std::vector<unsigned int>& cv_data, unsigned int oei) const {
	bool matches_conditioning = true;
	for (unsigned int cond_id = 0; cond_id < cv_index.size(); ++cond_id) {
		unsigned int oei_cond_var_state =
			base_ft->LinearIndexToVariableState(oei, cv_index[cond_id]);
		if (oei_cond_var_state != cv_data[cond_id]) {
			matches_conditioning = false;
			break;
		}
	}
	return (matches_conditioning);
}

// Map original energy index into conditioning expectation table index
unsigned int FactorConditioningTable::IndexMapConditioned(
	const FactorType* base_ft, const std::vector<unsigned int>& cv_index,
	unsigned int oei) {
	const std::vector<unsigned int>& card = base_ft->Cardinalities();
	unsigned int cei = 0;
	unsigned int prodcard = 1;
	for (unsigned int cond_id = 0; cond_id < cv_index.size(); ++cond_id) {
		unsigned int cv_var = cv_index[cond_id];
		cei += prodcard * base_ft->LinearIndexToVariableState(oei, cv_var);
		prodcard *= card[cv_var];
	}
	return (cei);
}

unsigned int FactorConditioningTable::IndexMapUnconditioned(
	const FactorType* base_ft, const std::vector<unsigned int>& cv_index,
	unsigned int oei) {
	unsigned int nei = 0;
	unsigned int prodcard = 1;
	unsigned int cond_id = 0;

	const std::vector<unsigned int>& card = base_ft->Cardinalities();
	for (unsigned int fvi = 0; fvi < card.size(); ++fvi) {
		// Conditioned -> skip
		if (cond_id < cv_index.size() && cv_index[cond_id] == fvi) {
			if (cond_id > 0) {
				assert(cv_index[cond_id - 1] < cv_index[cond_id]);
			}
			cond_id += 1;
			continue;
		}

		// Unconditioned
		nei += prodcard * base_ft->LinearIndexToVariableState(oei, fvi);
		prodcard *= card[fvi];
	}
	return (nei);
}

ConditionedFactorType* FactorConditioningTable::AddCFT(
	ConditionedFactorType* cft) {
	// Attempt to locate it in the table of conditioned factors
	condfac_table_t::const_iterator cti = condfac_table.find(cft);
	if (cti == condfac_table.end()) {
		// Unique factor type conditioning variant, add
		condfac_table.insert(cft);
	} else {
		// This exact conditioning variant already exists
		// TODO: check this actually happens
		delete cft;
		cft = *cti;
	}
	return (cft);
}

}

