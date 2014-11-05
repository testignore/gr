
#include <limits>
#include <tr1/unordered_map>

#include "Conditioning.h"

namespace Grante {

FactorGraph* Conditioning::ConditionFactorGraph(
	FactorConditioningTable* ftab, const FactorGraph* fg_base,
	const FactorGraphPartialObservation* pobs,
	std::vector<unsigned int>& var_new_to_orig)
{
	std::vector<unsigned int> fac_new_to_orig;
	return (ConditionFactorGraph(ftab, fg_base, pobs, var_new_to_orig,
		fac_new_to_orig));
}

FactorGraph* Conditioning::ConditionFactorGraph(
	FactorConditioningTable* ftab, const FactorGraph* fg_base,
	const FactorGraphPartialObservation* pobs,
	std::vector<unsigned int>& var_new_to_orig,
	std::vector<unsigned int>& fac_new_to_orig) {
	const std::vector<unsigned int>& condition_var_set =
		pobs->ObservedVariableSet();

	// Check variable counts
	const std::vector<unsigned int>& card = fg_base->Cardinalities();
	assert(condition_var_set.size() < card.size());
	size_t cond_var_count = card.size() - condition_var_set.size();
	std::vector<unsigned int> cond_card;	// cardinalities of conditioned fg
	cond_card.reserve(cond_var_count);
	var_new_to_orig.reserve(cond_var_count);

	std::vector<unsigned int> var_orig_to_new(card.size(),
		std::numeric_limits<unsigned int>::max());
	// For discrete observations:
	//   cond_abs_var[abs_var_idx]: index into condition_var_set/state
	std::tr1::unordered_map<unsigned int, unsigned int> cond_abs_var;

	unsigned int cvi = 0;
	for (unsigned int vi = 0; vi < card.size(); ++vi) {
		if (cvi > 0 && cvi < condition_var_set.size()) {
			assert(condition_var_set[cvi-1] < condition_var_set[cvi]);
		}
		if (cvi < condition_var_set.size() && condition_var_set[cvi] == vi) {
			// Conditioned variable, skip
			cond_abs_var.insert(std::pair<unsigned int, unsigned int>(vi, cvi));
			cvi += 1;
			continue;
		}
		// Unconditioned variable: add
		cond_card.push_back(card[vi]);
		var_orig_to_new[vi] =
			static_cast<unsigned int>(var_new_to_orig.size());
		var_new_to_orig.push_back(vi);
	}

	// Create new conditioned factor graph defined on all remaining
	// unconditioned variables
	FactorGraph* fg_cond = new FactorGraph(fg_base->Model(), cond_card);

	// Condition all factors
	const std::vector<Factor*>& facs = fg_base->Factors();
	for (unsigned int fi = 0; fi < facs.size(); ++fi) {
		Factor* fac = facs[fi];
		const std::vector<unsigned int>& fac_card = fac->Cardinalities();
		const std::vector<unsigned int>& fac_var = fac->Variables();

		// Build factor-relative conditioning set
		std::vector<unsigned int> fac_condition_var_set;
		std::vector<unsigned int> fac_condition_var_state;
		std::vector<unsigned int> var_index;
		fac_condition_var_set.reserve(fac_card.size());
		if (pobs->Type() == FactorGraphObservation::DiscreteLabelingType)
			fac_condition_var_state.reserve(fac_card.size());
		var_index.reserve(fac_card.size());

		// Partition factor variables in conditioned (fac_condition_*) and
		// unconditioned (var_index) variables
		std::tr1::unordered_map<unsigned int, unsigned int>::iterator cavi;
		for (unsigned int fvi = 0; fvi < fac_var.size(); ++fvi) {
			cavi = cond_abs_var.find(fac_var[fvi]);
			bool is_conditioned = (cavi != cond_abs_var.end());
			if (is_conditioned) {
				// Conditioned
				fac_condition_var_set.push_back(fvi);

				// If it is a discrete observation: collect state
				if (pobs->Type() ==
					FactorGraphObservation::DiscreteLabelingType)
				{
					fac_condition_var_state.push_back(
						pobs->ObservedVariableState()[cavi->second]);
				}
			} else {
				// Unconditioned
				assert(var_orig_to_new[fac_var[fvi]] !=
					std::numeric_limits<unsigned int>::max());
				var_index.push_back(var_orig_to_new[fac_var[fvi]]);
			}
		}

		// Case 1: This factor does not contain any conditioned variable
		if (fac_condition_var_set.empty()) {
			assert(var_index.size() == fac_card.size());
			assert(fac_condition_var_state.empty());

			// The factor is essentially replicated but changing the variable
			// indices.  The forward/backward map is done by means of the
			// original factor.
			Factor* fac_cond = ftab->ConditionAndAddFactor(fac,
				fac_condition_var_set, fac_condition_var_state, var_index);
			fg_cond->AddFactor(fac_cond);	// takes ownership
			fac_new_to_orig.push_back(fi);
			continue;
		}
		// Case 2: All variables of this factor are conditioned and hence the
		// factor becomes a constant and can be dropped.  This can
		// substantially reduce the size of a factor graph.
		if (fac_condition_var_set.size() == fac_card.size())
			continue;

		// Case 3: Factor contains a strict non-empty subset of conditioned
		// variables.
		// Condition this factor and add it
		assert((fac_condition_var_set.size()+var_index.size()) ==
			fac_card.size());
		Factor* fac_cond = 0;
		if (pobs->Type() == FactorGraphObservation::DiscreteLabelingType) {
			// Discrete observation
			fac_cond = ftab->ConditionAndAddFactor(fac, fac_condition_var_set,
				fac_condition_var_state, var_index);
		} else {
			// Expectation observation
			assert(pobs->Type() == FactorGraphObservation::ExpectationType);
			fac_cond = ftab->ConditionAndAddFactor(fac, fac_condition_var_set,
				pobs->ObservedMarginals(fi), var_index);
		}
		assert(fac_cond != 0);
		fac_new_to_orig.push_back(fi);
		fg_cond->AddFactor(fac_cond);	// takes ownership
	}
	return (fg_cond);
}

}

