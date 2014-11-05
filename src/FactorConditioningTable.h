
#ifndef GRANTE_FACTORCONDITIONINGTABLE_H
#define GRANTE_FACTORCONDITIONINGTABLE_H

#include <vector>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <functional>

#include "Factor.h"
#include "ConditionedFactorType.h"

namespace Grante {

/* Efficient lookup table used for providing conditioned factor types
 * information about which factors are conditioned on which states.
 */
class FactorConditioningTable {
public:
	FactorConditioningTable();
	~FactorConditioningTable();

	// Produce a new factor by conditioning a full order-two or higher-order
	// factor on a subset of its adjacent variables.  A new conditioned factor
	// type is created when necessary.
	//
	// full_factor: The full factor instance in the original factor graph.
	//    This is not const because we will hand it back to the user by the
	//    OriginalFactor method.  But this object will not change the factor.
	// condition_var_set: The set of adjacent variables, in relative order of
	//    the original factor.
	// condition_var_state: Same size as condition_var_set.  Denotes the
	//    states the variables are conditioned on.
	// var_index: The unconditioned absolute variable indices that still
	//    remain unconditioned.  We must have
	//    var_index.size()+condition_var_set.size() ==
	//    full_factor->Cardinalities().size().
	Factor* ConditionAndAddFactor(Factor* full_factor,
		const std::vector<unsigned int>& condition_var_set,
		const std::vector<unsigned int>& condition_var_state,
		const std::vector<unsigned int>& var_index);
	Factor* ConditionAndAddFactor(Factor* full_factor,
		const std::vector<unsigned int>& condition_var_set,
		const std::vector<double>& condition_var_expectations,
		const std::vector<unsigned int>& var_index);

	// Return the original factor related to the conditioned factor
	// 'new_factor'.
	Factor* OriginalFactor(const Factor* new_factor) const;

	// Update conditioning information in the internal data structures.  This
	// can be used for methods that keep partial factor graphs conditioned on
	// information that is updated during the course of the algorithm.  For
	// example, mean field, expectation maximization and iterated conditional
	// modes require this functionality.
	void UpdateConditioningInformation(const Factor* new_factor,
		const std::vector<unsigned int>& condition_var_state);
	void UpdateConditioningInformation(const Factor* new_factor,
		const std::vector<double>& condition_var_expectations);

	// Translate the full unconditioned energies of the unconditioned factor
	// underlying 'new_factor' to the conditioned energy table.  For doing
	// this the conditioning states are used.
	void ConditionEnergies(const Factor* new_factor,
		const std::vector<double>& orig_energies,
		std::vector<double>& new_energies) const;

	// Extend the conditional marginals to the full marginals of the original
	// factor type using the conditioning states.
	void ExtendMarginals(const Factor* new_factor,
		const std::vector<double>& marginals,
		std::vector<double>& ext_marginals, bool replicate = false) const;

	// Project a table of full extended marginals onto the conditioned-on
	// expectations cond_var_expect.  The full marginals need not be proper
	// marginals.
	void ProjectExtendedMarginalsCond(const Factor* new_factor,
		const std::vector<double>& ext_marginals,
		std::vector<double>& cond_var_expect) const;

	// Map oei into the corresponding conditioned-on marginal table entry.
	//
	// base_ft: Original factor type.
	// cv_index: Conditioning variable set as relative index of the original
	//    factor type.
	// oei: Original factor type energy index.
	//
	// Returns the _conditioning_ expectation table index (cei) corresponding
	// to the original energy index oei.
	static unsigned int IndexMapConditioned(const FactorType* base_ft,
		const std::vector<unsigned int>& cv_index, unsigned int oei);
	// Returns the corresponding unconditioned energy index of the new
	// conditioned factor that corresponds to oei.
	static unsigned int IndexMapUnconditioned(const FactorType* base_ft,
		const std::vector<unsigned int>& cv_index, unsigned int oei);

private:
	// new_to_original_factormap[conditioned_factor] = original_factor
	std::tr1::unordered_map<const Factor*, Factor*> new_to_original_factormap;
	typedef std::tr1::unordered_map<const Factor*, std::vector<unsigned int> >
		new_to_condstate_t;
	typedef std::tr1::unordered_map<const Factor*, std::vector<double> >
		new_to_condexpect_t;
	// new_to_condstate[conditioned_factor] = conditioning states
	new_to_condstate_t new_to_condstate;
	// new_to_condexpect[conditioned_factor] = partial marginal distribution
	//    of variables conditioned on
	new_to_condexpect_t new_to_condexpect;

	// True semantic equality check for the conditioned factor type table
	struct condfac_t_eq {
		bool operator()(const ConditionedFactorType* cft1,
			const ConditionedFactorType* cft2) const {
			return (*cft1 == *cft2);
		}
	};

	// All conditioned factor types created by ConditionAndAddFactor method
	// are stored here.  In case the same FactorConditioningTable is used for
	// conditioning a set of factor graphs, no conditional factor type is
	// created twice.
	typedef std::tr1::unordered_set<ConditionedFactorType*,
		ConditionedFactorType::condfac_tp_hash, condfac_t_eq> condfac_table_t;
	condfac_table_t condfac_table;

	// Predicate: is the energy indexed by 'oei' part of the conditioned
	// factor energy table?
	//
	// base_ft: The factor type of the original, unconditioned factor.
	// cv_index: The conditioning variable indices, base_ft factor-relative.
	// cv_data: The states conditioned on.
	// oei: The index into the energy table of the original unconditioned
	//    factor.
	//
	// Return true if 'oei' appears in the conditioned table, false otherwise.
	bool IsIndexMatchingConditioning(const FactorType* base_ft,
		const std::vector<unsigned int>& cv_index,
		const std::vector<unsigned int>& cv_data, unsigned int oei) const;

	// Attempt to add a new conditioned factor type.  If it is already
	// present, the passed object is deleted.
	// The unique conditioned factor type is returned.
	ConditionedFactorType* AddCFT(ConditionedFactorType* cft);
};

}

#endif

