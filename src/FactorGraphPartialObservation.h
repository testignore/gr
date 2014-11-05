
#ifndef GRANTE_FACTORGRAPHPARTIALOBSERVATION_H
#define GRANTE_FACTORGRAPHPARTIALOBSERVATION_H

#include <vector>
#include <tr1/unordered_map>

#include "FactorGraphObservation.h"

namespace Grante {

/* This class is used for conditioning (and will be used in the future for
 * hidden variable models).
 */
class FactorGraphPartialObservation {
public:
	// Observation: set of discrete observations on a variable subset
	//
	// var_subset: The non-empty subset of indices of observed variables.
	// var_state: The discrete states of observed variables
	//    (var_subset.size()==var_state.size()).
	FactorGraphPartialObservation(const std::vector<unsigned int>& var_subset,
		const std::vector<unsigned int>& var_state);

	// Observation: marginals on a variable subset (var_subset).
	//
	// var_subset: The non-empty subset of indices of observed variables.
	// fac_subset: Set of all factor indices in which at least one observed
	//   and at least one unobserved variables participate.
	// observed_expectation: for each factor fac_subset[i], the
	//    observed_expectation[i] is a marginal distribution over the observed
	//    variables of that factor, in the variable ordering of that factor.
	FactorGraphPartialObservation(const std::vector<unsigned int>& var_subset,
		const std::vector<unsigned int>& fac_subset,
		const std::vector<std::vector<double> >& observed_expectations);

	// Return the type of partial observation (discrete or expectation).
	FactorGraphObservation::ObservationType Type() const;

	const std::vector<unsigned int>& ObservedVariableSet() const;

	// Obtain the state vector for observed variables.  Use only for
	// the Type()==DiscreteLabelingType case.
	const std::vector<unsigned int>& ObservedVariableState() const;

	// Obtain the observed marginals for a specific factor.
	// fi: Factor graph factor index.
	const std::vector<double>& ObservedMarginals(unsigned int fi) const;

private:
	// Type of partial observation (discrete or expectation)
	FactorGraphObservation::ObservationType type;

	// Observed variable subset
	std::vector<unsigned int> var_subset;

	// var_state[var_index] = observed discrete state
	std::vector<unsigned int> var_state;

	// obs_marg[factor_index] = partial marginal distribution of the observed
	//    subset of the variables participating in the factor.
	std::tr1::unordered_map<unsigned int, std::vector<double> > obs_marg;
};

}

#endif

