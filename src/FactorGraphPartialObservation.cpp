
#include <cassert>

#include "FactorGraphPartialObservation.h"

namespace Grante {

FactorGraphPartialObservation::FactorGraphPartialObservation(
	const std::vector<unsigned int>& var_subset,
	const std::vector<unsigned int>& var_state)
	: type(FactorGraphObservation::DiscreteLabelingType),
	var_subset(var_subset), var_state(var_state) {
	assert(var_subset.size() == var_state.size());
}

FactorGraphPartialObservation::FactorGraphPartialObservation(
	const std::vector<unsigned int>& var_subset,
	const std::vector<unsigned int>& fac_subset,
	const std::vector<std::vector<double> >& observed_expectations)
	: type(FactorGraphObservation::ExpectationType), var_subset(var_subset) {
	assert(fac_subset.size() == observed_expectations.size());
	for (unsigned int n = 0; n < fac_subset.size(); ++n) {
		assert(obs_marg.find(fac_subset[n]) == obs_marg.end());
		obs_marg[fac_subset[n]] = observed_expectations[n];
	}
}

FactorGraphObservation::ObservationType
FactorGraphPartialObservation::Type() const {
	return (type);
}

const std::vector<unsigned int>&
FactorGraphPartialObservation::ObservedVariableSet() const {
	return (var_subset);
}

const std::vector<unsigned int>&
FactorGraphPartialObservation::ObservedVariableState() const {
	assert(type == FactorGraphObservation::DiscreteLabelingType);
	return (var_state);
}

const std::vector<double>&
FactorGraphPartialObservation::ObservedMarginals(unsigned int fi) const {
	std::tr1::unordered_map<unsigned int, std::vector<double> >::const_iterator
		omi = obs_marg.find(fi);
	assert(omi != obs_marg.end());
	return (omi->second);
}

}

