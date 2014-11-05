
#include <cassert>

#include "FactorGraphObservation.h"

namespace Grante {

FactorGraphObservation::FactorGraphObservation(
	const std::vector<unsigned int>& observed_state)
	: type(DiscreteLabelingType), observed_state(observed_state) {
}

FactorGraphObservation::FactorGraphObservation(
	const std::vector<std::vector<double> >& observed_expectation)
	: type(ExpectationType), observed_expectation(observed_expectation) {
}

FactorGraphObservation::ObservationType FactorGraphObservation::Type() const {
	return (type);
}

const std::vector<unsigned int>& FactorGraphObservation::State() const {
	assert(type == DiscreteLabelingType);
	return (observed_state);
}

const std::vector<std::vector<double> >&
FactorGraphObservation::Expectation() const {
	assert(type == ExpectationType);
	return (observed_expectation);
}

std::vector<std::vector<double> >& FactorGraphObservation::Expectation() {
	assert(type == ExpectationType);
	return (observed_expectation);
}

}

