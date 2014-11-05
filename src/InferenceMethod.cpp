
#include <cassert>

#include "InferenceMethod.h"

namespace Grante {

InferenceMethod::InferenceMethod(const FactorGraph* fg)
	: fg(fg) {
}

InferenceMethod::~InferenceMethod() {
}

double InferenceMethod::Entropy() const {
	assert(Marginals().size() == fg->Factors().size());
	return (LogPartitionFunction() + fg->EvaluateEnergy(Marginals()));
}

}

