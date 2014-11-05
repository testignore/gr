
#include "StructuredLossFunction.h"

namespace Grante {

StructuredLossFunction::StructuredLossFunction(
	const FactorGraphObservation* y_truth)
	: y_truth(y_truth) {
}

StructuredLossFunction::~StructuredLossFunction() {
	delete (y_truth);
}

const FactorGraphObservation* StructuredLossFunction::Truth() const {
	return (y_truth);
}

}

