
#include <cassert>

#include "FactorGraphStructurizer.h"
#include "MaximumTreePseudoLikelihood.h"

namespace Grante {

MaximumTreePseudoLikelihood::MaximumTreePseudoLikelihood(
	FactorGraphModel* fg_model)
	: MaximumCompositeLikelihood(fg_model) {
}

MaximumTreePseudoLikelihood::~MaximumTreePseudoLikelihood() {
}

void MaximumTreePseudoLikelihood::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<std::vector<unsigned int> >& cond_var_set,
	const std::vector<InferenceMethod*> inference_methods) {
	assert(training_data.size() > 0);
	assert(training_data.size() == cond_var_set.size());
	assert(training_data.size() == inference_methods.size());

	// Add conditioned components
	for (size_t n = 0; n < training_data.size(); ++n) {
		const FactorGraph* fg = training_data[n].first;
		const FactorGraphObservation* obs = training_data[n].second;

		assert(FactorGraphStructurizer::IsTreeStructured(fg));
		AddTrainingComponentCond(fg, obs, inference_methods[n],
			cond_var_set[n]);
	}

	// Initialize MLE training data from created components
	SetupMLETrainingData();
}

void MaximumTreePseudoLikelihood::UpdateTrainingLabeling(
	const std::vector<labeled_instance_type>& training_update) {
	assert(0);	// XXX: not supported yet
}

}

