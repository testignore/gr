
#include <cassert>

#include <boost/timer.hpp>

#include "FactorGraphStructurizer.h"
#include "MaximumCrissCrossLikelihood.h"

namespace Grante {

MaximumCrissCrossLikelihood::MaximumCrissCrossLikelihood(
	FactorGraphModel* fg_model)
	: MaximumCompositeLikelihood(fg_model) {
}

MaximumCrissCrossLikelihood::~MaximumCrissCrossLikelihood() {
}

void MaximumCrissCrossLikelihood::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*> inference_methods) {
	// For each factor graph, check that it is a grid graph and decompose it
	// into columns and rows
	size_t comp_count = 0;
	boost::timer decomp_timer;
	int training_data_size = static_cast<int>(training_data.size());
	for (int n = 0; n < training_data_size; ++n) {
		// Get factor graph and full observation
		FactorGraph* fg = training_data[n].first;
		const FactorGraphObservation* obs = training_data[n].second;

		// Check that we can recognize it as a grid graph
		std::vector<std::vector<unsigned int> > var_rows;
		std::vector<std::vector<unsigned int> > var_cols;
		bool is_grid =
			Grante::FactorGraphStructurizer::IsOrderedPairwiseGridStructured(
				fg, var_rows, var_cols);
		assert(is_grid);

		// Add rows
		assert(var_rows.size() > 0);
		for (unsigned int ri = 0; ri < var_rows.size(); ++ri) {
			AddTrainingComponentUncond(fg, obs, inference_methods[n],
				var_rows[ri]);
		}
		// and add columns
		assert(var_cols.size() > 0);
		for (unsigned int ci = 0; ci < var_cols.size(); ++ci) {
			AddTrainingComponentUncond(fg, obs, inference_methods[n],
				var_cols[ci]);
		}
		comp_count += var_rows.size() + var_cols.size();
	}
	std::cout << "MXXL, decomposed " << training_data.size() << " instances "
		<< "into " << comp_count << " instances in "
		<< decomp_timer.elapsed() << "s." << std::endl;

	SetupMLETrainingData();
}

}

