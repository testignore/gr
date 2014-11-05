
#ifndef GRANTE_CRISSCROSS_H
#define GRANTE_CRISSCROSS_H

#include "MaximumCompositeLikelihood.h"

namespace Grante {

/* Criss cross likelihood: a composite likelihood specialized to grid graphs
 * with 4-neighborhood.  These graphs are common in computer vision
 * applications.
 *
 * The idea is to decompose the grid into a set of vertical and horizontal
 * chains over which inference is efficient.  Moreover, as it is based on
 * composite likelihood with properly conditioned components, the maximum
 * criss-cross likelihood estimator is consistent.
 */
class MaximumCrissCrossLikelihood : public MaximumCompositeLikelihood {
public:
	explicit MaximumCrissCrossLikelihood(FactorGraphModel* fg_model);
	virtual ~MaximumCrissCrossLikelihood();

	// Decompose given factor graphs and initialize training data and
	// inference methods.
	// All factor graphs must be grid graphs with 4-neighborhood.
	virtual void SetupTrainingData(
		const std::vector<labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*> inference_methods);
};

}

#endif

