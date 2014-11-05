
#ifndef GRANTE_MAXIMUMTREEPSEUDOLIKELIHOOD_H
#define GRANTE_MAXIMUMTREEPSEUDOLIKELIHOOD_H

#include "MaximumCompositeLikelihood.h"

namespace Grante {

/* EXPERIMENTAL
 * Maximum Tree Pseudolikelihood Estimation
 *
 * This class is applicable to tree-structured factor graphs only.
 */
class MaximumTreePseudoLikelihood : public MaximumCompositeLikelihood {
public:
	MaximumTreePseudoLikelihood(FactorGraphModel* fg_model);
	virtual ~MaximumTreePseudoLikelihood();

	// Setup training data by conditioning on a set of leaf nodes.
	//
	// training_data: fully-observed tree-structured factor graphs.
	// cond_var_set: subset of nodes that should be conditioned on.
	// inference_methods: inference objects to perform inference on
	//    conditioned subgraphs with (typically TreeInference objects).
	virtual void SetupTrainingData(
		const std::vector<labeled_instance_type>& training_data,
		const std::vector<std::vector<unsigned int> >& cond_var_set,
		const std::vector<InferenceMethod*> inference_methods);

	// XXX: not supported yet
	virtual void UpdateTrainingLabeling(
		const std::vector<labeled_instance_type>& training_update);
};

}

#endif

