
#ifndef GRANTE_NAIVEPIECEWISETRAINING_H
#define GRANTE_NAIVEPIECEWISETRAINING_H

#include "ParameterEstimationMethod.h"
#include "MaximumLikelihood.h"
#include "SubFactorGraph.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* Reference
 * 1. Charles Sutton, Andrew McCallum, "Piecewise Training for Undirected
 * Models", UAI 2005.
 */
class NaivePiecewiseTraining : public ParameterEstimationMethod {
public:
	explicit NaivePiecewiseTraining(FactorGraphModel* fg_model);
	virtual ~NaivePiecewiseTraining();

	void SetOptimizationMethod(
		MaximumLikelihood::MLEOptimizationMethod opt_method);

	// FIXME: remove
	MaximumLikelihood::MLEProblem* GetLearnProblem();

	// Decompose given factor graphs and initialize training data and
	// inference methods.
	virtual void SetupTrainingData(
		const std::vector<labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*> inference_methods);

	virtual void UpdateTrainingLabeling(
		const std::vector<labeled_instance_type>& training_update);

	virtual void AddPrior(const std::string& factor_type, Prior* prior);
	virtual double Train(double conv_tol, unsigned int max_iter = 0);

private:
	MaximumLikelihood mle;

	// Training data on subgraphs
	std::vector<labeled_instance_type> pw_training_data;
	std::vector<InferenceMethod*> pw_inference_methods;

	std::vector<SubFactorGraph*> subfgs;
};

}

#endif

