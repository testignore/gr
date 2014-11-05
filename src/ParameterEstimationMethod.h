
#ifndef GRANTE_ESTIMATIONMETHOD_H
#define GRANTE_ESTIMATIONMETHOD_H

#include <vector>
#include <utility>
#include <map>

#include "FactorGraphModel.h"
#include "FactorGraphObservation.h"
#include "FactorGraphPartialObservation.h"
#include "FactorGraph.h"
#include "Prior.h"
#include "InferenceMethod.h"

namespace Grante {

/* Interface for all parameter estimation methods, including MAP-based
 * learning approaches.
 *
 * Given a FactorGraphModel, a set of FactorGraph objects instantiating this
 * model, and a set of full observations for these factor graphs, the method
 * provides a learned parameter vector.
 */
class ParameterEstimationMethod {
public:
	virtual ~ParameterEstimationMethod();

	// Annotated training data type: factor graph and observation label
	typedef std::pair<FactorGraph*, const FactorGraphObservation*>
		labeled_instance_type;
	typedef std::pair<FactorGraph*, const FactorGraphPartialObservation*>
		partially_labeled_instance_type;

	// Add a prior distribution over the weights associated with one factor.
	// This class will take ownership of 'prior'.
	virtual void AddPrior(const std::string& factor_type, Prior* prior);

	// Initialize training data and inference methods.
	// Both training_data and inference_methods are copied, however the
	// inference_methods objects must remain valid throughout the algorithm.
	virtual void SetupTrainingData(
		const std::vector<labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*> inference_methods);

	// Output some information about the learning problem: number of factors,
	// number of data instances, total state space size, number of parameters.
	virtual void PrintProblemStatistics() const;

	// Update the target labels/expectations.
	// The given training_update must be exactly identical to the
	// training_data, except that the target label could have changed.  That
	// is, the inference_methods provided in SetupTrainingData must still work
	// as before on the original factor graphs.
	virtual void UpdateTrainingLabeling(
		const std::vector<labeled_instance_type>& training_update);

	// Train the model (adjust parameters) to fit the training data
	// conv_tol: Convergence tolerance (method dependent).
	// max_iter: Maximum number of iterations (method dependent), where zero
	//    denotes no limit.
	//
	// Return the value of the objective function at the last iteration.  The
	// interpretation depends on the particular training method.
	virtual double Train(double conv_tol, unsigned int max_iter = 0) = 0;

protected:
	FactorGraphModel* fg_model;

	// The training data
	std::vector<labeled_instance_type> training_data;
	std::vector<InferenceMethod*> inference_methods;

	// Priors
	std::multimap<std::string, Prior*> priors;

	// Constructor
	ParameterEstimationMethod(FactorGraphModel* fg_model);
};

}

#endif

