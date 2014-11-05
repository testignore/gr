
#ifndef GRANTE_EXPECTATION_MAXIMIZATION_H
#define GRANTE_EXPECTATION_MAXIMIZATION_H

#include <vector>
#include <map>
#include <utility>
#include <tr1/unordered_map>

#include "FactorGraphModel.h"
#include "FactorGraphPartialObservation.h"
#include "FactorConditioningTable.h"
#include "ParameterEstimationMethod.h"
#include "InferenceMethod.h"
#include "Prior.h"

namespace Grante {

// TODO: at some point, factor the interface out to a
// PartialObservedParameterEstimationMethod
class ExpectationMaximization {
public:
	// model: The factor graph model whose parameters will be estimated.
	// parest_method: The parameter learning method used for carrying out
	//    the fully observed parameter maximization.  It must be a freshly
	//    instantiated object, we will call AddPrior and SetupTrainingData
	//    accordingly.  Note: this object takes ownership of parest_method and
	//    deletes it properly.
	explicit ExpectationMaximization(FactorGraphModel* model,
		ParameterEstimationMethod* parest_method);

	~ExpectationMaximization();

	virtual void AddPrior(const std::string& factor_type, Prior* prior);

	typedef ParameterEstimationMethod::partially_labeled_instance_type
		partially_labeled_instance_type;

	// Setup the partially observed training data.
	//
	// training_data: A set of factor graph instances.
	// hidden_inference_methods: The probabilistic inference methods used to
	//    compute expectations for all cross-factors involving hidden
	//    variables.  This object does not take ownership but assumes the
	//    objects stay valid throughout its lifetime.
	// observed_inference_methods: The inference method used for parameter
	//    estimation on the full the model.  The object does not take
	//    ownership but assumes the objects stay valid throughout its
	//    lifetime.
	//
	// (optional)
	// parest_inference_methods: The inference methods passed to the M-step
	//    fully-observed parameter estimation problem.  If not supplied, then
	//    the observed_inference_methods are used for this purpose as well.
	void SetupTrainingData(
		const std::vector<partially_labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*>& hidden_inference_methods,
		const std::vector<InferenceMethod*>& observed_inference_methods);
	void SetupTrainingData(
		const std::vector<partially_labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*>& hidden_inference_methods,
		const std::vector<InferenceMethod*>& observed_inference_methods,
		const std::vector<InferenceMethod*>& parest_inference_methods);

	// Train the model iteratively using EM.
	//
	// conv_tol: Convergence tolerance of EM objective.
	// max_iter: Maximum number of EM iterations, where zero denotes no limit.
	// parest_conv_tol: Convergence tolerance for the fully observed parameter
	//    estimation method.
	// parest_max_iter: Maximum number of inner iterations per EM iteration.
	//    This can be smaller than for usual maximum likelihood estimation.
	void Train(double conv_tol, unsigned int max_iter,
		double parest_conv_tol, unsigned int parest_max_iter);

private:
	FactorGraphModel* fg_model;
	ParameterEstimationMethod* parest_method;

	std::vector<partially_labeled_instance_type> training_data;

	// hidden_inference_methods: Used to compute expectations for
	//    hidden-hidden and hidden-observed factors,
	// observed_inference_methods: Used for computing the EM objective,
	// parest_inference_methods: Not used by EM but passed to parameter
	//    estimation method as argument.
	std::vector<InferenceMethod*> hidden_inference_methods;
	std::vector<InferenceMethod*> observed_inference_methods;
	std::vector<InferenceMethod*> parest_inference_methods;

	FactorConditioningTable ftab;
	std::multimap<std::string, Prior*> priors;

	// E-data
	std::vector<FactorGraph*> E_fg;
	std::vector<InferenceMethod*> E_inf;

	// M-data
	std::vector<ParameterEstimationMethod::labeled_instance_type>
		fg_m_training_data;

	// fi_to_efi[fg_n] is a map from the factor indices in the M-graph to the
	// factor index in the E-graph.  This map holds only the factor indices of
	// cross factors.
	typedef std::tr1::unordered_map<unsigned int, unsigned int> fi_to_efi_t;
	std::vector<fi_to_efi_t> fi_to_efi;
	typedef std::tr1::unordered_set<unsigned int> hidden_efi_t;
	std::vector<hidden_efi_t> hidden_efi;

	// E-step: compute expectations of hidden variables under current
	// parameters.
	// Return the sum of log-partition functions of the conditioned models,
	// A_y(theta) in Wainwright (6.9).  See also (3.45) in Wainwright.
	double ComputeHiddenVariableExpectations();
	double ComputeHiddenVariableEntropies();

	// Update the target label used during the M-step.
	// Return \sum_n -E_n(M_expects)
	double UpdateMExpectationTargets();

	// M-step: compute maximum likelihood parameters under current hidden
	// expectations.
	//
	// Return parameter estimation objective (usually negative log-likelihood
	// or a composite likelihood approximation).
	double EstimateParameters(double conv_tol, unsigned int max_iter);

	// Compute full model log-partition functions:
	//    sum_{n=1}^N log Z(x_n,w).
	double ComputeLogZ();
	// Compute the prior influence log p(w).
	double ComputeLogP();

	// Partition all factors in a training instance into three classes,
	//    1. fg_crossfactors, factors that involve both observed and
	//       unobserved variables,
	//    2. fg_hiddenfactors, factors that involve only unobserved variables,
	//    3. Fully observed factors.
	//
	// instance: The factor graph and partial observation,
	// fg_crossfactors: the cross factors, fg_crossfactors is the ordered
	//    list of factor indices for the factor graph.
	// fg_hiddenfactors: the factors involving only hidden variables, same
	//    indexing as fg_crossfactors.
	void PartitionFactors(const partially_labeled_instance_type& instance,
		std::vector<unsigned int>& fg_obsfactors,
		std::vector<unsigned int>& fg_crossfactors,
		std::vector<unsigned int>& fg_hiddenfactors) const;

	void ComputeFullState(const partially_labeled_instance_type& instance,
		std::vector<unsigned int>& full_state) const;
};

}

#endif

