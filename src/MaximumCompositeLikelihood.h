
#ifndef GRANTE_MAXIMUMCOMPOSITELIKELIHOOD_H
#define GRANTE_MAXIMUMCOMPOSITELIKELIHOOD_H

#include "ParameterEstimationMethod.h"
#include "MaximumLikelihood.h"
#include "FactorConditioningTable.h"
#include "FactorGraphPartialObservation.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* MCLE - Maximum Composite Likelihood.
 *
 * References
 *
 * 1. Joshua V. Dillon, Guy Lebanon, "Statistical and Computational Tradeoffs
 * in Stochastic Composite Likelihood", AISTATS 2009, (extended version at
 * http://arxiv.org/abs/1003.0691)
 *
 * 2. Bruce G. Lindsay, "Composite Likelihood Methods", Contemporary
 * Mathematics, Vol. 80, 1988.
 * http://www.stat.psu.edu/~bgl/center/tr/PUB88a.pdf
 */
class MaximumCompositeLikelihood : public ParameterEstimationMethod {
public:
	// How the factor graph is decomposed for learning.  We do not know
	// the strength of the individual factors as we have not learned
	// parameters yet.  Therefore, the choices are limited to the following:
	//   DecomposePseudolikelihood: Merely for testing, this decomposition
	//      removes every factor and yields the pseudolikelihood objective.
	//   DecomposeUniform: Use a single decomposition and try to maximize the
	//      number of factors retained in the approximation.
	//   DecomposeRandomized* (or integer >0): Use one or multiple
	//      decompositions, each obtained by approximately maximizing a random
	//      weight function on the retained factors.  This covers the original
	//      factor graph multiple times with tractable v-acyclic
	//      subgraphs.
	enum DecompositionType {
		DecomposePseudolikelihood = -1,
		DecomposeUniform = 0,
		DecomposeRandomizedOnce = 1,
		DecomposeRandomizedTwice = 2,
		// ... (integers >2)
	};

	MaximumCompositeLikelihood(FactorGraphModel* fg_model,
		int decomp = DecomposeUniform);
	virtual ~MaximumCompositeLikelihood();

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

protected:
	int decomp;	// DecompositionType

	// Add a single likelihood component (B, V\B) where B is uncond_var_set,
	// the set of variables remaining unconditioned in the component.
	virtual void AddTrainingComponentUncond(
		const FactorGraph* fg, const FactorGraphObservation* obs,
		InferenceMethod* inference_method,
		const std::vector<unsigned int>& uncond_var_set);

	// Add a single likelihood component (V\A, A), where A is cond_var_set.
	// (fg,obs,inference_method) is the factor graph, full observation and
	// inference method, respectively.  The inference method must be able to
	// do inference over the subgraph indexed by cond_var_set.
	virtual void AddTrainingComponentCond(
		const FactorGraph* fg, const FactorGraphObservation* obs,
		InferenceMethod* inference_method,
		const std::vector<unsigned int>& cond_var_set);

	// Initialize MLE training data from components
	void SetupMLETrainingData();

private:
	MaximumLikelihood mle;

	FactorConditioningTable ftab;
	// fg_orig_index: The fg_orig_index[i] contains the original training
	//    instance index for the component i.
	// fg_cc_var_label: The original connected-component variable label
	//    output from the v-acyclic decomposition.  Needed to relate the
	//    original factor graph variables to the MCLE components.
	// fg_cc_count: Number of components in the i'th factor graph
	//    decomposition.
	std::vector<unsigned int> fg_orig_index;
	std::vector<std::vector<unsigned int> > fg_cc_var_label;
	std::vector<unsigned int> fg_cc_count;

	// Component training data (decomposed original graphs)
	std::vector<labeled_instance_type> comp_training_data;
	std::vector<InferenceMethod*> comp_inference_methods;

	// Produce the partial information used for conditioning.
	FactorGraphPartialObservation* CreatePartialObservationCond(
		const FactorGraph* fg, const FactorGraphObservation* obs,
		const std::vector<unsigned int>& cond_var_set) const;

	// Produce the fully-observed information (on a subgraph) used as ground
	// truth on the conditioned factor graphs.
	FactorGraphObservation* CreatePartialObservationUncond(
		const FactorGraph* fg, const FactorGraph* fg_cond,
		const FactorGraphObservation* obs,
		const std::vector<unsigned int>& var_new_to_orig,
		const std::vector<unsigned int>& fac_new_to_orig) const;

	// cti: absolute index in comp_training_data.
	void UpdateTrainingComponentCond(
		const FactorGraph* fg, const FactorGraphObservation* obs,
		const std::vector<unsigned int>& cond_var_set,
		unsigned int cti);
};

}

#endif

