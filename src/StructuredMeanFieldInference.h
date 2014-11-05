
#ifndef GRANTE_STRUCTURED_MEANFIELD_H
#define GRANTE_STRUCTURED_MEANFIELD_H

#include <tr1/unordered_set>
#include <tr1/unordered_map>

#include "FactorGraph.h"
#include "FactorConditioningTable.h"
#include "TreeInference.h"
#include "InferenceMethod.h"

namespace Grante {

/* v-acyclic structured mean field
 *
 * Relevant references
 * [Bouchard-Cote2009] Alexandre Bouchard-Cote and Michael I. Jordan,
 *    "Optimization of Structured Mean Field Objectives", UAI 2009.
 * [Wainwright2008] Martin J. Wainwright and Michael I. Jordan,
 *    "Graphical Models, Exponential Families, and Variational Inference",
 *    chapter 5.
 * [Xing2003] Eric P. Xing, Michael I. Jordan, and Stuart Russell,
 *    "A generalized mean field algorithm for variational inference in
 *    exponential families", UAI 2003.
 */
class StructuredMeanFieldInference : public InferenceMethod {
public:
	enum DecompositionType {
		UniformFactorWeights = 0,
		TotalCorrelationWeights,
	};

	// fg: Factor graph to perform inference on.
	// fcond_tab: Table to manage conditioned factor types.
	// decomp_type: What criterion to maximize.
	StructuredMeanFieldInference(const FactorGraph* fg,
		FactorConditioningTable* fcond_tab,
		DecompositionType decomp_type =
			TotalCorrelationWeights);

	// factor_is_removed: custom user-specified decomposition.  By removing
	//    all factors fi that have factor_is_removed[fi] true the resulting
	//    factor graph should become v-acyclic.  It is the users duty to
	//    ensure this.
	StructuredMeanFieldInference(const FactorGraph* fg,
		FactorConditioningTable* fcond_tab,
		const std::vector<bool>& factor_is_removed);

	virtual ~StructuredMeanFieldInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set parameters.
	//
	// verbose: If true, output.  Default: true,
	// conv_tol: Convergence tolerance wrt change in log_z.  Default: 1.0e-6,
	// max_iter: Maximum number of mean field block-coordinate ascent
	//    directions.  Use zero for no limit.  Default: 50.
	void SetParameters(bool verbose, double conv_tol,
		unsigned int max_iter);

	// Perform block-coordinate mean field optimization to compute realizable
	// marginals and bound on logZ
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Approximate but realizable marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// Return a lower bound on the log-partition function
	virtual double LogPartitionFunction() const;

	// NOT IMPLEMENTED
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// NOT IMPLEMENTED
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// The condition-manager object (can be shared among multiple inference
	// objects).
	FactorConditioningTable* fcond_tab;

	// Mean field approximation component factor graphs
	std::vector<FactorGraph*> mf_comp;
	std::vector<TreeInference*> mf_comp_inf;
	// fac_to_fi[orig_factor] is the factor index in the original factor graph
	std::tr1::unordered_map<Factor*, unsigned int> fac_to_fi;
	// For any cross-conditioned factor cfac we can locate its component as
	// cfac_to_mfi[cfac] and its factor index in mf_comp[cfac_to_mfi[cfac]] as
	// cfac_to_cfi[cfac].
	std::tr1::unordered_map<Factor*, unsigned int> cfac_to_mfi;
	std::tr1::unordered_map<Factor*, unsigned int> cfac_to_cfi;
	// fi_to_condfac[orig_fi] is the set of conditioned factors derived from
	// orig_fi.
	std::tr1::unordered_map<unsigned int, std::tr1::unordered_set<Factor*> >
		fi_to_condfac;

	std::tr1::unordered_set<Factor*> meanfield_factor_set;

	// Inference result: realizable marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Inference result: lower bound on the log-partition function
	double log_z;

	// Parameters
	bool verbose;
	double conv_tol;
	unsigned int max_iter;

	// Initialize internal data structures using a v-acyclic decomposition.
	// factor_is_removed[fi] is true if the factor is not retained.  If all
	// elements are set to true, then this corresponds to naive mean field.
	void InitializeVAC(const std::vector<bool>& factor_is_removed);

	// Update the energies of the component 'mfi' based on the inference
	// result of all its neighboring components.
	void UpdateComponentEnergies(unsigned int mfi);

	// Produce the joint marginals from component marginals
	void ProduceMarginals();
	double ComputeLogPartitionFunction();
};

}

#endif

