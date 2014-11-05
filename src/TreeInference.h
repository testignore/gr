
#ifndef GRANTE_TREEINFERENCE_H
#define GRANTE_TREEINFERENCE_H

#include <vector>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <set>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "InferenceMethod.h"
#include "FactorGraphStructurizer.h"

namespace Grante {

/* Perform probabilistic inference on a tree-structured factorgraph.
 *
 * This class is fixed to a single factor graph, whose structure must remain
 * unchanged during the lifetime of this object.  (Some precomputations for
 * the graph structure take place once.)  The energies and data values of the
 * factor graph are allowed to change, but they invalidate the previous
 * inference result.
 */
class TreeInference : public InferenceMethod {
public:
	// The factorgraph must be forest structured (one or more trees)
	explicit TreeInference(const FactorGraph* fg);
	virtual ~TreeInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Perform exact sum-product inference on the current factor graph
	// energies
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Return the marginal distribution for the given factor index.
	// factor_id: the index into the fg->Factors() array.
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// Return the log-partition function of the distribution
	virtual double LogPartitionFunction() const;

	// Return H(p), the entropy, measured in nats.  PerformInference() must be
	// called prior to calling this function.
	virtual double Entropy() const;

	// Obtain the given number of exact samples from the distribution
	// specified by the tree-structured factor graph.  The complexity is
	//    O(sample_count T),
	// where T is the sum-product complexity for a tree graph.
	//
	// states: [i] contains the i'th sample, where states[i].size() is the
	//    number of variables.
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// Exact max-sum energy minimization for tree-structured factor graphs.
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// Inference result 1: marginal distributions for all factors
	std::vector<std::vector<double> > marginals;
	// Inference result 2: log-partition function
	double log_z;

	// Leaf-to-root tree order
	std::vector<FactorGraphStructurizer::OrderStep> leaf_to_root;
	std::tr1::unordered_set<unsigned int> tree_roots;

	// Leaf-to-root variable to 'all messages directed to this variable'
	// lookup.
	// What is stored: [var_index] = msg_indices for all incoming messages
	std::tr1::unordered_map<unsigned int, std::set<unsigned int> >
		ltr_msg_for_var;
	// What is stored: [var_index] = msg_index for the 'to-root' message.
	std::tr1::unordered_map<unsigned int, unsigned int>
		ltr_var_toroot;
	// What is stored: [factor_index] = msg_index for the 'to-root'
	// factor-to-variable message.
	std::tr1::unordered_map<unsigned int, unsigned int>
		ltr_factor_toroot;

	// Leaf-to-root and root-to-leaf passes in message passing terminology.
	// min_sum: If true, pass min-sum messages (for energy minimization).
	void PassLeafToRoot(std::vector<std::vector<double> >& msg,
		bool min_sum = false);
	void PassRootToLeaf(std::vector<std::vector<double> >& msg,
		std::vector<std::vector<double> >& msg_rev);
	void PassRootToLeaf(std::vector<std::vector<double> >& msg,
		std::vector<std::vector<double> >& msg_rev,
		std::vector<unsigned int>& sample, bool min_sum = false);

	// Utility functions to sample or maximize explicitly from an unnormalized
	// distribution.
	unsigned int SampleConditionalUnnormalized(
		const std::vector<double>& cond_unnorm) const;
	unsigned int MaximizeConditionalUnnormalized(
		const std::vector<double>& cond_unnorm) const;

	// Random number generation, for the sampler
	boost::mt19937 rgen;
	boost::uniform_real<double> rdestu;	// range [0,1]
	mutable boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu;
};

}

#endif

