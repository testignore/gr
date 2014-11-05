
#ifndef GRANTE_INFERENCEMETHOD_H
#define GRANTE_INFERENCEMETHOD_H

#include <vector>

#include "FactorGraph.h"

namespace Grante {

/* Probabilistic inference method: exact and approximate methods.
 * Allows the computation of marginals for a given model.
 */
class InferenceMethod {
public:
	// Every inference method must accept a null pointer (0) as constructor
	// argument.  Then, no inference can be performed but the Produce method
	// can act as a factory.
	explicit InferenceMethod(const FactorGraph* fg);
	virtual ~InferenceMethod();

	// Simple factory-like production of a new object of the same kind.
	// Note: this method might set additional parameters of the inference
	// method.  If it does, these parameters should be set in the same way as
	// on the object it is invoked on.
	virtual InferenceMethod* Produce(const FactorGraph* fg) const = 0;

	// Perform exact sum-product inference on the current factor graph
	// energies
	virtual void PerformInference() = 0;

	// After the inference results have been used this method should be called
	// to free data structures.  This does not slow down the next call to
	// PerformInference.
	virtual void ClearInferenceResult() = 0;

	// Return the marginal distribution for the given factor index.
	// factor_id: the index into the fg->Factors() array.
	//
	// The marginal is ordered in (y_1,\dots,y_k) with the
	// lowest-index-runs-fast ordering.
	virtual const std::vector<double>& Marginal(
		unsigned int factor_id) const = 0;
	virtual const std::vector<std::vector<double> >& Marginals() const = 0;

	// Return the log-partition function of the distribution
	virtual double LogPartitionFunction() const = 0;

	// Obtain the given number of exact or approximate samples from the
	// distribution specified by the factor graph.
	//
	// states: [i] contains the i'th sample, where states[i].size() is the
	//    number of variables.
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count) = 0;

	// Obtain an exact or approximate minimum energy state for the current
	// factor graph energies.
	//
	// Return the energy value of the state.  The state is returned in
	// 'state', which will be initialized to have the correct length.
	virtual double MinimizeEnergy(std::vector<unsigned int>& state) = 0;

	// For classes supporting probabilistic inference, the entropy of the
	// distribution is obtained as
	//   H(p) = -\sum_y p(y) log p(y) = log Z + \expects_{y \sim p(y)}[E(y)].
	//
	// PerformInference() must have been called prior to calling this method.
	virtual double Entropy() const;

protected:
	const FactorGraph* fg;
};

}

#endif

