
#ifndef GRANTE_NAIVE_MEANFIELD_H
#define GRANTE_NAIVE_MEANFIELD_H

#include <vector>

#include "FactorGraph.h"
#include "FactorGraphUtility.h"
#include "InferenceMethod.h"

namespace Grante {

/* Naive mean field approximation
 *
 * The method is described in many references, but this one is closest to this
 * implementation:
 *
 * [Nowozin2011] Sebastian Nowozin and Christoph H. Lampert, "Structured
 *    Learning and Prediction in Computer Vision", now FnT Graphics and
 *    Computer Vision, 2011.
 */
class NaiveMeanFieldInference : public InferenceMethod {
public:
	NaiveMeanFieldInference(const FactorGraph* fg);
	virtual ~NaiveMeanFieldInference();

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
	FactorGraphUtility fgu;

	// Inference result: realizable marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Inference result: lower bound on the log-partition function
	double log_z;

	// Parameters
	bool verbose;
	double conv_tol;
	unsigned int max_iter;

	// Update a single variable distribution
	double UpdateSite(std::vector<std::vector<double> >& vmarg,
		unsigned int vi) const;

	double ComputeLogPartitionFunction(
		const std::vector<std::vector<double> >& vmarg) const;
	void ProduceMarginals(const std::vector<std::vector<double> >& vmarg);
};

}

#endif

