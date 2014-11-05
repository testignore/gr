
#ifndef GRANTE_GIBBSINFERENCE_H
#define GRANTE_GIBBSINFERENCE_H

#include "FactorGraph.h"
#include "InferenceMethod.h"
#include "GibbsSampler.h"

namespace Grante {

class GibbsInference : public InferenceMethod {
public:
	explicit GibbsInference(const FactorGraph* fg);
	virtual ~GibbsInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set Gibbs sampling parameters.
	//
	// burnin_sweeps: number of sweeps to discard initially.
	//    Default: 100.
	// spacing_sweeps: number of sweeps to discard between samples.
	//    Default: 0.
	// sample_count: number of samples used to estimate marginals.
	//    Default: 10000.
	void SetSamplingParameters(unsigned int burnin_sweeps,
		unsigned int spacing_sweeps, unsigned int sample_count);

	// Perform Gibbs sampling to compute marginals.
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Approximate marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// Gibbs sampling does not support computation of the log-partition
	// function.
	// This method always returns the signaling_NaN value.
	virtual double LogPartitionFunction() const;

	// Produce approximate samples from the distribution
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// NOT IMPLEMENTED
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// Inference result: estimated marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Inference result: estimated log-partition function.
	// XXX: Not available through Gibbs sampling
	double log_z;

	// Workhorse: the actual Gibbs sampler used
	GibbsSampler gibbs;

	// Gibbs sampling parameters
	unsigned int burnin_sweeps;
	unsigned int spacing_sweeps;
	unsigned int sample_count;

	void PerformBurninPhase();
};

}

#endif

