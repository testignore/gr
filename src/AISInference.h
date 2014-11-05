
#ifndef GRANTE_AISINFERENCE_H
#define GRANTE_AISINFERENCE_H

#include "FactorGraph.h"
#include "InferenceMethod.h"
#include "GibbsSampler.h"

namespace Grante {

/* References
 * Neal, "Annealed Importance Sampling", TechReport 1998.
 * Salakhutdinov, Murray, "On the Quantitative Analysis of Deep Belief
 * Networks", ICML 2008.
 */
class AISInference : public InferenceMethod {
public:
	explicit AISInference(const FactorGraph* fg);
	virtual ~AISInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set annealed importance sampling parameters.
	//
	// anneal_k: total number of annealing distributions, >=2.
	//    The default is 80 and in general the approximation improves the
	//    larger the number of intermediate distributions.
	// gibbs_sweeps: how many Gibbs sweeps to use to produce samples from the
	//    intermediate distributions, >=1.
	// sample_count: how many samples to use in total, >=1.
	void SetSamplingParameters(unsigned int anneal_k,
		unsigned int gibbs_sweeps, unsigned int sample_count);

	// Perform AIS to obtain approximate marginals and log-partition function
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Approximate marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// Return the log partition function.
	virtual double LogPartitionFunction() const;

	// NOT SUPPORTED, this is an importance sampling method that can only
	// approximate expectations/marginals, not produce independent samples
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// NOT SUPPORTED
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// Inference result: estimated marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Inference result: estimated log-partition function.
	double log_z;

	// Workhorse: the actual Gibbs sampler used
	GibbsSampler gibbs;

	// Gibbs sampling parameters
	unsigned int K;
	unsigned int gibbs_sweeps;
	unsigned int sample_count;
};

}

#endif

