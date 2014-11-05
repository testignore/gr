
#ifndef GRANTE_MCHAINGIBBSINFERENCE_H
#define GRANTE_MCHAINGIBBSINFERENCE_H

#include "FactorGraph.h"
#include "InferenceMethod.h"
#include "GibbsSampler.h"

namespace Grante {

/* Gibbs sampling inference with multiple chains and convergence diagnostics.
 *
 * References
 *
 * [Brooks1998], Stephen P. Brooks, Andrew Gelman,
 *     "General Methods for Monitoring Convergence of Iterative Simulations",
 *     Journal of Computational and Graphical Statistics,
 *     Vol. 7, No. 4, pages 434--455, December 1998.
 *
 * [Gelman1992], Andrew Gelman, Donald B. Rubin,
 *     "Inference from Iterative Simulation using Multiple Sequences",
 *     Statistical Science,
 *     Vol. 7, pages 457--511, 1992.
 */
class MultichainGibbsInference : public InferenceMethod {
public:
	explicit MultichainGibbsInference(const FactorGraph* fg);
	virtual ~MultichainGibbsInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set multi-chain Gibbs sampling parameters.
	//
	// number_of_chains: >=3, number of parallel chains used.
	// accept_psrf: >1.0, potential scale reduction factor acceptance
	//    threshold; if max_i PSRF_c <= accept_psnr, then we assume the chains
	//    have converged.
	// spacing_sweeps: number of sweeps to discard between samples.
	//    Default: 0.
	// sample_count: number of samples used to estimate marginals after
	//    convergence has been determined.  Default: 10000.
	void SetSamplingParameters(unsigned int number_of_chains,
		double accept_psrf, unsigned int spacing_sweeps, unsigned int sample_count);

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
	// Marginal distribution means and variances for M chains
	typedef std::vector<std::vector<double> > marginals_t;
	std::vector<marginals_t> chain_mean;
	std::vector<marginals_t> chain_varm;

	// Inference result: estimated marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Inference result: estimated log-partition function.
	// XXX: Not available through Gibbs sampling
	double log_z;

	// Workhorse: the actual Gibbs sampler used
	std::vector<GibbsSampler> chain_gibbs;

	// Gibbs sampling parameters
	unsigned int number_of_chains;
	double accept_psrf;
	unsigned int spacing_sweeps;
	unsigned int sample_count;

	void PerformBurninPhase();

	// Setup chain_mean and chain_varm
	void SetupChains(void);

	void UpdateMeanVariance(const GibbsSampler& gibbs,
		marginals_t& mean, marginals_t& varm, unsigned int n);
	double ComputePSRF(unsigned int n) const;
};

}

#endif

