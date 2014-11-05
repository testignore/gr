
#ifndef GRANTE_SWENDSENWANGINFERENCE_H
#define GRANTE_SWENDSENWANGINFERENCE_H

#include "FactorGraph.h"
#include "InferenceMethod.h"
#include "SwendsenWangSampler.h"

namespace Grante {

/* Swendsen-Wang inference class, using the SwendsenWangSampler class to do
 * the heavy lifting.
 *
 * This inference method works only for models with variables having the same
 * number of states.  Furthermore, only unary and pairwise interactions are
 * allowed.
 */
class SwendsenWangInference : public InferenceMethod {
public:
	// TODO: doc
	SwendsenWangInference(const FactorGraph* fg,
		const std::vector<std::vector<double> >& marg);

	// qf: factor edge appearance or desired cocluster probabilities,
	//     i.e. qf[fi] >= 0.0, qf[fi] < 1.
	// cocluster: if true, qf is a desired co-cluster probability.  If false,
	//     it is the actual edge appearance probability.
	SwendsenWangInference(const FactorGraph* fg,
		const std::vector<double>& qf, bool cocluster = false);
	virtual ~SwendsenWangInference();

	// NOTE: this only works for identical factor graphs, as Swendsen-Wang
	// requires additional parameters for each factor graph.
	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set sampling parameters.
	//
	// verbose: display some statistics (average partition size).
	// spacing_sweeps: number of equivalent-sample sweeps to discard between
	//    samples.
	// sample_count: number of samples used to estimate marginals.
	// use_single_swsteps: if true, one 'sweep' is a single SW transition.  If
	//    false one sweep is an equivalent-sample sweep.
	void SetSamplingParameters(bool verbose, unsigned int burnin_sweeps,
		unsigned int spacing_sweeps, unsigned int sample_count,
		bool use_single_swsteps = false);

	// Perform SW sampling to compute marginals.
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Approximate marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// SW sampling does not support computation of the log-partition function.
	// This method always returns the signaling_NaN value.
	virtual double LogPartitionFunction() const;

	// Produce approximate samples from the distribution
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// NOT IMPLEMENTED
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

	// Get access the underlying sampler object
	SwendsenWangSampler* Sampler(void);
	const std::vector<double>& EdgeAppearanceProbabilities(void) const;

private:
	// Inference result: estimated marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Inference result: estimated log-partition function.
	// XXX: Not available through SW sampling
	double log_z;

	bool verbose;

	// Workhorse: the actual SW sampler used
	SwendsenWangSampler* sw;

	// Edge appearance probabilities
	std::vector<double> edgeprob_out;

	// Gibbs sampling parameters
	unsigned int burnin_sweeps;
	unsigned int spacing_sweeps;
	unsigned int sample_count;
	bool use_single_swsteps;

	void PerformBurninPhase();
	void EdgeAppearanceFromCoclusterProb(const std::vector<double>& qf);
};

}

#endif

