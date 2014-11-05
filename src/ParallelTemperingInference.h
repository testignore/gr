
#ifndef GRANTE_PTINFERENCE_H
#define GRANTE_PTINFERENCE_H

#include <vector>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "InferenceMethod.h"
#include "GibbsSampler.h"

namespace Grante {

/* Parallel tempering, also known as Replica-Exchange Monte Carlo
 *
 * References
 * [Hukushima1996] Hukushima, Nemoto, "Exchange Monte Carlo method and
 *   application to spin glass simulations", Journal of the Physical Society
 *   of Japan, Vol. 65, No. 4, pages 1604-1608, 1996.
 * [Liu2004] Jun S. Liu, "Monte Carlo Strategies in Scientific Computing",
 *   Springer, 2004.
 *
 * TODO: better temperature ladder.  Test acceptance rate functions on real models
 */
class ParallelTemperingInference : public InferenceMethod {
public:
	explicit ParallelTemperingInference(const FactorGraph* fg);
	virtual ~ParallelTemperingInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set sampling parameters for parallel tempering.
	//
	// levels: number of parallel temperized Markov chains to run.  The
	//    default is 20.  The runtime is linear in the number of chains.
	// high_temp: temperature of the highest chain in the ladder.  The lowest
	//    chain has temperature 1.0.  We must have high_temp > 1.0.  The
	//    default is 20.0.
	// swap_probability: the probability by which to attempt a temperature
	//    swap.  Default: 0.5.
	// burnin_sweeps: number of sweeps to perform during burn-in phase.
	//    Default: 1000.
	// sample_count: number of approximate samples to use to estimate marginal
	//    distributions.  Default: 1000.
	void SetSamplingParameters(unsigned int levels, double high_temp,
		double swap_probability, unsigned int burnin_sweeps,
		unsigned int sample_count);

	// Perform parallel tempering to obtain approximate marginals
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Return average acceptance probabilities in the ladder.  The returned
	// vector has length levels-1.
	const std::vector<double>& AcceptanceProbabilities(void) const;

	// Approximate marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// Parallel tempering does not support computation of the log-partition
	// function.
	// This method always returns the signaling_NaN value.
	virtual double LogPartitionFunction() const;

	// Produce a set of approximate samples from the target distributions.
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// NOT SUPPORTED
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// Inference result: estimated marginal distributions for all factors
	std::vector<std::vector<double> > marginals;

	// Workhorse: the set of temperized Gibbs samplers
	std::vector<GibbsSampler*> ladder;

	// Random number generation, for chain/swap selection
	boost::mt19937 rgen;
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu;

	std::vector<double> accept_prob;

	// Parallel tempering parameters
	unsigned int levels;
	double high_temp;
	double swap_probability;
	unsigned int burnin_sweeps;
	unsigned int sample_count;

	void InitializeLadder(const FactorGraph* fg, unsigned int levels,
		double high_temp);
	void DestroyLadder(void);
};

}

#endif

