
#ifndef GRANTE_SAMCINFERENCE_H
#define GRANTE_SAMCINFERENCE_H

#include <vector>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "InferenceMethod.h"

namespace Grante {

/* Stochastic Approximation Monte Carlo, also known as generalized Wang-Landau
 * method.
 *
 * References
 * [Liang2010] Faming Liang, Chuanhai Liu, Raymond J. Carroll, "Advanced
 *   Markov Chain Monte Carlo Methods: Learning from Past Samples", Wiley,
 *   2010.
 */
class SAMCInference : public InferenceMethod {
public:
	explicit SAMCInference(const FactorGraph* fg);
	virtual ~SAMCInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set sampling parameters for SAMC.
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

	// Return temperature space histogram, 'levels' elements with absolute
	// visit counts.
	const std::vector<unsigned int>& TemperatureHistogram(void) const;
	const std::vector<double>& LogPartitionEstimates(void) const;

	// Approximate marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// SAMC does not support computation of the absolute log-partition
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

	// Random number generation, for chain/swap selection
	boost::mt19937 rgen;
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu;

	std::vector<double> temperatures;	// Temperature ladder, [0]=1.0
	std::vector<unsigned int> histogram;	// Visit count
	std::vector<double> theta;	// Relative log-partition function estimates

	// Parallel tempering parameters
	unsigned int levels;
	double high_temp;
	double swap_probability;
	unsigned int burnin_sweeps;
	unsigned int sample_count;

	std::vector<std::vector<unsigned int> > samples;

	void InitializeLadder(const FactorGraph* fg, unsigned int levels,
		double high_temp);
	void DestroyLadder(void);

	void PerformInference(bool keep_samples, unsigned int sample_count);
};

}

#endif

