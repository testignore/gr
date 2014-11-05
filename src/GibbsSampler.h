
#ifndef GRANTE_GIBBSSAMPLER_H
#define GRANTE_GIBBSSAMPLER_H

#include <vector>
#include <tr1/unordered_set>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorGraphUtility.h"

namespace Grante {

/* A simple Gibbs sampler doing fixed-order sweeps on general factor graphs.
 * Used for debugging other algorithms.
 */
class GibbsSampler {
public:
	explicit GibbsSampler(const FactorGraph* fg);

	// Perform a resampling of each variable except fixed onces.
	// The default update is one Metropolized Gibbs update for each variable,
	// scheduled in a random order.
	void Sweep(unsigned int sweep_count = 1);

	const std::vector<unsigned int>& State() const;

	// Force resampling of the given variable
	unsigned int SampleSite(unsigned int var_index) const;
	// Additionally return E(new)-E(old)
	unsigned int SampleSite(unsigned int var_index,
		double& energy_delta);

	// Metropolized Gibbs sampler update, [Liu2001].
	unsigned int SampleSiteMetropolized(unsigned int var_index) const;

	// Set a new state for the sampler, always changing all variables (even
	// fixed ones)
	void SetStateUniformRandom();
	void SetState(const std::vector<unsigned int>& new_state);
	void SetState(unsigned int var_index, unsigned int var_state);

	// Two methods to fix a subset of variables.  The
	// SetFixedVariableIndices() method frees all fixed variables.
	// The SetFixedVariableIndices(var_indices) fixes all given variable
	// indices to their current state.
	// Fixing applies only to the Sweep method.
	void SetFixedVariableIndices();
	void SetFixedVariableIndices(const std::vector<unsigned int>& var_indices);

	// Set temperature: 1.0 is the original distribution, 0.0 the uniform
	// distribution.
	void SetInverseTemperature(double inv_temperature);
	double InverseTemperature(void) const;

private:
	const FactorGraph* fg;
	bool metropolized;
	mutable std::vector<unsigned int> state;

	// Optional, sparse set of fixed variables (for conditional sampling)
	std::tr1::unordered_set<unsigned int> fixed_variables;

	// Random number generation, for the sampler
	boost::mt19937 rgen;
	boost::uniform_real<double> rdestu;	// range [0,1]
	mutable boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu;

	FactorGraphUtility fgu;

	double inv_temperature;

	// Random number generation for metropolized random-scan Gibbs updates
	boost::mt19937 rgen_vc;
	boost::uniform_int<boost::uint32_t> dest_vc;
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<boost::uint32_t> > rand_vc;

	unsigned int SampleSiteUniform(unsigned int var_index) const;
};

}

#endif

