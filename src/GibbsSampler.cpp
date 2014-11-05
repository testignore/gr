
#include <algorithm>
#include <cmath>
#include <cassert>

#include "RandomSource.h"
#include "GibbsSampler.h"

namespace Grante {

GibbsSampler::GibbsSampler(const FactorGraph* fg)
	: fg(fg), metropolized(false), rgen(RandomSource::GetGlobalRandomSeed()),
		randu(rgen, rdestu), fgu(fg), inv_temperature(1.0),
		rgen_vc(RandomSource::GetGlobalRandomSeed()),
		dest_vc(0, static_cast<boost::uint32_t>(fg->Cardinalities().size()-1)),
		rand_vc(rgen_vc, dest_vc)
{
	// Initialize state
	state.resize(fg->Cardinalities().size());
	std::fill(state.begin(), state.end(), 0);
}

void GibbsSampler::Sweep(unsigned int sweep_count) {
	if (sweep_count == 0)
		return;

	size_t var_count = fg->Cardinalities().size();

	// Metropolized Gibbs sampler, random-scan
	// Note: the sequential version is not aperiodic, hence cannot be used
	// without making the chain lazy (which we do not want).
	if (metropolized) {
		for (unsigned int sweep = 0; sweep < sweep_count; ++sweep) {
			for (size_t vci = 0; vci < var_count; ++vci) {
				unsigned int vi = rand_vc();
				if (fixed_variables.empty() == false &&
					fixed_variables.count(vi) > 0)
					continue;

				state[vi] = SampleSiteMetropolized(vi);
				assert(state[vi] < fg->Cardinalities()[vi]);
			}
		}
		return;
	}

	// Order variables randomly
	std::vector<unsigned int> vec(var_count);
	for (unsigned int vi = 0; vi < var_count; ++vi)
		vec[vi] = vi;

	for (unsigned int sweep = 0; sweep < sweep_count; ++sweep) {
		RandomSource::ShuffleRandom(vec);
		for (unsigned int cvi = 0; cvi < var_count; ++cvi) {
			unsigned int vi = vec[cvi];
			assert(vi < var_count);

			// Do not resample fixed variables
			if (fixed_variables.empty() == false &&
				fixed_variables.count(vi) > 0)
				continue;

			state[vi] = SampleSite(vi);
			assert(state[vi] < fg->Cardinalities()[vi]);
		}
	}
}

const std::vector<unsigned int>& GibbsSampler::State() const {
	return (state);
}

void GibbsSampler::SetStateUniformRandom() {
	size_t var_count = fg->Cardinalities().size();
	for (size_t vi = 0; vi < var_count; ++vi)
		state[vi] = SampleSiteUniform(static_cast<unsigned int>(vi));
}

void GibbsSampler::SetState(const std::vector<unsigned int>& new_state) {
	assert(new_state.size() == state.size());
	std::copy(new_state.begin(), new_state.end(), state.begin());
}

void GibbsSampler::SetState(unsigned int var_index, unsigned int var_state) {
	assert(var_index < state.size());
	state[var_index] = var_state;
}

void GibbsSampler::SetFixedVariableIndices() {
	// Release all variables
	fixed_variables.clear();
}

void GibbsSampler::SetFixedVariableIndices(
	const std::vector<unsigned int>& var_indices) {
	fixed_variables.clear();
	fixed_variables.insert(var_indices.begin(), var_indices.end());
}

void GibbsSampler::SetInverseTemperature(double inv_temperature) {
	this->inv_temperature = inv_temperature;
}

double GibbsSampler::InverseTemperature(void) const {
	return (inv_temperature);
}

unsigned int GibbsSampler::SampleSite(unsigned int var_index) const {
	unsigned int var_card = fg->Cardinalities()[var_index];
	std::vector<double> cond_dist_unnorm(var_card);

	double Z = fgu.ComputeConditionalSiteDistribution(state, var_index,
		cond_dist_unnorm, inv_temperature);
	double rval = Z * randu();
	double cumsum = 0.0;
	for (unsigned int vi = 0; vi < var_card; ++vi) {
		cumsum += cond_dist_unnorm[vi];
		if (rval <= cumsum)
			return (vi);
	}
	assert(0);
	return (std::numeric_limits<unsigned int>::max());
}

// Metropolized Gibbs sampler, statistically more efficient than the single
// site Gibbs sampler.  See [Liu2001], section 6.3.2.
unsigned int GibbsSampler::SampleSiteMetropolized(unsigned int var_index) const {
	unsigned int var_card = fg->Cardinalities()[var_index];
	unsigned int vi_state = state[var_index];
	std::vector<double> cond_dist_unnorm(var_card);

	double Z = fgu.ComputeConditionalSiteDistribution(state, var_index,
		cond_dist_unnorm, inv_temperature);
	double rval = (Z - cond_dist_unnorm[vi_state]) * randu();
	double cumsum = 0.0;
	unsigned int svi = 0;
	for (unsigned int vi = 0; vi < var_card; ++vi) {
		// Draw from p(y_i)/(1-p(y'_i))
		if (vi == vi_state)
			continue;

		cumsum += cond_dist_unnorm[vi];
		if (rval <= cumsum) {
			svi = vi;
			break;
		}
	}

	// Perform Metropolis accept-reject step
	double alpha = std::min(1.0,
		(1.0-cond_dist_unnorm[vi_state]/Z) / (1.0-cond_dist_unnorm[svi]/Z));
	assert(alpha >= 0.0);
	if (randu() <= alpha)
		return (svi);	// accept

	return (vi_state);	// reject
}

unsigned int GibbsSampler::SampleSiteUniform(unsigned int var_index) const {
	unsigned int var_card = fg->Cardinalities()[var_index];
	unsigned int state = static_cast<unsigned int>(
		randu() * static_cast<double>(var_card));
	assert(state < var_card);
	return (state);
}

unsigned int GibbsSampler::SampleSite(unsigned int var_index,
	double& energy_delta) {
	energy_delta = 0.0;
	unsigned int old_state = state[var_index];
	unsigned int new_state = SampleSite(var_index);
	if (old_state != new_state) {
		energy_delta = fgu.ComputeEnergyChange(state, var_index,
			old_state, new_state);
	}
	return (new_state);
}

}

