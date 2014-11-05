
#include <limits>
#include <cmath>
#include <cassert>

#include "GibbsSampler.h"
#include "SimulatedAnnealingInference.h"

namespace Grante {

SimulatedAnnealingInference::SimulatedAnnealingInference(
	const FactorGraph* fg, bool verbose)
	: InferenceMethod(fg), verbose(verbose),
	sa_steps(100), T0(10.0), Tfinal(0.05) {
}

SimulatedAnnealingInference::~SimulatedAnnealingInference() {
}

InferenceMethod* SimulatedAnnealingInference::Produce(
	const FactorGraph* new_fg) const {
	SimulatedAnnealingInference* sainf =
		new SimulatedAnnealingInference(new_fg, verbose);
	sainf->SetParameters(sa_steps, T0, Tfinal);
	return (sainf);
}

void SimulatedAnnealingInference::SetParameters(unsigned int sa_steps,
	double T0, double Tfinal) {
	this->sa_steps = sa_steps;
	this->T0 = T0;
	this->Tfinal = Tfinal;
}

void SimulatedAnnealingInference::PerformInference() {
	// Initialize best solution
	unsigned int var_count =
		static_cast<unsigned int>(fg->Cardinalities().size());
	primal_best.resize(var_count);
	primal_best_energy = std::numeric_limits<double>::infinity();

	// Initialize Gibbs sampler
	GibbsSampler gibbs(fg);

	// Calculate exponential multiplier alpha such that T(sa_steps)=Tfinal.
	double alpha = std::exp(std::log(Tfinal / T0) /
		static_cast<double>(sa_steps));
	gibbs.SetInverseTemperature(1.0 / T0);
	gibbs.Sweep(1);

	// Perform annealing
	for (unsigned int k = 1; k <= sa_steps; ++k) {
		// Logarithmic schedule (Geman and Geman)
		// temperature = anneal_C / std::log(1.0 + static_cast<double>(k));

		// Exponential schedule
		double temperature = T0 * std::pow(alpha, static_cast<double>(k));
		gibbs.SetInverseTemperature(1.0 / temperature);
		double cur_energy = fg->EvaluateEnergy(gibbs.State());
		double energy_delta;
		for (unsigned int vi = 0; vi < var_count; ++vi) {
			unsigned int new_state = gibbs.SampleSite(vi, energy_delta);
			gibbs.SetState(vi, new_state);
			cur_energy += energy_delta;
			if (cur_energy < primal_best_energy) {
				// Re-evaluate for numerical reasons
				cur_energy = fg->EvaluateEnergy(gibbs.State());
				if (cur_energy < primal_best_energy) {
					primal_best_energy = cur_energy;
					std::copy(gibbs.State().begin(), gibbs.State().end(),
						primal_best.begin());
				}
			}
		}

	}
}

void SimulatedAnnealingInference::ClearInferenceResult() {
	// nothing to do
}

// XXX: not implemented
const std::vector<double>& SimulatedAnnealingInference::Marginal(
	unsigned int factor_id) const {
	assert(0);
	return (dummy);
}

const std::vector<std::vector<double> >&
SimulatedAnnealingInference::Marginals() const {
	assert(0);
	return (dummy2);
}

// XXX: not implemented
double SimulatedAnnealingInference::LogPartitionFunction() const {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

// XXX: not implemented
void SimulatedAnnealingInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(0);
}

double SimulatedAnnealingInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	PerformInference();
	state = primal_best;

	return (primal_best_energy);
}

}

