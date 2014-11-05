
#ifndef GRANTE_SAINFERENCE_H
#define GRANTE_SAINFERENCE_H

#include "InferenceMethod.h"

namespace Grante {

/* Simulated annealing approximate MAP inference
 */
class SimulatedAnnealingInference : public InferenceMethod {
public:
	SimulatedAnnealingInference(const FactorGraph* fg, bool verbose = false);
	virtual ~SimulatedAnnealingInference();

	virtual InferenceMethod* Produce(const FactorGraph* new_fg) const;

	// Set the annealing parameters
	//
	// sa_steps: Number of simulated annealing distributions, default: 100,
	// T0: initial Boltzmann temperature, default: 10.0,
	// Tfinal: final Boltzmann temperature, default: 0.05.
	//
	// For high-quality solutions, you might want to increase sa_steps and
	// lower Tfinal to a smaller value (eg. 0.001).
	void SetParameters(unsigned int sa_steps, double T0, double Tfinal);

	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// XXX: not implemented
	virtual const std::vector<double>& Marginal(
		unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// XXX: not implemented
	virtual double LogPartitionFunction() const;
	// XXX: not implemented
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// Obtain an approximate minimum energy state for the current factor graph
	// energies.
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	bool verbose;

	// Primal feasible labeling and its energy
	std::vector<unsigned int> primal_best;
	double primal_best_energy;

	// Simulated annealing parameters
	unsigned int sa_steps;	// Total number of annealing steps
	double T0;	// Initial temperature (high)
	double Tfinal;	// Final temperature (low)

	// Dummy (for being able to return a const reference in the Marginals
	// methods)
	std::vector<double> dummy;
	std::vector<std::vector<double> > dummy2;
};

}

#endif

