
#ifndef GRANTE_BELIEFPROPAGATION_H
#define GRANTE_BELIEFPROPAGATION_H

#include <tr1/unordered_map>
#include <set>
#include <vector>

#include "FactorGraph.h"
#include "FactorGraphStructurizer.h"
#include "InferenceMethod.h"

namespace Grante {

/* Vanilla Belief Propagation
 */
class BeliefPropagation : public InferenceMethod {
public:
	enum MessageSchedule {
		ParallelSync = 0,
		Sequential,
	};

	BeliefPropagation(const FactorGraph* fg,
		MessageSchedule sched = Sequential);
	virtual ~BeliefPropagation();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	// Set parameters of the belief propagation inference method.
	//
	// verbose: Whether to print iteration statistics,
	// max_iter: Maximum number of message passing sweeps, zero for no limit,
	//    default: 100,
	// conv_tol: Convergence tolerance, default: 1.0e-5.
	void SetParameters(bool verbose, unsigned int max_iter, double conv_tol);

	// Perform loopy belief propagation (sum-product) inference on the current
	// factor graph energies
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Approximate marginals
	virtual const std::vector<double>& Marginal(unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	// Return an approximate log-partition function
	virtual double LogPartitionFunction() const;

	// NOT IMPLEMENTED
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	// Approximate min-sum energy minimization for loopy factor graphs.
	// Return the exact energy of the solution found.
	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// Parameters
	bool verbose;
	unsigned int max_iter;
	double conv_tol;
	MessageSchedule sched;

	bool min_sum;
	std::vector<unsigned int> best_state;	// Best state observed

	// Inference result 1: approximate marginal distributions for all factors
	std::vector<std::vector<double> > marginals;
	// Inference result 2: approximate log-partition function (negative Bethe
	// free energy)
	double log_z;
	// Inference result 3: variable beliefs
	std::vector<std::vector<double> > var_beliefs;

	typedef std::tr1::unordered_map<unsigned int, std::vector<unsigned int> >
		msg_list_t;
	// msglist_for_var[var_index] = list of message indices towards the variable
	msg_list_t msglist_for_var;
	// msglist_for_factor[fac_index] = list of messages indices towards the
	//    factor, ordered by how the variables appear in the factor
	msg_list_t msglist_for_factor;

	// Messages: factor-to-variable
	std::vector<std::vector<double> > msg_for_var;
	std::vector<unsigned int> msg_for_var_srcfactor;

	// Messages: variable-to-factor
	std::vector<std::vector<double> > msg_for_factor;
	std::vector<unsigned int> msg_for_factor_srcvar;

	// If a sequential schedule is used, this is the message order used
	std::vector<FactorGraphStructurizer::OrderStep> order;
	std::vector<unsigned int> order_msgid;

	void PassFactorToVariable();
	void PassFactorToVariable(const Factor* factor,
		unsigned int vi, std::vector<double>& msg,
		const std::vector<unsigned int>& msglist_for_factor_cur);
	void PassVariableToFactor();
	void PassVariableToFactor(unsigned int fi, std::vector<double>& msg,
		unsigned int from_var);

	// Return maximum absolute difference to existing marginals
	double ConstructMarginals();
	double ComputeVariableBeliefs();

	// Reconstruct the approximate minimum energy state from the variable
	// beliefs.  Prior to calling this method the variable beliefs must be
	// computed.
	// Return the exact energy of the state.
	double ReconstructMinimumEnergyState(
		std::vector<unsigned int>& state) const;

	double ComputeBetheFreeEnergy() const;

	void InferenceInitialize();
	void InferenceTeardown();

	// Perform one sweep in a parallel synchronous schedule
	void PerformInferenceStepParallel();

	// Perform one sweep in a sequential schedule
	void PerformInferenceSequential();
};

}

#endif

