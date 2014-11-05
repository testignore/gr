
#ifndef GRANTE_LPMAPINFERENCE_H
#define GRANTE_LPMAPINFERENCE_H

#include <vector>
#include <tr1/unordered_map>

#include "InferenceMethod.h"
#include "TreeInference.h"
#include "SubFactorGraph.h"

namespace Grante {

/* Tree-decomposition based first order MAP-MRF LP relaxation solver.  This
 * method provides both a lower and upper bound to the optimal energy.  It
 * additionally provides a labeling with energy greater than or equal to the
 * optimal labeling and a fractional labeling defined on the same index set as
 * the marginals.
 */
class LinearProgrammingMAPInference : public InferenceMethod {
public:
	/* Note: right now this class works only on factor graphs that have at
	 * least one unary factor for each variable.
	 */
	LinearProgrammingMAPInference(const FactorGraph* fg, bool verbose = false);
	virtual ~LinearProgrammingMAPInference();

	virtual InferenceMethod* Produce(const FactorGraph* new_fg) const;

	// Set parameters of the MAP-MRF LP inference method.
	//
	// max_iter: Maximum number of subgradient steps, zero for no limit,
	//    default: 100,
	// conv_tol: Convergence tolerance, default: 1.0e-6.
	void SetParameters(unsigned int max_iter, double conv_tol);

	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// XXX: returns the relaxed solution
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

	// Maximum number of subgradient steps
	unsigned int max_iter;
	// Convergence tolerance: duality_grap / abs(dual_obj)
	double conv_tol;

	// Primal feasible labeling and its energy
	std::vector<unsigned int> primal_best;
	double primal_best_energy;

	// (Infeasible) primal LP solution and its energy.  We always have
	// relaxed_sol_energy <= primal_best_energy.
	std::vector<std::vector<double> > relaxed_sol;
	double relaxed_sol_energy;

	// A small set of trees that together cover the original graph
	size_t T;
	std::vector<SubFactorGraph*> trees;
	std::vector<std::vector<unsigned int> > tree_factor_indices;
	std::vector<unsigned int> factor_cover_count;

	typedef std::tr1::unordered_map<unsigned int, unsigned int>
		var_to_factor_map_t;

	// tree_var_to_factor_map[ti][vi] gives the factor index in tree ti that
	// is a unary factor acting on the original FactorGraph's variable vi.
	// This allows us to modify the energies of all unaries across all trees.
	std::vector<var_to_factor_map_t> tree_var_to_factor_map;

	// The actual tree min-sum inference objects
	std::vector<TreeInference*> tree_inf;
};

}

#endif

