
#ifndef GRANTE_DIFFUSIONINFERENCE_H
#define GRANTE_DIFFUSIONINFERENCE_H

#include "FactorGraph.h"
#include "InferenceMethod.h"

namespace Grante {

/* Min sum diffusion is an approximate energy minimization algorithm for
 * general factor graphs.  It is provably convergent and monotonically
 * increases a dual bound on the optimal energy but it is not guaranteed to
 * convergence to the optimal solution.
 *
 * References
 * 1. Tomas Werner, "High-arity Interactions, Polyhedral Relaxations, and
 *    Cutting Plane Algorithm for Soft Constraint Optimisation (MAP-MRF)",
 *    CVPR 2008. (min-sum diffusion for factor graphs)
 * 2. Tomas Werner, "Fixed Points of Loopy Belief Propagation as Zero
 *    Gradients of a Function of Reparameterizations", CTU-CMP-2010-05
 *    techreport, 2010. (sum-product diffusion)
 */
class DiffusionInference : public InferenceMethod {
public:
	DiffusionInference(const FactorGraph* fg);

	virtual ~DiffusionInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;
	virtual void PerformInference();
	virtual void ClearInferenceResult();

	// Set parameters of the min-sum diffusion inference method.
	//
	// verbose: Whether to print iteration statistics,
	// max_iter: Maximum number of subgradient steps, zero for no limit,
	//    default: 100,
	// conv_tol: Convergence tolerance, default: 1.0e-5.
	void SetParameters(bool verbose, unsigned int max_iter, double conv_tol);

	virtual const std::vector<double>& Marginal(
		unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;
	// The returned value is an upper bound on the log-partition function.  It
	// is not exact, even for tree-structured graphs.  (The sum-product
	// version of the algorithm is a very simple approximation and should not
	// be used.)
	virtual double LogPartitionFunction() const;

	// NOT IMPLEMENTED
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	// True if energy minimization is to be performed
	bool min_sum;

	// Primal solution and lower bound on the optimal energy
	std::vector<unsigned int> primal_sol;
	double primal_sol_lb;

	// Parameters
	bool verbose;
	unsigned int max_iter;
	double conv_tol;

	// Inference result 1: approximate marginal distributions for all factors
	std::vector<std::vector<double> > marginals;
	// Inference result 2: upper bound on the log-partition function
	double log_z;

	void PerformInferenceSumProduct();

	double ComputeSumProductObjective(
		const std::vector<std::vector<double> >& phi,
		const std::vector<std::vector<double> >& phi_u) const;
};

}

#endif

