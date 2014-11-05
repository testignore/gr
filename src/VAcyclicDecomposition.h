
#ifndef GRANTE_VACYCLICDECOMPOSITION_H
#define GRANTE_VACYCLICDECOMPOSITION_H

#include <vector>
#include <list>
#include <set>
#include <tr1/unordered_set>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorGraphUtility.h"
#include "Factor.h"
#include "DisjointSetBT.h"

namespace Grante {

class VAcyclicDecomposition {
public:
	explicit VAcyclicDecomposition(const FactorGraph* fg);

	double ComputeDecompositionGreedy(
		const std::vector<double>& factor_weights,
		std::vector<bool>& factor_is_removed);

	// Perform an approximate very-acyclic decomposition of a factor graph.
	// This removes factors from the factorgraph such that it decomposes into
	// a forest.  Moreover, each removed factor could be added individually
	// while retaining the forest property.  This is called v-acyclicity, see
	//
	//    [Bouchard-Cote2009] Alexandre Bouchard-Cote, Michael I. Jordan,
	//       "Optimization of Structured Mean Field Objectives", UAI 2009.
	//
	// The decomposition is performed such that approximately the sum of
	// weights of the retained factors is maximized.  'factor_cost' contains a
	// real-valued cost for the removal of each factor.  The output is
	// returned in 'factor_is_removed', such that factor_is_removed[fi] is
	// true in case the factor is removed.  The return value is the objective
	// function realized.
	double ComputeDecompositionSA(const std::vector<double>& factor_weights,
		std::vector<bool>& factor_is_removed);

	// Same functionality as
	// ComputeDecompositionSA/ComputeDecompositionGreedy, but a fast heuristic
	// based on iterative set-packing/matching subproblems.
	// Of the three, this is asymptotically the fastest.
	double ComputeDecompositionSP(const std::vector<double>& factor_weights,
		std::vector<bool>& factor_is_removed);

	// Same functionality as ComputeDecompositionGreedy, but optimal exact
	// result.  Worst-case exponential complexity.
	double ComputeDecompositionExact(const std::vector<double>& factor_weights,
		std::vector<bool>& factor_is_removed, double opt_eps = 0.0);

	// Approximately solve a set packing problem (generalized hypergraph
	// matching).
	//
	// S: is a collection of vertex sets indexing hyperedges.
	// S_weights: real valued weights, one for each set.
	// S_is_selected: (output) vector indicating whether a set was selected.
	// lr_max_iter: (optional) maximum number of Lagrangian relaxation
	//    iterations.
	//
	// Return the achieved objective S_weights(S_is_selected).
	static double ComputeSetPacking(
		const std::vector<std::tr1::unordered_set<unsigned int> >& S,
		const std::vector<double>& S_weights,
		std::vector<bool>& S_is_selected,
		unsigned int lr_max_iter = 50);

private:
	const FactorGraph* fg;
	FactorGraphUtility fgu;

	// Simulated annealing related constants
	static const unsigned int sa_steps;	// Number of SA steps
	static const double sa_t0;	// Start temperature
	static const double sa_tfinal;	// Final temperature

	// Return true if the factor given by the factor index is the only
	// factor linking its adjacent components
	bool IsComponentBridge(
		std::vector<unsigned int>& node_to_comp,
		std::vector<std::tr1::unordered_set<unsigned int> >& comps,
		unsigned int factor_index) const;

	void SplitComponents(
		std::tr1::unordered_set<unsigned int>& removed_factors,
		std::vector<unsigned int>& node_to_comp,
		std::vector<std::tr1::unordered_set<unsigned int> >& comps,
		unsigned int fac_index) const;
	void MergeComponents(std::vector<unsigned int>& node_to_comp,
		std::vector<std::tr1::unordered_set<unsigned int> >& comps,
		const Factor* fac) const;

	double ComputeDecomposition(const std::vector<double>& factor_cost,
		std::vector<bool>& factor_is_removed,
		unsigned int csa_steps, double csa_t0, double csa_tfinal);

	// Exact algorithm for vac problem, based on reverse search
	class ReverseSearch {
	public:
		ReverseSearch(VAcyclicDecomposition* vac,
			const std::vector<double>& factor_weights, double opt_eps = 0.0);

		double Search(std::vector<bool>& factor_is_removed_out);

	private:
		size_t factor_count;
		std::vector<double> factor_weights;

		VAcyclicDecomposition* vac;
		double best_global;
		std::vector<bool> best_factor_is_removed;
		unsigned long examined;

		// Epsilon for epsilon beam-search
		double opt_eps;

		std::set<unsigned int>::const_iterator Recurse(double obj,
			DisjointSetBT& dset,
			std::list<unsigned int>& factor_in,
			std::list<unsigned int>& factor_cand,
			std::set<unsigned int>& factor_out);

		static unsigned int AddFactor(DisjointSetBT& dset, const Factor* fac);
	};
};

}

#endif

