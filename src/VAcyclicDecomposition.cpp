
#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>
#include <queue>
#include <utility>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <limits>
#include <cmath>
#include <ctime>
#include <cassert>

#include <boost/random.hpp>
#include <boost/functional/hash.hpp>

#include "Factor.h"
#include "DisjointSet.h"
#include "DisjointSetBT.h"
#include "RandomSource.h"
#include "VAcyclicDecomposition.h"

namespace Grante {

const unsigned int VAcyclicDecomposition::sa_steps = 20;
const double VAcyclicDecomposition::sa_t0 = 5.0;
const double VAcyclicDecomposition::sa_tfinal = 0.1;

VAcyclicDecomposition::VAcyclicDecomposition(const FactorGraph* fg)
	: fg(fg), fgu(fg) {
}

double VAcyclicDecomposition::ComputeDecompositionGreedy(
	const std::vector<double>& factor_weights,
	std::vector<bool>& factor_is_removed) {
	return (ComputeDecomposition(factor_weights, factor_is_removed,
		1, 1.0e-6, 1.0e-6));
}

double VAcyclicDecomposition::ComputeDecompositionSA(
	const std::vector<double>& factor_weights,
	std::vector<bool>& factor_is_removed) {
	return (ComputeDecomposition(factor_weights, factor_is_removed,
		sa_steps, sa_t0, sa_tfinal));
}

// Compute solution based on iterative set packing heuristic
double VAcyclicDecomposition::ComputeDecompositionSP(
	const std::vector<double>& factor_weights,
	std::vector<bool>& factor_is_removed) {
	// Throughout the algorithm, we keep a partition of the factor graph
	// variables
	const std::vector<Factor*>& factors = fg->Factors();
	DisjointSet wset(fg->Cardinalities().size());
	std::set<unsigned int> E;
	for (unsigned int iter = 1; true; ++iter) {
		// Go through all the factors of the model and decide whether they are
		// still active.  A factor is inactive if any of the following
		// conditions are met:
		//
		// 1. All its adjacent variables are already mapped to one component,
		// 2. There are two or more factors joining the same components.
		//    If we would allow these factors to be active, the components
		//    could be merged, violating v-acyclicity.
		//
		// If 1. and 2. are false, the factor is 'active'.
		std::tr1::unordered_map<std::set<unsigned int>, unsigned int,
			boost::hash<std::set<unsigned int> > > EF;
		std::vector<bool> factor_active(factors.size(), true);
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			const Factor* fac = factors[fi];
			const std::vector<unsigned int>& fac_vars = fac->Variables();

			// Build mapped edge set
			E.clear();
			for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi)
				E.insert(wset.FindSet(fac_vars[fvi]));
			// -> All mapped to one component?
			if (E.size() == 1) {
				factor_active[fi] = false;
				continue;
			}

			// Check whether the same component link exist already -> delete
			std::tr1::unordered_map<std::set<unsigned int>,
				unsigned int>::const_iterator efi = EF.find(E);
			if (efi != EF.end()) {
				factor_active[fi] = false;
				factor_active[efi->second] = false;
				continue;
			}

			// Is active, add to dupe checking list
			EF[E] = fi;
		}

		// Solve set packing problem
		std::vector<std::tr1::unordered_set<unsigned int> > S;
		std::vector<double> S_weights;
		std::vector<unsigned int> S_fi;
		std::vector<bool> S_is_selected;
		for (std::tr1::unordered_map<std::set<unsigned int>,
			unsigned int>::const_iterator efi = EF.begin(); efi != EF.end();
			++efi) {
			if (factor_active[efi->second] == false)
				continue;

#if 0
			std::cout << "VASP iter " << iter << ", adding:";
			for (std::set<unsigned int>::const_iterator
				esi = efi->first.begin(); esi != efi->first.end(); ++esi)
				std::cout << " " << *esi;
			std::cout << std::endl;
#endif

			// Add, negative cost to benefit
			S.push_back(std::tr1::unordered_set<unsigned int>(
				efi->first.begin(), efi->first.end()));
			S_weights.push_back(factor_weights[efi->second]);
			S_fi.push_back(efi->second);
		}
		// No further merging possible -> break
		if (S.empty())
			break;
		Grante::VAcyclicDecomposition::ComputeSetPacking(
			S, S_weights, S_is_selected, 5);	// TODO: make 5 configurable

		// Perform merging
		unsigned int merged = 0;
		for (unsigned int si = 0; si < S.size(); ++si) {
			if (S_is_selected[si] == false)
				continue;

			const std::vector<unsigned int>& fac_vars =
				factors[S_fi[si]]->Variables();
			assert(fac_vars.size() >= 2);
			for (unsigned int fvi = 1; fvi < fac_vars.size(); ++fvi) {
				wset.Link(wset.FindSet(fac_vars[0]),
					wset.FindSet(fac_vars[fvi]));
			}
			merged += 1;
		}
		// This can happen due to symmetry, where the Lagrangian relaxation
		// method fails to identify a solution
		if (merged == 0)
			break;
	}

	// Reconstruct final solution: every factor not mapped to one component is
	// removed.
	factor_is_removed.resize(factors.size());
	std::fill(factor_is_removed.begin(), factor_is_removed.end(), true);
	double obj = 0.0;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const Factor* fac = factors[fi];
		const std::vector<unsigned int>& fac_vars = fac->Variables();

		// Build mapped edge set
		E.clear();
		for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi)
			E.insert(wset.FindSet(fac_vars[fvi]));

		// All mapped to one component? -> factor is kept
		if (E.size() == 1) {
			factor_is_removed[fi] = false;
			obj += factor_weights[fi];
		}
	}
	return (obj);
}

double VAcyclicDecomposition::ComputeSetPacking(
	const std::vector<std::tr1::unordered_set<unsigned int> >& S,
	const std::vector<double>& S_weights,
	std::vector<bool>& S_is_selected, unsigned int lr_max_iter) {
	assert(S.size() == S_weights.size());

	// Random number generation
	boost::mt19937 rgen(RandomSource::GetGlobalRandomSeed());
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Fast vertex->edgeset map, VE[v]: list of set indices
	std::tr1::unordered_map<unsigned int, std::set<unsigned int> > VE;
	// Lagrange multipliers for constraints: \sum_{e, v \in e} x(e) <= 1.
	std::tr1::unordered_map<unsigned int, double> V_mu;
	std::tr1::unordered_map<unsigned int, double> V_mu_subg;
	for (unsigned int si = 0; si < S.size(); ++si) {
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			ssi = S[si].begin(); ssi != S[si].end(); ++ssi) {
			VE[*ssi].insert(si);

			// Minimally random initialization to prevent symmetry problems:
			// if many costs have the same weight, and we initialize all
			// multipliers to zero, then we are at a dual degenerate ridge of
			// the cost function, where the subgradient is symmetric: we never
			// manage to break the symmetry, producing trivial solutions.
			V_mu[*ssi] = 1.0e-3*randu();
			V_mu_subg[*ssi] = 0.0;
		}
	}

	// Current, possibly infeasible solution
	std::vector<bool> cur_lr_solution(S.size(), false);
	std::vector<bool> cur_feas_solution(S.size(), false);
	double upper_bound = std::numeric_limits<double>::infinity();
	double lower_bound = -std::numeric_limits<double>::infinity();

	// Perform Lagrangian relaxation iterations
	for (unsigned int lr_iter = 1; lr_iter < lr_max_iter; ++lr_iter) {
		// Solve for x_e
		double cur_obj = 0.0;
		double feas_obj = 0.0;
		for (unsigned int si = 0; si < S.size(); ++si) {
			double si_obj = S_weights[si];
			for (std::tr1::unordered_set<unsigned int>::const_iterator
				ssi = S[si].begin(); ssi != S[si].end(); ++ssi) {
				si_obj += V_mu[*ssi];
			}
			cur_lr_solution[si] = (si_obj > 0.0) ? true : false;
			cur_obj += cur_lr_solution[si] ? si_obj : 0.0;

			cur_feas_solution[si] = cur_lr_solution[si];
			feas_obj += cur_feas_solution[si] ? S_weights[si] : 0.0;
		}
		// - \sum_v \mu_v
		for (std::tr1::unordered_map<unsigned int, double>::const_iterator
			vmi = V_mu.begin(); vmi != V_mu.end(); ++vmi) {
			cur_obj -= vmi->second;
		}

		// cur_obj provides a global upper bound on the solution
		if (cur_obj < upper_bound)
			upper_bound = cur_obj;

		// Check feasibility.  If solution is infeasible, produce a primal
		// feasible solution
		double cslackness = 0.0;
		double feasible = true;
		double subg_norm = 0.0;
		for (std::tr1::unordered_map<unsigned int, double>::iterator
			vmi = V_mu.begin(); vmi != V_mu.end(); ++vmi) {
			double mu_v_sd = -1.0;
			const std::set<unsigned int>& eset = VE[vmi->first];
			for (std::set<unsigned int>::const_iterator ei = eset.begin();
				ei != eset.end(); ++ei) {
				mu_v_sd += cur_lr_solution[*ei] ? 1.0 : 0.0;
			}
			cslackness += vmi->second * mu_v_sd;

			// Constraint is violated, adjust multiplier
			if (mu_v_sd > 0.0) {
				feasible = false;

				// Remove sets until feasible
				double infeas_m = mu_v_sd;
				for (std::set<unsigned int>::const_iterator ei = eset.begin();
					ei != eset.end(); ++ei) {
					// Sufficient number of edges removed? -> done
					if (infeas_m <= 1.0e-8)
						break;

					if (cur_lr_solution[*ei] == false)
						continue;

					infeas_m -= 1.0;

					// Was the set removed already? -> nothing to do
					if (cur_feas_solution[*ei] == false)
						continue;

					// Remove edge
					cur_feas_solution[*ei] = false;
					feas_obj -= S_weights[*ei];
				}
				assert(infeas_m <= 1.0e-8);
			}

			// Adjust Lagrange multiplier by projected gradient method
			subg_norm += mu_v_sd * mu_v_sd;
			V_mu_subg[vmi->first] = mu_v_sd;
		}

		// Compute step size: Polyak
		double beta_m = 1.0;
		double beta = (1.0 + beta_m) /
			(beta_m + static_cast<double>(lr_iter));
		double alpha = 0.0;
		if (subg_norm >= 1.0e-8) {
			alpha = (beta * (cur_obj - feas_obj)) / subg_norm;
		}
		for (std::tr1::unordered_map<unsigned int, double>::iterator
			vmi = V_mu_subg.begin(); vmi != V_mu_subg.end(); ++vmi) {
			// TODO: use subg_norm for LR mult update
			V_mu[vmi->first] -= alpha * vmi->second;
			V_mu[vmi->first] = std::min(0.0, V_mu[vmi->first]);
		}

		// Update lower bound and solution
		if (feas_obj > lower_bound) {
			lower_bound = feas_obj;
			S_is_selected = cur_feas_solution;
		}
#if 0
		std::cout << "set packing LR iter " << lr_iter
			<< ", lb " << lower_bound << ", ub " << upper_bound
			<< ", |subg| " << std::sqrt(subg_norm)
			<< std::endl;
#endif

		// Sufficient optimality condition:
		// primal feasible and \mu'v(y(\mu)) = cslackness = 0.
		if (feasible && std::fabs(cslackness) <= 1.0e-8) {
			S_is_selected = cur_lr_solution;
			return (cur_obj);
		}
	}
	return (lower_bound);
}

double VAcyclicDecomposition::ComputeDecomposition(
	const std::vector<double>& factor_weights,
	std::vector<bool>& factor_is_removed,
	unsigned int csa_steps, double csa_t0, double csa_tfinal) {
	const std::vector<Factor*>& factors = fg->Factors();
	assert(factor_weights.size() == factors.size());

	// Initialization: all factors removed
	std::tr1::unordered_set<unsigned int> removed_factors(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi)
		removed_factors.insert(fi);

	// Components
	std::vector<unsigned int> node_to_comp;
	const std::vector<unsigned int>& card = fg->Cardinalities();
	node_to_comp.reserve(card.size());
	std::vector<std::tr1::unordered_set<unsigned int> > comps(card.size());
	for (unsigned int ni = 0; ni < card.size(); ++ni) {
		node_to_comp[ni] = ni;
		comps[ni].insert(ni);
	}

	// Objective: weights of all included factors (zero)
	double obj = 0.0;

	// Best solution so far
	std::tr1::unordered_set<unsigned int>
		best_removed_factors(removed_factors);
	double best_obj = obj;

	// Random number generators: factor index
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(
		reinterpret_cast<size_t>(fg) ^ std::time(0))+14);
	boost::uniform_int<unsigned int> rdestd(0,
		static_cast<boost::uint32_t>(factors.size()-1));
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_fi(rgen, rdestd);

	// Random number generator: Metropolis chain
	boost::mt19937 rgen2(static_cast<const boost::uint32_t>(
		reinterpret_cast<size_t>(fg) ^ std::time(0))+13);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > rand_m(rgen2, rdestu);

	// Calculate exponential multiplier alpha such that T(sa_steps)=Tfinal.
	double alpha = std::exp(std::log(csa_tfinal / csa_t0) /
		static_cast<double>(csa_steps));
	for (unsigned int k = 1; k <= csa_steps; ++k) {
		// Exponential schedule
		double temperature = csa_t0 * std::pow(alpha, static_cast<double>(k));
#if 0
		std::cout << "iter " << k << ", temp " << temperature
			<< ", best obj " << best_obj << std::endl;
#endif

		// One epoch
		for (unsigned int fi_d = 0; fi_d < factors.size(); ++fi_d) {
			unsigned int fi = rand_fi();
			// Special case of one pass: linear
			if (csa_steps == 1)
				fi = fi_d;

			double delta_E = std::numeric_limits<double>::signaling_NaN();
			bool is_in_G = (removed_factors.count(fi) == 0);
			if (is_in_G) {
				// Factor is currently in G, can be removed.
				delta_E = -factor_weights[fi];
			} else {
				// Factor is currently removed, check whether it can be added.
				if (IsComponentBridge(node_to_comp, comps, fi) == false)
					continue;	// not possible, reject

				delta_E = factor_weights[fi];
			}
			// Reject
			if (delta_E <= 0.0 && rand_m() >= std::exp(delta_E/temperature))
				continue;

			// Accept
			if (is_in_G) {
				// Split components adjacent to the factor
				SplitComponents(removed_factors, node_to_comp, comps, fi);
			} else {
				// Merge components adjacent to the factor
				MergeComponents(node_to_comp, comps, factors[fi]);
				removed_factors.erase(fi);	// its in the graph now
			}
			obj += delta_E;

			// Keep track of best solution
			if (obj > best_obj) {
				best_obj = obj;
				best_removed_factors = removed_factors;
			}
		}
	}

	// Return list of removed factors
	factor_is_removed.resize(factors.size());
	std::fill(factor_is_removed.begin(), factor_is_removed.end(), false);
	for (std::tr1::unordered_set<unsigned int>::const_iterator
		ri = best_removed_factors.begin(); ri != best_removed_factors.end();
		++ri) {
		factor_is_removed[*ri] = true;
	}
	return (best_obj);
}

bool VAcyclicDecomposition::IsComponentBridge(
	std::vector<unsigned int>& node_to_comp,
	std::vector<std::tr1::unordered_set<unsigned int> >& comps,
	unsigned int factor_index) const {
	const Factor* fac = fg->Factors()[factor_index];
	const std::vector<unsigned int>& fvars = fac->Variables();

	// 1. Collect adjacent factor sets for each component
	std::vector<std::tr1::unordered_set<unsigned int> >
		comp_facset(fvars.size());;
	for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
		const std::tr1::unordered_set<unsigned int>& cur_vars =
			comps[node_to_comp[fvars[fvi]]];
		assert(cur_vars.empty() == false);
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			cvi = cur_vars.begin(); cvi != cur_vars.end(); ++cvi) {
			const std::set<unsigned int>& cur_facset =
				fgu.AdjacentFactors(*cvi);
			comp_facset[fvi].insert(cur_facset.begin(), cur_facset.end());
		}
		// Do not consider the factor of interest
		comp_facset[fvi].erase(factor_index);
	}

	// 2. Find factors that would link the components
	for (unsigned int c1 = 0; c1 < comp_facset.size(); ++c1) {
		const std::tr1::unordered_set<unsigned int>& c1_fset = comp_facset[c1];
		for (unsigned int c2 = c1 + 1; c2 < comp_facset.size(); ++c2) {
			for (std::tr1::unordered_set<unsigned int>::const_iterator
				c2i = comp_facset[c2].begin(); c2i != comp_facset[c2].end();
				++c2i) {
				// Check whether another factor between the components exists.
				// If so, factor_index is no bridge.
				if (c1_fset.count(*c2i) > 0)
					return (false);
			}
		}
	}
	return (true);
}

void VAcyclicDecomposition::SplitComponents(
	std::tr1::unordered_set<unsigned int>& removed_factors,
	std::vector<unsigned int>& node_to_comp,
	std::vector<std::tr1::unordered_set<unsigned int> >& comps,
	unsigned int fac_index) const {
	// Obtain variable indices (these are all in the same component)
	const Factor* fac = fg->Factors()[fac_index];
	const std::vector<unsigned int>& fvar = fac->Variables();
	for (unsigned int fvi = 0; fvi < fvar.size(); ++fvi) {
		assert(node_to_comp[fvar[fvi]] == node_to_comp[fvar[0]]);
	}

	// Save joined component, remove factor from graph
#if 0
	std::tr1::unordered_map<unsigned int> comp_old;
	comp_old.swap(comps[node_to_comp[fvar[0]]]);
#endif
	comps[node_to_comp[fvar[0]]].clear();
	removed_factors.insert(fac_index);

	// Relabel tree rooted in factor-adjacent node to a new component
	// FIXME: this seems to be very slow
	for (unsigned int fvi = 0; fvi < fvar.size(); ++fvi) {
		unsigned int vi = fvar[fvi];	// var and component index
		std::tr1::unordered_set<unsigned int>& comp_vi = comps[vi];
		assert(comp_vi.empty());

		// store (var_idx, came_from_factor_index) in queue
		std::queue<std::pair<unsigned int, unsigned int> > var_q;
		var_q.push(std::pair<unsigned int, unsigned int>(vi, fac_index));

		// Recurse on tree-structured partial component, relabeling in the
		// process
		while (var_q.empty() == false) {
			const std::pair<unsigned int, unsigned int>& cur = var_q.front();

			// Relabel this node
			comp_vi.insert(cur.first);
			node_to_comp[cur.first] = vi;

			// Insert adjacent nodes
			const std::set<unsigned int>& cur_facset =
				fgu.AdjacentFactors(cur.first);
			for (std::set<unsigned int>::const_iterator
				fi = cur_facset.begin(); fi != cur_facset.end(); ++fi) {
				if (*fi == cur.second)
					continue;	// this is the factor we came from

				// If factor is removed, skip
				if (removed_factors.count(*fi) > 0)
					continue;

				// Factor is valid, add all its variables except this one
				const Factor* cur_factor = fg->Factors()[*fi];
				const std::vector<unsigned int>& cur_facvar =
					cur_factor->Variables();
				for (std::vector<unsigned int>::const_iterator
					cfvi = cur_facvar.begin(); cfvi != cur_facvar.end();
					++cfvi) {
					if (*cfvi == cur.first)
						continue;	// do not add ourselves again

					// Put variable into the queue
					var_q.push(std::pair<unsigned int, unsigned int>(
						*cfvi, *fi));
				}
			}
			var_q.pop();
		}
	}
}

void VAcyclicDecomposition::MergeComponents(
	std::vector<unsigned int>& node_to_comp,
	std::vector<std::tr1::unordered_set<unsigned int> >& comps,
	const Factor* fac) const {
	// Obtain component indices (these are all be disjoint)
	const std::vector<unsigned int>& fvar = fac->Variables();
	std::vector<unsigned int> comp_indices(fvar.size());
	for (unsigned int fvi = 0; fvi < fvar.size(); ++fvi)
		comp_indices[fvi] = node_to_comp[fvar[fvi]];

	// Map them all to the first component
	unsigned int target = comp_indices[0];
	for (unsigned int ci = 1; ci < comp_indices.size(); ++ci) {
		// 1. relabel all nodes to the new component
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			si = comps[comp_indices[ci]].begin();
			si != comps[comp_indices[ci]].end(); ++si) {
			node_to_comp[*si] = target;
		}
		// 2. merge component sets
		comps[target].insert(comps[comp_indices[ci]].begin(),
			comps[comp_indices[ci]].end());
		comps[comp_indices[ci]].clear();
	}
}

double VAcyclicDecomposition::ComputeDecompositionExact(
	const std::vector<double>& factor_weights,
	std::vector<bool>& factor_is_removed, double opt_eps) {
	// Call reverse search enumeration function
	ReverseSearch rsearch(this, factor_weights, opt_eps);
	double obj = rsearch.Search(factor_is_removed);

	return (obj);
}

VAcyclicDecomposition::ReverseSearch::ReverseSearch(VAcyclicDecomposition* vac,
	const std::vector<double>& factor_weights, double opt_eps)
	: factor_count(vac->fg->Factors().size()), factor_weights(factor_weights),
		vac(vac), best_global(0.0),
		best_factor_is_removed(vac->fg->Factors().size(), true),
		opt_eps(opt_eps) {
}

double VAcyclicDecomposition::ReverseSearch::Search(
	std::vector<bool>& factor_is_removed_out) {
	std::list<unsigned int> factor_in;	// start with empty set
	std::list<unsigned int> factor_cand;
	std::set<unsigned int> factor_out;
	for (unsigned int fi = 0; fi < factor_count; ++fi) {
		factor_cand.push_back(fi);
		factor_out.insert(fi);
	}

	examined = 0;
	std::cout << "SEARCH BEGIN" << std::endl;
	DisjointSetBT dset(factor_count);
	Recurse(0.0, dset, factor_in, factor_cand, factor_out);
	std::cout << "SEARCH END" << std::endl << std::endl;

	factor_is_removed_out = best_factor_is_removed;

	return (best_global);
}

// This has polynomial delay at O(F log F + F N(F)), where the output
// operation is a an enumeration of a vac structure.  So in effect if the
// maximum factor scope size N(F) is small, the runtime is an almost linear
// function in the number of factors of the graph.
std::set<unsigned int>::const_iterator
VAcyclicDecomposition::ReverseSearch::Recurse(double obj, DisjointSetBT& dset,
	std::list<unsigned int>& factor_in, std::list<unsigned int>& factor_cand,
	std::set<unsigned int>& factor_out)
{
	examined += 1;
	if (examined % 1000000 == 1) {
		std::cout << "Recurse(obj=" << obj << ", factor_in (" << factor_in.size()
			<< "), factor_cand (" << factor_cand.size() << "), "
			<< "factor_out (" << factor_out.size() << "))"
			<< " best: " << best_global << std::endl;
	}

	// 1. Determine whether this tree is still vac.
	// When we find that it is not vac, return.
	//
	// Complexity: O(F log log F)

	// We need to check all factors not in factor_in
	for (std::set<unsigned int>::const_iterator foi = factor_out.begin();
		foi != factor_out.end(); ++foi) {
		const Factor* fac = vac->fg->Factors()[*foi];
		const std::vector<unsigned int>& fvars = fac->Variables();

		for (unsigned int fvi1 = 0; fvi1 < fvars.size(); ++fvi1) {
			unsigned int root1 = dset.Find(fvars[fvi1]);
			for (unsigned int fvi2 = fvi1 + 1; fvi2 < fvars.size(); ++fvi2) {
				unsigned int root2 = dset.Find(fvars[fvi2]);
#if 0
				std::cout << "   fi " << *foi << ", " << fvars[fvi1]
					<< "--" << fvars[fvi2] << " map to roots "
					<< root1 << "--" << root2 << std::endl;
#endif

				if (root1 == root2) {
#if 0
					std::cout << "   * not vac, returning" << std::endl;
#endif
					return (foi);
				}
			}
		}
	}

	// 2. Current set is vac, so update objective
	//
	// Complexity: O(1)
	if (obj > best_global) {
		best_global = obj;
		best_factor_is_removed.resize(factor_count);
		std::fill(best_factor_is_removed.begin(),
			best_factor_is_removed.end(), true);
		for (std::list<unsigned int>::const_iterator fil = factor_in.begin();
			fil != factor_in.end(); ++fil) {
			best_factor_is_removed[*fil] = false;
		}
		std::cout << "   found new best vac with obj " << obj << std::endl;
	}

	// 3. Compute bound and if it turns out that we cannot beat the best
	// solution so far, stop recursion
	//
	// Complexity: O(F)
	double obj_upper_bound = obj;
	for (std::list<unsigned int>::const_iterator fci = factor_cand.begin();
		fci != factor_cand.end(); ++fci) {
		obj_upper_bound += std::max(0.0, factor_weights[*fci]);
	}
#if 0
	if (examined % 10000 == 1) {
		std::cout << "   upper bound at this node: " << obj_upper_bound << std::endl;
	}
#endif

	if (obj_upper_bound <= best_global + opt_eps)
		return (factor_out.end());	// cannot beat best global one

	// 4. For all remaining factors, add and recurse
	//
	// Complexity: O(F N(F) + F log F) + recursion
	//  i) Sort by weight
	factor_cand.sort(
		[this](unsigned int v1, unsigned int v2) -> bool {
			return (this->factor_weights[v1] > this->factor_weights[v2]);
		});

	// ii) Recurse for each, greedily
	std::list<unsigned int>::iterator fii = factor_cand.begin();
	while (fii != factor_cand.end()) {
		// Skip factors that would not contribute anything to the objective
		unsigned int fi = *fii;
		if (factor_weights[fi] <= 0.0) {
			++fii;
			continue;
		}

		const Factor* fac = vac->fg->Factors()[fi];
		unsigned int union_count = AddFactor(dset, fac);

		// Prepare recursion
		factor_in.push_back(fi);	// more factor into retained factor set
		factor_out.erase(fi);

		// Note that here we change the list factor_cand, so that we realize
		// the reverse search ordering and avoid enumerating duplicates
		fii = factor_cand.erase(fii);	// erase factor from candidate list
		std::list<unsigned int> remaining_cand(fii, factor_cand.end());

		// Recurse
		std::set<unsigned int>::iterator conflict_iter =
			Recurse(obj + factor_weights[fi], dset, factor_in,
				remaining_cand, factor_out);

		// Handling discovered vac conflicts in the remaining candidate set
		if (conflict_iter != factor_out.end()) {
			unsigned int conflict_fi = *conflict_iter;

#if 0
			if (std::find(factor_cand.begin(), factor_cand.end(), conflict_fi)
				!= factor_cand.end()) {
				std::cout << "   removed conflict" << std::endl;
			}
#endif
			factor_cand.remove_if([conflict_fi](unsigned int k) -> bool {
				return (conflict_fi == k);
			});
			fii = factor_cand.begin();
		}

		factor_out.insert(fi);
		factor_in.pop_back();

		// Undo
		for (unsigned int uc = 0; uc < union_count; ++uc)
			dset.Deunion();
	}
	return (factor_out.end());
}

unsigned int VAcyclicDecomposition::ReverseSearch::AddFactor(DisjointSetBT& dset,
	const Factor* fac) {
	const std::vector<unsigned int>& fvars = fac->Variables();
	unsigned int union_count = 0;

	// Union all variables in the factor scope N(F)
	unsigned int root = dset.Find(fvars[0]);
	for (unsigned int fvi = 1; fvi < fvars.size(); ++fvi) {
		unsigned int root2 = dset.Find(fvars[fvi]);
		assert(root != root2);
		root = dset.Union(root, root2);
		union_count += 1;
	}

	return (union_count);
}

}

