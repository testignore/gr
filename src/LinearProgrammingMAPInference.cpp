
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <cmath>
#include <cassert>

#include <boost/lambda/lambda.hpp>

#include "TreeCoverDecomposition.h"
#include "LinearProgrammingMAPInference.h"

using namespace boost::lambda;

namespace Grante {

LinearProgrammingMAPInference::LinearProgrammingMAPInference(
	const FactorGraph* fg, bool verbose)
	: InferenceMethod(fg), verbose(verbose), max_iter(100), conv_tol(1.0e-6),
		primal_best_energy(0.0), T(0) {
	// Check each variable has at least one unary factors attached
	size_t var_count = fg->Cardinalities().size();
	std::vector<int> has_unary(var_count, -1);
	has_unary.resize(var_count, -1);
	const std::vector<Factor*>& factors = fg->Factors();
	for (size_t fi = 0; fi < factors.size(); ++fi) {
		const Factor* fac = factors[fi];
		if (fac->Variables().size() > 1)
			continue;

		assert(fac->Variables().size() == 1);
		unsigned int var_index = fac->Variables()[0];
		if (has_unary[var_index] >= 0)
			continue;
		has_unary[var_index] = static_cast<int>(fi);
	}
	if (std::find(has_unary.begin(), has_unary.end(), -1) !=
		has_unary.end()) {
		std::cout << "Factor graph has variables without unary factors.  "
			<< "This is required for MAP-MRF LP inference." << std::endl;
		assert(false);
	}

	// Build a fast factor-to-variable lookup map for the unary factors
	std::tr1::unordered_map<unsigned int, unsigned int> unary_fi_to_var;
	for (unsigned int ui = 0; ui < has_unary.size(); ++ui)
		unary_fi_to_var[static_cast<unsigned int>(has_unary[ui])] = ui;

	primal_best.resize(var_count, 0);

	// Perform tree decomposition
	TreeCoverDecomposition tcov_decomp(fg);
	tcov_decomp.ComputeDecompositionGreedy(tree_factor_indices,
		factor_cover_count);
	T = tree_factor_indices.size();

	// Instantiate spanning trees
	trees.resize(T, 0);
	tree_var_to_factor_map.resize(T);
	tree_inf.resize(T, 0);
	for (size_t t = 0; t < T; ++t) {
		// Scale all factors by the inverse number of times they are covered
		std::vector<double> f_scale(tree_factor_indices[t].size());
		for (size_t tfi = 0; tfi < f_scale.size(); ++tfi) {
			// Factor tfi in tree t
			f_scale[tfi] = 1.0 /static_cast<double>(
				factor_cover_count[tree_factor_indices[t][tfi]]);
		}

		// Create a subgraph with the given set of factors
		trees[t] = new SubFactorGraph(fg, tree_factor_indices[t], f_scale);
		assert(trees[t]->FG()->Cardinalities().size() == var_count);

		// Build a map for "global variable index -> per-tree factor index"
		// lookups
		for (size_t tfi = 0; tfi < tree_factor_indices[t].size(); ++tfi) {
			// Is it a unary factor?
			unsigned int fi = tree_factor_indices[t][tfi];
			if (unary_fi_to_var.count(fi) == 0)
				continue;

			// It is, find the variable index
			unsigned int var_index = unary_fi_to_var[fi];
			tree_var_to_factor_map[t][var_index] =
				static_cast<unsigned int>(tfi);
		}

		// Tree inference object for this subgraph
		tree_inf[t] = new TreeInference(trees[t]->FG());
	}
}

LinearProgrammingMAPInference::~LinearProgrammingMAPInference() {
	for (size_t t = 0; t < T; ++t) {
		delete (tree_inf[t]);
		delete (trees[t]);
	}
}

InferenceMethod* LinearProgrammingMAPInference::Produce(
	const FactorGraph* new_fg) const {
	return (new LinearProgrammingMAPInference(new_fg));
}

void LinearProgrammingMAPInference::SetParameters(
	unsigned int max_iter, double conv_tol) {
	assert(conv_tol >= 0.0);
	this->max_iter = max_iter;
	this->conv_tol = conv_tol;
}

void LinearProgrammingMAPInference::PerformInference() {
	// Distribute energies uniformly over decomposed trees
	for (size_t t = 0; t < T; ++t)
		trees[t]->ForwardMap();

	// Initialize primal labeling
	const std::vector<unsigned int>& var_card = fg->Cardinalities();
	size_t var_count = fg->Cardinalities().size();
	primal_best.resize(var_count);
	std::fill(primal_best.begin(), primal_best.end(), 0);
	primal_best_energy = std::numeric_limits<double>::infinity();

	// Initialize relaxed solution
#if 0
	const std::vector<Factor*>& factors = fg->Factors();
	relaxed_sol.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		relaxed_sol[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(relaxed_sol[fi].begin(), relaxed_sol[fi].end(), 0.0);
	}
	relaxed_sol_energy = -std::numeric_limits<double>::infinity();
#endif

	// \lambda_{t,vi,state} = 0
	std::vector<std::vector<double> > sol_avg(var_count);
	for (size_t vi = 0; vi < var_count; ++vi) {
		unsigned int vi_card = var_card[vi];
		sol_avg[vi].resize(vi_card, 0.0);
	}
	// Individual tree solutions
	std::vector<std::vector<unsigned int> > cur_sol_t(T);
	for (size_t t = 0; t < T; ++t)
		cur_sol_t[t].resize(var_count);

	// Stepsize control
	// The control mechanism is from Section 8.2, "Path-Based Incremental
	// Target Level Algorithm" in Bertsekas, Nedic, Ozdaglar, "Convex Analysis
	// and Optimization".
	double delta = -1.0;
	double B = 2.0;	// Travel-length bound
	double sigma = 0.0;	// Actual travel-length since last control
	double dual_obj_target = -std::numeric_limits<double>::infinity();
	double dual_obj_best = -std::numeric_limits<double>::infinity();

	// Iterate
	double sol_step = 1.0 / static_cast<double>(T);
	for (int iter = 1; max_iter == 0 || iter <= static_cast<int>(max_iter);
		++iter) {
		// Clear averaged solution
		for (unsigned int vi = 0; vi < var_count; ++vi)
			std::fill(sol_avg[vi].begin(), sol_avg[vi].end(), 0.0);

		// Perform inference for all submodels
		double dual_obj = 0.0;
		bool new_primal_best = false;
		for (unsigned int t = 0; t < T; ++t) {
			double t_obj = tree_inf[t]->MinimizeEnergy(cur_sol_t[t]);

			// Dual objective is simply the sum of all tree objectives
			dual_obj += t_obj;

			// Produce averaged solution (primal infeasible)
			for (unsigned int vi = 0; vi < var_count; ++vi)
				sol_avg[vi][cur_sol_t[t][vi]] += sol_step;

			// Identify best feasible integral labeling
			double t_primal_energy = fg->EvaluateEnergy(cur_sol_t[t]);
			if (t_primal_energy < primal_best_energy) {
				std::copy(cur_sol_t[t].begin(), cur_sol_t[t].end(),
					primal_best.begin());
				primal_best_energy = t_primal_energy;
				new_primal_best = true;
			}
		}

		if (dual_obj > dual_obj_best)
			dual_obj_best = dual_obj;

		// Initial delta: half the primal-dual gap
		if (delta < 0.0)
			delta = 0.5 * (primal_best_energy - dual_obj);

		bool sufficient_descent = false;
		bool oscillation = false;
		if (dual_obj >= (dual_obj_target + 0.5 * delta)) {
			// Sufficient descent because target level is reached
			sufficient_descent = true;
			sigma = 0.0;
			dual_obj_target = dual_obj_best;
		} else if (sigma > B) {
			// Oscillation detected
			oscillation = true;
			sigma = 0.0;
			delta *= 0.5;
			dual_obj_target = dual_obj_best;
		}

#if 0
		// Linear program primal solution recovery by subgradient averaged
		// solution (Anstreicher and Wolsey, MathProg 2009)
#if 1
		// Uniform average
		double new_sol_scale = 1.0 / static_cast<double>(T * iter);
		double old_sol_scale = static_cast<double>(iter - 1) /
			static_cast<double>(iter);
#endif
#if 0
		// Geometric average
		double vol_alpha = 0.005;
		double new_sol_scale = vol_alpha / static_cast<double>(T);
		double old_sol_scale = (1.0 - vol_alpha);
		if (iter == 1) {
			old_sol_scale = 0.0;
			new_sol_scale = 1.0;
		}
#endif
		relaxed_sol_energy = 0.0;
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			std::transform(relaxed_sol[fi].begin(), relaxed_sol[fi].end(),
				relaxed_sol[fi].begin(), _1 * old_sol_scale);

			for (unsigned int t = 0; t < T; ++t) {
				unsigned int ei =
					factors[fi]->ComputeAbsoluteIndex(cur_sol_t[t]);
				assert(ei < relaxed_sol[fi].size());
				relaxed_sol[fi][ei] += new_sol_scale * 1.0;
			}
			relaxed_sol_energy += std::inner_product(
				relaxed_sol[fi].begin(), relaxed_sol[fi].end(),
				factors[fi]->Energies().begin(), 0.0);
		}
#endif

		// Output statistics
		if (verbose) {
			std::cout << "iter " << iter << ", primal " << primal_best_energy
				<< ", dual " << dual_obj << ", best dual " << dual_obj_best
				<< ", gap " << (primal_best_energy - dual_obj) << std::endl;
		}

		// Compute step size
		double subgradient_norm = 0.0;
		for (size_t t = 0; t < T; ++t) {
			for (unsigned int vi = 0; vi < var_count; ++vi) {
				// Obtain a unary factor of the variable
				unsigned int vi_card = var_card[vi];
				for (unsigned int vs = 0; vs < vi_card; ++vs) {
					subgradient_norm += std::pow(
						(cur_sol_t[t][vi] == vs ? 1.0 : 0.0)
						- sol_avg[vi][vs], 2.0);
				}
			}
		}
		//subgradient_norm = std::sqrt(subgradient_norm);
		double gamma = 1.95;
		double alpha = gamma * ((dual_obj_target + delta) - dual_obj) /
			subgradient_norm;
		double dgap = primal_best_energy - dual_obj;
		double convergence_measure = dgap / (fabs(dual_obj) + 1.0e-5);
		sigma += std::min(1.0, alpha * subgradient_norm);
		if (subgradient_norm <= 1.0e-5 || convergence_measure <= conv_tol) {
			if (verbose) {
				std::cout << "Converged, subg norm " << subgradient_norm
					<< ", conv " << convergence_measure << std::endl;
			}
			break;
		}

		// 4. Compute step size
#if 0
		alpha = 1.0 / (10.0 + static_cast<double>(iter));
#endif

		// Update Lagrange multipliers implicitly by directly updating the
		// energies of the trees in the decomposition
		for (unsigned int t = 0; t < T; ++t) {
			const std::vector<Factor*>& factors = trees[t]->FG()->Factors();
			for (unsigned int vi = 0; vi < var_count; ++vi) {
				// Obtain a unary factor of the variable
				unsigned int t_fi = tree_var_to_factor_map[t][vi];
				std::vector<double>& t_fi_energies = factors[t_fi]->Energies();

				// Modify the energies
				unsigned int vi_card = var_card[vi];
				for (unsigned int vs = 0; vs < vi_card; ++vs) {
					// Subgradient update
					t_fi_energies[vs] += alpha * (
						(cur_sol_t[t][vi] == vs ? 1.0 : 0.0) -
						sol_avg[vi][vs]);
				}
			}
		}
	}
}

void LinearProgrammingMAPInference::ClearInferenceResult() {
#if 0
	relaxed_sol.clear();
#endif
}

// Returns one part of the (approximate) relaxed solution
const std::vector<double>& LinearProgrammingMAPInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < relaxed_sol.size());
	return (relaxed_sol[factor_id]);
}

// Returns the relaxed solution
const std::vector<std::vector<double> >&
LinearProgrammingMAPInference::Marginals() const {
	assert(0);
	return (relaxed_sol);
}

// XXX: not implemented
double LinearProgrammingMAPInference::LogPartitionFunction() const {
	assert(false);
	return (std::numeric_limits<double>::signaling_NaN());
}

// XXX: not implemented
void LinearProgrammingMAPInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(false);
}

// Obtain an approximate minimum energy state for the current factor graph
// energies.
double LinearProgrammingMAPInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	PerformInference();
	state = primal_best;

	return (primal_best_energy);
}

}

