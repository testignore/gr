
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>

#include "LogSumExp.h"
#include "DiffusionInference.h"

namespace Grante {

DiffusionInference::DiffusionInference(const FactorGraph* fg)
	: InferenceMethod(fg), min_sum(false),
	primal_sol_lb(-std::numeric_limits<double>::infinity()),
	verbose(false), max_iter(100), conv_tol(1.0e-5) {
}

DiffusionInference::~DiffusionInference() {
}

InferenceMethod* DiffusionInference::Produce(
	const FactorGraph* fg) const {
	return (new DiffusionInference(fg));
}

void DiffusionInference::PerformInference() {
	if (min_sum == false) {
		PerformInferenceSumProduct();
		return;
	}

	// 1. Setup unary energies
	const std::vector<unsigned int>& var_card = fg->Cardinalities();
	size_t var_count = var_card.size();
	std::vector<std::vector<double> > phi_u(var_count);
	for (size_t vi = 0; vi < var_count; ++vi) {
		phi_u[vi].resize(var_card[vi]);
		std::fill(phi_u[vi].begin(), phi_u[vi].end(), 0.0);
	}

	// 2. Initialize: copy all energies (except unaries)
	// The energies will be modified through the course of the algorithm and
	// we need exactly one unary factor for each variable.  Because this is
	// not guaranteed in the original model, we explicitly represent unary
	// factors (phi_u) and merge all original unary factors into these.
	const std::vector<Factor*>& factors = fg->Factors();
	size_t fac_count = factors.size();
	std::vector<std::vector<double> > phi(fac_count);
	std::vector<unsigned int> phi_minelem(fac_count);
	primal_sol_lb = 0.0;
	for (size_t fi = 0; fi < fac_count; ++fi) {
		const Factor* fac = factors[fi];
		if (fac->Variables().size() > 1) {
			phi[fi] = fac->Energies();
			// Find minimum and add to global lower bound
			phi_minelem[fi] = static_cast<unsigned int>(
				std::min_element(phi[fi].begin(), phi[fi].end())
					- phi[fi].begin());
			primal_sol_lb += phi[fi][phi_minelem[fi]];
			continue;
		}

		// Add to separate unary factor
		unsigned int vi = fac->Variables()[0];
		std::transform(factors[fi]->Energies().begin(),
			factors[fi]->Energies().end(),
			phi_u[vi].begin(), phi_u[vi].begin(),
			std::plus<double>());
	}

	// 3. Construct primal solution (primal_sol and phi_minelem)
	primal_sol.resize(var_count);
	for (size_t vi = 0; vi < var_count; ++vi) {
		primal_sol[vi] = static_cast<unsigned int>(
			std::min_element(phi_u[vi].begin(), phi_u[vi].end())
				- phi_u[vi].begin());
		primal_sol_lb += phi_u[vi][primal_sol[vi]];
	}

	// 4. Perform n-ary min-sum diffusion (Algorithm 1 in Werner CVPR 2008)
	double conv = std::numeric_limits<double>::infinity();
	double lb_prev = -std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; (max_iter == 0 || iter < max_iter) &&
		conv >= conv_tol; ++iter) {
		conv = primal_sol_lb - lb_prev;

		if (verbose) {
			std::cout << "iter " << iter << ", lb " << primal_sol_lb
				<< ", prev lb " << lb_prev << ", conv " << conv << std::endl;
		}
		lb_prev = primal_sol_lb;

		// A factor defines a set of (A,B,.) tripplets, where A is the set of
		// adjacent variables to that factor, and B is a single variable of
		// the factor.  This is a specific choice of J in Section 3.1 of
		// Werner CVPR 2008.  (It is the simplest, but not the strongest
		// possible.)
		for (unsigned int fi = 0; fi < fac_count; ++fi) {
			// Unary factor -> skip
			if (phi[fi].empty())
				continue;

			const FactorType* ftype = factors[fi]->Type();
			const std::vector<unsigned int>& fac_vars =
				factors[fi]->Variables();
			std::vector<double>& tphi = phi[fi];	// \theta_A^{\phi}
			size_t tphi_size = tphi.size();

			for (size_t fvi = 0; fvi < fac_vars.size(); ++fvi) {
				// A: fac_vars, B: fac_vars[fvi]
				unsigned int B = fac_vars[fvi];
				unsigned int card_vi = var_card[B];

				// Initialization with +inf is correct for min-sum
				std::vector<double> msum(card_vi,
					std::numeric_limits<double>::infinity());

				// Find min-marginals/log-sum-exp of A for all x_B
				for (size_t ei = 0; ei < tphi_size; ++ei) {
					unsigned int var_state =
						ftype->LinearIndexToVariableState(ei, fvi);
					// min-marginals
					msum[var_state] = std::min(msum[var_state], tphi[ei]);
				}
				// Compute min-sum/sum-product diffusion update
				primal_sol_lb -= phi_u[B][primal_sol[B]];
				double phi_u_min = std::numeric_limits<double>::infinity();
				unsigned int phi_u_min_idx = 0;

				for (unsigned int var_state = 0; var_state < card_vi;
					++var_state) {
					msum[var_state] = 0.5 * (phi_u[B][var_state]
						- msum[var_state]);
					// Apply update to unary
					phi_u[B][var_state] -= msum[var_state];

					// Keep track of unary min-index
					if (phi_u[B][var_state] < phi_u_min) {
						phi_u_min = phi_u[B][var_state];
						phi_u_min_idx = var_state;
					}
				}
				primal_sol_lb += phi_u[B][phi_u_min_idx];
				primal_sol[B] = phi_u_min_idx;

				// Apply update to factor A
				primal_sol_lb -= tphi[phi_minelem[fi]];
				double phi_min = std::numeric_limits<double>::infinity();
				unsigned int phi_min_idx = 0;

				for (size_t ei = 0; ei < tphi_size; ++ei) {
					unsigned int var_state =
						ftype->LinearIndexToVariableState(ei, fvi);
					tphi[ei] += msum[var_state];

					// Keep track of factor min-index
					if (tphi[ei] < phi_min) {
						phi_min = tphi[ei];
						phi_min_idx = static_cast<unsigned int>(ei);
					}
				}
				primal_sol_lb += tphi[phi_min_idx];
				phi_minelem[fi] = phi_min_idx;
			}
		}
	}
}

void DiffusionInference::PerformInferenceSumProduct() {
	// 1. Setup unary energies
	const std::vector<unsigned int>& var_card = fg->Cardinalities();
	size_t var_count = var_card.size();
	std::vector<std::vector<double> > phi_u(var_count);
	for (size_t vi = 0; vi < var_count; ++vi) {
		phi_u[vi].resize(var_card[vi]);
		std::fill(phi_u[vi].begin(), phi_u[vi].end(), 0.0);
	}

	// 2. Initialize: copy all energies (except unaries) and flip sign
	const std::vector<Factor*>& factors = fg->Factors();
	size_t fac_count = factors.size();
	std::vector<std::vector<double> > phi(fac_count);
	for (size_t fi = 0; fi < fac_count; ++fi) {
		const Factor* fac = factors[fi];
		if (fac->Variables().size() > 1) {
			phi[fi] = fac->Energies();
			for (size_t ei = 0; ei < phi[fi].size(); ++ei)
				phi[fi][ei] *= -1.0;	// flip sign
			continue;
		}

		// Add to separate unary factor (sign-flipped)
		unsigned int vi = fac->Variables()[0];
		for (size_t vsi = 0; vsi < phi_u[vi].size(); ++vsi)
			phi_u[vi][vsi] -= factors[fi]->Energies()[vsi];
	}

	// 4. Perform n-ary min-sum diffusion (Algorithm 1 in Werner CVPR 2008)
	double conv = std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; (max_iter == 0 || iter < max_iter) &&
		conv >= conv_tol; ++iter) {
		if (verbose) {
			// double F = ComputeSumProductObjective(phi, phi_u);
			std::cout << "iter " << iter << ", conv " << conv
				<< std::endl;
		}
		conv = 0.0;

		// A factor defines a set of (A,B,.) tripplets, where A is the set of
		// adjacent variables to that factor, and B is a single variable of
		// the factor.  This is a specific choice of J in Section 3.1 of
		// Werner CVPR 2008.  (It is the simplest, but not the strongest
		// possible.)
		for (size_t fi = 0; fi < fac_count; ++fi) {
			// Unary factor -> skip
			if (phi[fi].empty())
				continue;

			const FactorType* ftype = factors[fi]->Type();
			const std::vector<unsigned int>& fac_vars =
				factors[fi]->Variables();
			std::vector<double>& tphi = phi[fi];	// \theta_A^{\phi}
			size_t tphi_size = tphi.size();

			for (size_t fvi = 0; fvi < fac_vars.size(); ++fvi) {
				// A: fac_vars, B: fac_vars[fvi]
				unsigned int B = fac_vars[fvi];
				unsigned int card_vi = var_card[B];

				// Initialization with +inf is correct for min-sum, -inf for
				// sum-product.
				//    min-sum: computing min-marginals,
				//    sum-product: computing factor log-sum-exp.
				std::vector<double> msum(card_vi,
					-std::numeric_limits<double>::infinity());

				// Find min-marginals/log-sum-exp of A for all x_B
				for (size_t ei = 0; ei < tphi_size; ++ei) {
					unsigned int var_state =
						ftype->LinearIndexToVariableState(ei, fvi);
					// log-sum-exp
					msum[var_state] = std::log(
						std::exp(msum[var_state]) + std::exp(tphi[ei]));
				}
				// Compute min-sum/sum-product diffusion update
				for (size_t var_state = 0; var_state < card_vi;
					++var_state) {
					msum[var_state] = 0.5 * (phi_u[B][var_state]
						- msum[var_state]);
					conv += std::fabs(msum[var_state]);

					// Apply update to unary
					phi_u[B][var_state] -= msum[var_state];
				}

				// Apply update to factor A
				// sum-product
				for (size_t ei = 0; ei < tphi_size; ++ei) {
					unsigned int var_state =
						ftype->LinearIndexToVariableState(ei, fvi);
					tphi[ei] += msum[var_state];
				}
			}
		}
	}

	// Produce marginals and log_z
	log_z = 0.0;
	marginals.resize(fac_count);
	for (size_t fi = 0; fi < fac_count; ++fi) {
		const Factor* fac = factors[fi];
		marginals[fi].resize(fac->Type()->ProdCardinalities());
		std::vector<double>& phi_cur = (fac->Variables().size() > 1) ?
			phi[fi] : phi_u[fac->Variables()[0]];

		// Compute local log-partition function
		double cur_logz = LogSumExp::Compute(phi_cur);

		log_z += cur_logz;
		for (size_t ei = 0; ei < phi_cur.size(); ++ei)
			marginals[fi][ei] = std::exp(phi_cur[ei] - cur_logz);
	}
}

void DiffusionInference::ClearInferenceResult() {
	marginals.clear();
	log_z = std::numeric_limits<double>::signaling_NaN();
}

void DiffusionInference::SetParameters(bool verbose,
	unsigned int max_iter, double conv_tol) {
	this->verbose = verbose;
	this->max_iter = max_iter;
	assert(conv_tol >= 0.0);
	this->conv_tol = conv_tol;
}

const std::vector<double>& DiffusionInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >&
DiffusionInference::Marginals() const {
	return (marginals);
}

double DiffusionInference::LogPartitionFunction() const {
	return (log_z);
}

void DiffusionInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(0);
}

double DiffusionInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	min_sum = true;
	PerformInference();
	min_sum = false;
	state = primal_sol;

	return (fg->EvaluateEnergy(state));
}

double DiffusionInference::ComputeSumProductObjective(
	const std::vector<std::vector<double> >& phi,
	const std::vector<std::vector<double> >& phi_u) const {
	double F = 0.0;
	const std::vector<Factor*>& factors = fg->Factors();
	size_t fac_count = factors.size();
	for (size_t fi = 0; fi < fac_count; ++fi) {
		const Factor* fac = factors[fi];
		const std::vector<double>& phi_cur = (fac->Variables().size() > 1) ?
			phi[fi] : phi_u[fac->Variables()[0]];

		F += LogSumExp::Compute(phi_cur);
	}
	return (F);
}

}

