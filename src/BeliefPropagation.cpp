
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <iostream>
#include <cmath>
#include <cassert>

#include <boost/lambda/lambda.hpp>

#include "BeliefPropagation.h"
#include "LogSumExp.h"

using namespace boost::lambda;

namespace Grante {

BeliefPropagation::BeliefPropagation(const FactorGraph* fg,
	MessageSchedule sched)
	: InferenceMethod(fg), verbose(false), max_iter(100), conv_tol(1.0e-5),
		sched(sched), min_sum(false),
		log_z(std::numeric_limits<double>::quiet_NaN())
{
	// Setup message indices
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const std::vector<unsigned int>& fac_vars = factors[fi]->Variables();
		for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
			// 1. Message from variable to factor
			unsigned int msg_id =
				static_cast<unsigned int>(msg_for_factor_srcvar.size());
			msglist_for_factor[fi].push_back(msg_id);
			msg_for_factor_srcvar.push_back(fac_vars[fvi]);

			// 2. Message from factor to variable
			msg_id = static_cast<unsigned int>(msg_for_var_srcfactor.size());
			msglist_for_var[fac_vars[fvi]].push_back(msg_id);
			msg_for_var_srcfactor.push_back(fi);
		}
	}

	// If using a sequential schedule, obtain a message order and id's
	if (sched == Sequential) {
		FactorGraphStructurizer::ComputeEulerianMessageTrail(fg, order);
		order_msgid.resize(order.size());

		for (size_t oi = 0; oi < order.size(); ++oi) {
			unsigned int vi = order[oi].VariableNode();
			unsigned int fi = order[oi].FactorNode();
			if (order[oi].steptype == FactorGraphStructurizer::LeafIsFactorNode) {
				// message: fi -> vi
				msg_list_t::const_iterator mli = msglist_for_var.find(vi);
				assert(mli != msglist_for_var.end());
				const std::vector<unsigned int>& msg_list = mli->second;
				for (unsigned int mi = 0; mi < msg_list.size(); ++mi) {
					if (msg_for_var_srcfactor[msg_list[mi]] != fi)
						continue;

					order_msgid[oi] = msg_list[mi];
					break;
				}
			} else {
				// message: vi -> fi
				msg_list_t::const_iterator mli = msglist_for_factor.find(fi);
				assert(mli != msglist_for_factor.end());
				const std::vector<unsigned int>& msg_list = mli->second;
				for (unsigned int mi = 0; mi < msg_list.size(); ++mi) {
					if (msg_for_factor_srcvar[msg_list[mi]] != vi)
						continue;

					order_msgid[oi] = msg_list[mi];
					break;
				}
			}
		}
	}
}

BeliefPropagation::~BeliefPropagation() {
}

InferenceMethod* BeliefPropagation::Produce(const FactorGraph* fg) const {
	return (new BeliefPropagation(fg));
}

void BeliefPropagation::SetParameters(bool verbose,
	unsigned int max_iter, double conv_tol) {
	this->verbose = verbose;
	this->max_iter = max_iter;
	assert(conv_tol >= 0.0);
	this->conv_tol = conv_tol;
}

void BeliefPropagation::PerformInference() {
	InferenceInitialize();

	// Min-sum variables
	double best_energy = std::numeric_limits<double>::infinity();

	// Perform message passing
	double conv_measure = std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; (max_iter == 0 || iter <= max_iter) &&
		conv_measure >= conv_tol; ++iter)
	{
		if (verbose) {
			std::cout << "iter " << iter << ", conv " << conv_measure;
			if (min_sum)
				std::cout << ", E* " << best_energy;
			std::cout << std::endl;
		}

		if (sched == ParallelSync) {
			PerformInferenceStepParallel();
		} else if (sched == Sequential) {
			PerformInferenceSequential();
		} else {
			assert(0);
		}

		// Convergence measure: maximum update to marginals
		// TODO
		if (min_sum) {
			conv_measure = ComputeVariableBeliefs();
			std::vector<unsigned int> cur_state(fg->Cardinalities().size());
			double cur_energy = ReconstructMinimumEnergyState(cur_state);
			if (cur_energy < best_energy) {
				best_energy = cur_energy;
				best_state = cur_state;
			}
		} else {
			conv_measure = ConstructMarginals();
		}
	}
	ConstructMarginals();
	ComputeVariableBeliefs();
	if (min_sum) {
		if (verbose) {
			std::cout << "Converged, tol " << conv_measure << ", E* "
				<< best_energy << std::endl;
		}
	} else {
		log_z = -ComputeBetheFreeEnergy();
		if (verbose) {
			std::cout << "Converged, tol " << conv_measure
				<< ", log_z(Bethe) " << log_z << std::endl;
		}
	}

	InferenceTeardown();
}

void BeliefPropagation::InferenceInitialize() {
	// Initialize messages
	const std::vector<unsigned int>& card = fg->Cardinalities();
	//  i) factor-to-variable
	msg_for_var.resize(msg_for_var_srcfactor.size());
	for (unsigned int vi = 0; vi < card.size(); ++vi) {
		msg_list_t::const_iterator mli = msglist_for_var.find(vi);
		assert(mli != msglist_for_var.end());
		const std::vector<unsigned int>& ml = mli->second;

		for (unsigned int mli = 0; mli < ml.size(); ++mli) {
			assert(ml[mli] < msg_for_var.size());

			// Resize and fill (value does not matter for parallel schedule,
			// but does for sequential)
			msg_for_var[ml[mli]].resize(card[vi]);
			std::fill(msg_for_var[ml[mli]].begin(),
				msg_for_var[ml[mli]].end(), 0.0);
		}
	}
	// ii) variable-to-factor
	msg_for_factor.resize(msg_for_factor_srcvar.size());
	for (unsigned int mfi = 0; mfi < msg_for_factor_srcvar.size(); ++mfi) {
		unsigned int vi = msg_for_factor_srcvar[mfi];
		assert(vi < card.size());
		// Resize and fill with log(1) = 0.
		msg_for_factor[mfi].resize(card[vi]);
		std::fill(msg_for_factor[mfi].begin(), msg_for_factor[mfi].end(), 0.0);
	}

	// Initialize marginals (beliefs)
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}
}

void BeliefPropagation::InferenceTeardown() {
	// Clear messages
	msg_for_var.clear();
	msg_for_factor.clear();
}

void BeliefPropagation::ClearInferenceResult() {
	marginals.clear();
	var_beliefs.clear();
}

const std::vector<double>& BeliefPropagation::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >& BeliefPropagation::Marginals() const
{
	return (marginals);
}

double BeliefPropagation::LogPartitionFunction() const {
	return (log_z);
}

// NOT IMPLEMENTED
void BeliefPropagation::Sample(std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(0);
}

double BeliefPropagation::MinimizeEnergy(std::vector<unsigned int>& state) {
	min_sum = true;
	PerformInference();
	state.resize(fg->Cardinalities().size());
	std::copy(best_state.begin(), best_state.end(), state.begin());
	min_sum = false;
	return (fg->EvaluateEnergy(state));
}


void BeliefPropagation::PerformInferenceStepParallel() {
	// 1. factor-to-variable
	PassFactorToVariable();
	// 2. variable-to-factor
	PassVariableToFactor();
}

void BeliefPropagation::PerformInferenceSequential() {
	// Perform one sequential pass over all messages
	for (unsigned int oi = 0; oi < order.size(); ++oi) {
		unsigned int vi = order[oi].VariableNode();
		unsigned int fi = order[oi].FactorNode();
		unsigned int msg_id = order_msgid[oi];

		if (order[oi].steptype == FactorGraphStructurizer::LeafIsFactorNode) {
			// message: fi -> vi
			const Factor* factor = fg->Factors()[fi];
			PassFactorToVariable(factor, vi, msg_for_var[msg_id],
				msglist_for_factor[fi]);
		} else {
			// message: vi -> fi
			PassVariableToFactor(fi, msg_for_factor[msg_id], vi);
		}
	}
}

void BeliefPropagation::PassFactorToVariable() {
	const std::vector<Factor*>& factors = fg->Factors();
	const std::vector<unsigned int>& card = fg->Cardinalities();

	// For each variable
	for (unsigned int vi = 0; vi < card.size(); ++vi) {
		msg_list_t::const_iterator mli = msglist_for_var.find(vi);
		assert(mli != msglist_for_var.end());
		const std::vector<unsigned int>& ml = mli->second;

		// For each factor connected to that variable:
		// Send message from factor to variable
		for (unsigned int mli = 0; mli < ml.size(); ++mli) {
			assert(ml[mli] < msg_for_var.size());

			unsigned int factor_index = msg_for_var_srcfactor[ml[mli]];
			const Factor* factor = factors[factor_index];
			PassFactorToVariable(factor, vi, msg_for_var[ml[mli]],
				msglist_for_factor[factor_index]);
		}
	}
}

// Single message variant
void BeliefPropagation::PassFactorToVariable(const Factor* factor,
	unsigned int vi, std::vector<double>& msg,
	const std::vector<unsigned int>& msglist_for_factor_cur) {
	// Obtain type and adjacent variables of the factor
	const FactorType* ftype = factor->Type();

	const std::vector<unsigned int>& fvars = factor->Variables();
	unsigned int fvi_to = static_cast<unsigned int>(
		std::find(fvars.begin(), fvars.end(), vi) - fvars.begin());

	// Target message to be computed
	// r_{m->n}(x_n) = log sum_{x_m \ n} exp(
	//    -E(x_m) + sum_{n' \in N(m) \ n} q_{n'->m}(x_{n'}) )
	std::fill(msg.begin(), msg.end(), 0.0);

	// Compute the message within the factor type
	ftype->ComputeBPMessage(factor, vi, fvi_to,
		msglist_for_factor_cur, msg_for_factor,
		msg_for_factor_srcvar, msg, min_sum);
}

void BeliefPropagation::PassVariableToFactor() {
	const std::vector<Factor*>& factors = fg->Factors();
	// For all factors
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// Obtain messages directed to factor fi
		msg_list_t::const_iterator mli = msglist_for_factor.find(fi);
		assert(mli != msglist_for_factor.end());
		const std::vector<unsigned int>& ml = mli->second;

		// For all adjacent variables
		for (unsigned int mli = 0; mli < ml.size(); ++mli) {
			assert(ml[mli] < msg_for_factor.size());

			std::vector<double>& msg = msg_for_factor[ml[mli]];
			unsigned int from_var = msg_for_factor_srcvar[ml[mli]];
			PassVariableToFactor(fi, msg, from_var);
		}
	}
}

void BeliefPropagation::PassVariableToFactor(unsigned int fi,
	std::vector<double>& msg, unsigned int from_var) {
	// Target message to be computed
	//    q_{n->m}(x_n) = sum_{m' \in M(n) \ m} r_{m'->n}(x_n),
	// (26.11) McKay, in log-domain.
	std::fill(msg.begin(), msg.end(), 0.0);
	msg_list_t::const_iterator mvi = msglist_for_var.find(from_var);
	assert(mvi != msglist_for_var.end());
	const std::vector<unsigned int>& mliv = mvi->second;
	for (std::vector<unsigned int>::const_iterator fvi = mliv.begin();
		fvi != mliv.end(); ++fvi) {
		unsigned int for_var_msg_index = *fvi;
		// Skip messages from the target factor
		if (msg_for_var_srcfactor[for_var_msg_index] == fi)
			continue;

		// Add log-sum messages, (26.11)
		std::transform(msg_for_var[for_var_msg_index].begin(),
			msg_for_var[for_var_msg_index].end(),
			msg.begin(), msg.begin(), std::plus<double>());
	}

	// Normalization for numerical stability,
	//   i) sum-product: log-sum-exp = 0,
	//  ii) min-sum: sum = 0.
	double norm_delta = min_sum ?
		(std::accumulate(msg.begin(), msg.end(), 0.0) /
			static_cast<double>(msg.size()))
		: LogSumExp::Compute(msg);
	std::transform(msg.begin(), msg.end(), msg.begin(), _1 - norm_delta);
}

double BeliefPropagation::ConstructMarginals() {
	const std::vector<Factor*>& factors = fg->Factors();
	// Compute mean and variance of log_z estimate
	double marg_max_diff = -std::numeric_limits<double>::infinity();

	// Compute marginals of all factors
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const Factor* factor = factors[fi];
		const FactorType* ftype = factor->Type();

		// Obtain messages directed to factor fi
		msg_list_t::const_iterator mli = msglist_for_factor.find(fi);
		assert(mli != msglist_for_factor.end());
		const std::vector<unsigned int>& msglist_for_factor_cur = mli->second;

		double cur_marg_max_diff = ftype->ComputeBPMarginal(factor,
			msglist_for_factor_cur, msg_for_factor, marginals[fi], min_sum);
		if (cur_marg_max_diff > marg_max_diff)
			marg_max_diff = cur_marg_max_diff;

	}

	return (marg_max_diff);
}

double BeliefPropagation::ComputeVariableBeliefs() {
	const std::vector<unsigned int>& card = fg->Cardinalities();
	var_beliefs.resize(card.size());

	// For each variable
	double max_change = -std::numeric_limits<double>::infinity();
	for (unsigned int vi = 0; vi < card.size(); ++vi) {
		// Initialize beliefs
		std::vector<double> var_belief_vi_old(var_beliefs[vi]);
		var_beliefs[vi].resize(card[vi]);
		std::fill(var_beliefs[vi].begin(), var_beliefs[vi].end(), 0.0);

		// For each message directed to variable vi
		msg_list_t::const_iterator mli = msglist_for_var.find(vi);
		assert(mli != msglist_for_var.end());
		const std::vector<unsigned int>& ml = mli->second;
		for (unsigned int mli = 0; mli < ml.size(); ++mli) {
			assert(ml[mli] < msg_for_var.size());
			std::vector<double>& msg = msg_for_var[ml[mli]];

			// sum_{m \in M(vi)} log r_{m->n}(x_n)
			assert(msg.size() == card[vi]);
			std::transform(msg.begin(), msg.end(), var_beliefs[vi].begin(),
				var_beliefs[vi].begin(), std::plus<double>());
		}

		// Compute normalized variable marginal (belief)
		double Z_vi = min_sum ?
			(std::accumulate(var_beliefs[vi].begin(),
				var_beliefs[vi].end(), 0.0) /
				static_cast<double>(var_beliefs[vi].size()))
			: LogSumExp::Compute(var_beliefs[vi]);
		for (unsigned int vs = 0; vs < var_beliefs[vi].size(); ++vs) {
			if (min_sum) {
				var_beliefs[vi][vs] -= Z_vi;
			} else {
				var_beliefs[vi][vs] = std::exp(var_beliefs[vi][vs] - Z_vi);
			}
			if (var_belief_vi_old.empty()) {
				max_change = std::numeric_limits<double>::infinity();
			} else {
				max_change = std::max(max_change,
					std::fabs(var_beliefs[vi][vs] - var_belief_vi_old[vs]));
			}
		}
	}
	return (max_change);
}

double BeliefPropagation::ReconstructMinimumEnergyState(
	std::vector<unsigned int>& state) const {
	const std::vector<unsigned int>& card = fg->Cardinalities();
	assert(state.size() == card.size());
	assert(var_beliefs.size() == card.size());

	for (size_t vi = 0; vi < card.size(); ++vi) {
		assert(var_beliefs[vi].size() == card[vi]);
		state[vi] = static_cast<unsigned int>(std::max_element(
			var_beliefs[vi].begin(), var_beliefs[vi].end())
			- var_beliefs[vi].begin());
	}
	return (fg->EvaluateEnergy(state));
}

// (3.45) in [Wainwright and Jordan], "A(theta) = negative free energy"
// Theorem 5 in [Yedidia, Freeman, and Weiss], "Interior stationary points of
// the constrained Bethe free energy must be BP fixed points".
// (37) in [Yedidia, Freeman, and Weiss] gives the Bethe free energy.  This is
// an approximation to the negative log partition function logZ.
double BeliefPropagation::ComputeBetheFreeEnergy() const {
	double U_Bethe = 0.0;	// Bethe average energy
	double H_Bethe = 0.0;	// Bethe entropy
	const std::vector<Factor*>& factors = fg->Factors();
	const std::vector<unsigned int>& card = fg->Cardinalities();
	std::vector<unsigned int> var_degree(card.size(), 0);
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const Factor* factor = factors[fi];
		const std::vector<double>& energies = factor->Energies();
		size_t energies_size = energies.size();
		for (size_t ei = 0; ei < energies_size; ++ei) {
			U_Bethe += -marginals[fi][ei] * (-energies[ei]);
			H_Bethe += -marginals[fi][ei] * std::log(marginals[fi][ei]);
		}

		// Increase degrees of all variables involved in this factor
		const std::vector<unsigned int>& fac_vars = factor->Variables();
		for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi)
			var_degree[fac_vars[fvi]] += 1;
	}

	size_t var_count = card.size();
	for (size_t vi = 0; vi < var_count; ++vi) {
		assert(var_degree[vi] >= 1);
		assert(var_beliefs[vi].size() == card[vi]);
		double corr = 0.0;
		for (unsigned int state = 0; state < card[vi]; ++state)
			corr += var_beliefs[vi][state] * std::log(var_beliefs[vi][state]);
		H_Bethe += static_cast<double>(var_degree[vi] - 1) * corr;
	}

	// Return the Bethe free energy, log Z = -Bethe = -(U-H)
	return (U_Bethe - H_Bethe);
}

}

