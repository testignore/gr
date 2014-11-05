
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <tr1/unordered_map>
#include <ctime>
#include <cmath>
#include <cassert>

#include "TreeInference.h"
#include "LogSumExp.h"
#include "FactorGraphStructurizer.h"

namespace Grante {

TreeInference::TreeInference(const FactorGraph* fg)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::quiet_NaN()),
		rgen(static_cast<const boost::uint32_t>(std::time(0))+1),
		randu(rgen, rdestu)
{
	// Null pointer argument: do nothing
	if (fg == 0)
		return;

	assert(FactorGraphStructurizer::IsForestStructured(fg));

	// Precompute leaf-to-root order once
	FactorGraphStructurizer::ComputeTreeOrder(fg, leaf_to_root, tree_roots);

	// Precompute:
	//   1. All leaf-to-root messages directed to a factor
	//   2. All leaf-to-root messages directed to a variable
	for (unsigned int lri = 0; lri < leaf_to_root.size(); ++lri) {
		if (leaf_to_root[lri].steptype ==
			FactorGraphStructurizer::LeafIsFactorNode) {
			// Variable 'root' receives message from factor 'leaf'
			ltr_msg_for_var[leaf_to_root[lri].root].insert(lri);
			ltr_factor_toroot[leaf_to_root[lri].leaf] = lri;
		} else {
			// Factor 'root' receives message from variable 'leaf'
			ltr_var_toroot[leaf_to_root[lri].leaf] = lri;
		}
	}
}

TreeInference::~TreeInference() {
}

InferenceMethod* TreeInference::Produce(const FactorGraph* fg) const {
	return (new TreeInference(fg));
}

void TreeInference::PerformInference() {
	// Messages and reverse-messages are defined along all edges of the factor
	// graph
	std::vector<std::vector<double> > msg;
	std::vector<std::vector<double> > msg_rev;

	// Marginals are defined for all factors
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}

	// Messages along leaf-to-root order, then root-to-leaf
	PassLeafToRoot(msg);
	PassRootToLeaf(msg, msg_rev);
}

void TreeInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& TreeInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >& TreeInference::Marginals() const {
	return (marginals);
}

double TreeInference::LogPartitionFunction() const {
	return (log_z);
}

double TreeInference::Entropy() const {
	const std::vector<Factor*>& factors = fg->Factors();
	assert(marginals.size() == factors.size());
	double H = 0.0;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		H += std::inner_product(marginals[fi].begin(), marginals[fi].end(),
			factors[fi]->Energies().begin(), 0.0);
	}
#if 0
	std::cout << "<mu,theta> (" << H << ") - logZ (" << log_z << ") = "
		<< (H-log_z) << std::endl;
#endif
	H += log_z;
	return (H);
}

void TreeInference::Sample(std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(sample_count > 0);

	// Obtain leaf-to-root messages once
	std::vector<std::vector<double> > msg;
	PassLeafToRoot(msg);

	// Initialize samples
	states.resize(sample_count);
	size_t var_count = fg->Cardinalities().size();
	std::vector<std::vector<double> > msg_rev(msg);
	for (unsigned int si = 0; si < sample_count; ++si) {
		// Always re-use the same leaf-to-root messages, but the root-to-leaf
		// messages change with each sample
		states[si].resize(var_count);
		PassRootToLeaf(msg, msg_rev, states[si]);
	}
}

double TreeInference::MinimizeEnergy(std::vector<unsigned int>& state) {
	// Obtain leaf-to-root messages once
	std::vector<std::vector<double> > msg;
	PassLeafToRoot(msg, true);

	size_t var_count = fg->Cardinalities().size();
	state.resize(var_count);

	std::vector<std::vector<double> > msg_rev(msg);
	PassRootToLeaf(msg, msg_rev, state, true);

	// Return energy (this could be made faster by using the max-product
	// result)
	if (std::fabs(fg->EvaluateEnergy(state) + log_z) > 1.0e-6) {
		std::cout << "WARNING: min-sum computed energy and factor graph energy "
			<< "disagree" << std::endl;
		std::cout << "  EvaluateEnergy: " << fg->EvaluateEnergy(state) << std::endl;
		std::cout << "  -log_z: " << -log_z << std::endl;
	}
	assert(std::fabs(fg->EvaluateEnergy(state) + log_z) <= 1.0e-2);
	return (-log_z);
}

unsigned int TreeInference::SampleConditionalUnnormalized(
	const std::vector<double>& cond_unnorm) const {
	double Z = std::accumulate(cond_unnorm.begin(), cond_unnorm.end(), 0.0);
	double rand_val = Z * randu();

	// Sample from conditional distribution
	double cumsum = 0.0;
	for (unsigned int state = 0; state < cond_unnorm.size(); ++state) {
		assert(cond_unnorm[state] >= 0.0);
		cumsum += cond_unnorm[state];
		if (rand_val <= cumsum)
			return (state);
	}
	assert(false);

	// This should never happen
	return (std::numeric_limits<unsigned int>::max());
}

unsigned int TreeInference::MaximizeConditionalUnnormalized(
	const std::vector<double>& cond_unnorm) const {
	return (static_cast<unsigned int>(
		std::max_element(cond_unnorm.begin(), cond_unnorm.end()) -
			cond_unnorm.begin()));
}

void TreeInference::PassLeafToRoot(std::vector<std::vector<double> >& msg,
	bool min_sum) {
	// Important lookup variables
	const std::vector<unsigned int>& card = fg->Cardinalities();
	const std::vector<Factor*>& factors = fg->Factors();

	// Initialize messages
	msg.resize(leaf_to_root.size());
	for (unsigned int lri = 0; lri < leaf_to_root.size(); ++lri) {
		msg[lri].resize(card[leaf_to_root[lri].VariableNode()]);
		std::fill(msg[lri].begin(), msg[lri].end(), 0.0);
	}

	// Follow the precomputed order for speed
	for (unsigned int lri = 0; lri < leaf_to_root.size(); ++lri) {
		if (leaf_to_root[lri].steptype ==
			FactorGraphStructurizer::LeafIsFactorNode) {
			// Factor-to-variable message
			unsigned int factor_index = leaf_to_root[lri].leaf;
			unsigned int up_var_index = leaf_to_root[lri].root;

			const Factor* factor = factors[factor_index];
			const FactorType* ftype = factor->Type();

			const std::vector<unsigned int>& fvars = factor->Variables();
			// Absolute variable index of the upward message variable
			unsigned int fvi_up = static_cast<unsigned int>(
				std::find(fvars.begin(), fvars.end(), up_var_index) -
				fvars.begin());

			// Setup summation on the same table as the energies
			const std::vector<double>& energies = factor->Energies();
			std::vector<double> msum_xn(energies.size());

			// For every setting of x_n, compute message by marginalizing out
			// all other variables of the factor.
			//
			// (26.12) in McKay, but in log-domain.
			size_t energies_size = energies.size();
			std::vector<double> msum_xn_max(card[up_var_index],
				-std::numeric_limits<double>::infinity());
			for (size_t ei = 0; ei < energies_size; ++ei) {
				msum_xn[ei] = -energies[ei];

				// Sum adjacent leaf-variables of this factor
				for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
					if (fvi == fvi_up)
						continue;	// Upward variable

					unsigned int var_index = fvars[fvi];
					unsigned int var_msg = ltr_var_toroot[var_index];
					unsigned int var_state =
						ftype->LinearIndexToVariableState(ei, fvi);

					// + log q_{v->f}(v_state)
					msum_xn[ei] += msg[var_msg][var_state];
				}

				// Compute maximum over state of xn for stable log-sum-exp.
				unsigned int xn_state =
					ftype->LinearIndexToVariableState(ei, fvi_up);
				assert(xn_state < msum_xn_max.size());
				if (msum_xn[ei] > msum_xn_max[xn_state])
					msum_xn_max[xn_state] = msum_xn[ei];
			}

			if (min_sum) {
				// Message: maximum negative energy value (minimum energy)
				// over domain of xn
				for (unsigned int xn_state = 0; xn_state < card[up_var_index];
					++xn_state) {
					msg[lri][xn_state] = msum_xn_max[xn_state];
				}
			} else {
				// Log-sum-exp (numerically stable), correctly split along msum_xn
				for (unsigned int ei = 0; ei < energies_size; ++ei) {
					unsigned int xn_state =
						ftype->LinearIndexToVariableState(ei, fvi_up);
					msg[lri][xn_state] +=
						std::exp(msum_xn[ei] - msum_xn_max[xn_state]);
				}
				for (unsigned int xn_state = 0; xn_state < card[up_var_index];
					++xn_state) {
					msg[lri][xn_state] = msum_xn_max[xn_state] +
						std::log(msg[lri][xn_state]);
				}
			}
		} else {
			// Variable-to-factor message
			unsigned int var_index = leaf_to_root[lri].leaf;

			// Obtain all children factor-to-var messages directed to this variable
			const std::set<unsigned int>& msg_for_var =
				ltr_msg_for_var[var_index];

			// Log-sum them (product), storing them directly in msg[lri]
			// (26.11) in McKay, but in log-domain
			// Note: here, min-sum is the same as sum-product in the log
			// domain.
			for (std::set<unsigned int>::const_iterator
				mi = msg_for_var.begin(); mi != msg_for_var.end(); ++mi) {
				std::transform(msg[*mi].begin(), msg[*mi].end(),
					msg[lri].begin(), msg[lri].begin(),
					std::plus<double>());
			}
		}
	}

	// Compute log-partition function:
	//    log Z = sum of factor-to-variable-0 messages of all tree roots
	log_z = 0;
	for (std::tr1::unordered_set<unsigned int>::const_iterator
		tri = tree_roots.begin(); tri != tree_roots.end(); ++tri) {
		const std::set<unsigned int>& msg_for_x0 = ltr_msg_for_var[*tri];
		std::vector<double> log_z_sum(card[*tri], 0.0);
		for (std::set<unsigned int>::const_iterator mzi = msg_for_x0.begin();
			mzi != msg_for_x0.end(); ++mzi) {
			// Sum over factors
			std::transform(msg[*mzi].begin(), msg[*mzi].end(),
				log_z_sum.begin(), log_z_sum.begin(), std::plus<double>());
		}

		if (min_sum) {
			// Maximum negative energy
			log_z += *std::max_element(log_z_sum.begin(), log_z_sum.end());
		} else {
			// Log-sum-exp over states
			log_z += LogSumExp::Compute(log_z_sum);
		}
	}
}

void TreeInference::PassRootToLeaf(std::vector<std::vector<double> >& msg,
	std::vector<std::vector<double> >& msg_rev) {
	std::vector<unsigned int> sample_dummy;
	PassRootToLeaf(msg, msg_rev, sample_dummy);
}

void TreeInference::PassRootToLeaf(std::vector<std::vector<double> >& msg,
	std::vector<std::vector<double> >& msg_rev,
	std::vector<unsigned int>& sample, bool min_sum) {
	// Important lookup variables
	const std::vector<unsigned int>& card = fg->Cardinalities();
	const std::vector<Factor*>& factors = fg->Factors();

	// If we should produce a sample, sample the root variable of the tree (as
	// it does not have a factor-to-variable message.
	if (sample.empty() == false) {
		assert(sample.size() == card.size());
		// Initialize with ::max() value to know when sampling has occured.
		// This is important for correctly sampling higher-order (>= 3)
		// factors.
		std::fill(sample.begin(), sample.end(),
			std::numeric_limits<unsigned int>::max());
	}

	// Initialize reverse messages
	msg_rev.resize(msg.size());
	for (unsigned int lri = 0; lri < leaf_to_root.size(); ++lri) {
		msg_rev[lri].resize(msg[lri].size());
		std::fill(msg_rev[lri].begin(), msg_rev[lri].end(), 0.0);
	}

	// Sample the tree root(s):
	// When the tree-root is sampled beforehand, the sampling algorithm
	// simplifies because we can always assume that in a root-to-leaf
	// variable-to-factor message the variable is already sampled.
	if (sample.empty() == false) {
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			tri = tree_roots.begin(); tri != tree_roots.end(); ++tri) {
			std::vector<double> m_root(card[*tri], 0.0);

			const std::set<unsigned int>& msg_for_root =
				ltr_msg_for_var[*tri];
			for (std::set<unsigned int>::const_iterator
				mi = msg_for_root.begin(); mi != msg_for_root.end(); ++mi) {
				std::transform(msg[*mi].begin(), msg[*mi].end(),
					m_root.begin(), m_root.begin(),
					std::plus<double>());
			}
			if (min_sum == false) {
				for (unsigned int n = 0; n < m_root.size(); ++n)
					m_root[n] = std::exp(m_root[n]);
			}

			// Maximize/sample from marginal distribution
			sample[*tri] = min_sum ?
				MaximizeConditionalUnnormalized(m_root)
				: SampleConditionalUnnormalized(m_root);
		}
	}

	// Reversely follow the precomputed order: root-to-leaf.
	// Each message enables a new marginal computation, but we only keep
	// marginals of factors.
	for (int lri = static_cast<int>(leaf_to_root.size())-1; lri >= 0; --lri) {
		if (leaf_to_root[lri].steptype ==
			FactorGraphStructurizer::LeafIsFactorNode) {
			// PART 1: Compute variable-to-factor root-to-leaf message:
			//         q_{var->f}
			unsigned int factor_index = leaf_to_root[lri].leaf;
			unsigned int var_index = leaf_to_root[lri].root;

			// Information about factor structure
			const Factor* factor = factors[factor_index];
			const FactorType* ftype = factor->Type();

			// (26.11) in McKay
			// q_{var->f} = r_{fromroot->var}
			//    + sum_{g in F(var) \ {f}} r_{g->var}
			if (tree_roots.count(var_index) == 0) {	// not root: has parent factor
				// Obtain parent factor's message to this variable
				unsigned int msg_from_root = ltr_var_toroot[var_index];
				if (sample.empty()) {
					// r_{fromroot->var}
					std::copy(msg_rev[msg_from_root].begin(),
						msg_rev[msg_from_root].end(), msg_rev[lri].begin());
				} else {
					assert(sample[var_index] !=
						std::numeric_limits<unsigned int>::max());

					// sampling: condition on variable state
					std::fill(msg_rev[lri].begin(), msg_rev[lri].end(),
						msg_rev[msg_from_root][sample[var_index]]);
				}
			}

			// Sum them, storing them directly in msg_rev[lri]
			// similar to (26.11) in McKay, but in log-domain
			const std::set<unsigned int>& msg_for_var =
				ltr_msg_for_var[var_index];
			for (std::set<unsigned int>::const_iterator
				mi = msg_for_var.begin(); mi != msg_for_var.end(); ++mi) {
				// \ {f}, the reverse message of this (lri) message
				if (leaf_to_root[*mi].leaf == factor_index)
					continue;

				// + r_{g->var}
				if (sample.empty()) {
					std::transform(msg[*mi].begin(), msg[*mi].end(),
						msg_rev[lri].begin(), msg_rev[lri].begin(),
						std::plus<double>());
				} else {
					// Messages through the variable node get conditioned on
					// the sampled variable state
					assert(sample[var_index] !=
						std::numeric_limits<unsigned int>::max());
					for (unsigned int n = 0; n < msg_rev[lri].size(); ++n)
						msg_rev[lri][n] += msg[*mi][sample[var_index]];
				}
			}

			// PART 2: Compute marginals of the factor

			// Absolute variable index of the upward message variable
			const std::vector<unsigned int>& fvars = factor->Variables();
			unsigned int fvi_up = static_cast<unsigned int>(
				std::find(fvars.begin(), fvars.end(), var_index) -
				fvars.begin());

			// Energies and marginals
			const std::vector<double>& energies = factor->Energies();
			std::vector<double> M_temp;
			if (sample.empty() == false) {
				M_temp.resize(energies.size());
				std::fill(M_temp.begin(), M_temp.end(), 0.0);
			}
			std::vector<double>& M = sample.empty() ?
				marginals[factor_index] : M_temp;
			assert(M.size() == energies.size());

			// Compute marginals for target factor:
			//   P_f(x) = exp(-E(x) + sum_{var} loq q_{var->f}(x_var) - log_z)
			size_t energies_size = energies.size();
			for (size_t ei = 0; ei < energies_size; ++ei) {
				if (sample.empty()) {
					assert(min_sum == false);
					M[ei] = -energies[ei];
				} else {
					// For sampling, the energy is always conditioned on one
					// variable
					assert(sample[var_index] !=
						std::numeric_limits<unsigned int>::max());
					M[ei] = -energies[ftype->LinearIndexChangeVariableState(
						ei, fvi_up, sample[var_index])];
				}

				// Sum adjacent leaf-variables of this factor
				for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
					if (fvi == fvi_up) {
						// Variable coming from root: use reverse message
						unsigned int vup_state =
							ftype->LinearIndexToVariableState(ei, fvi_up);

						// Condition
						if (sample.empty() == false &&
							sample[var_index] !=
								std::numeric_limits<unsigned int>::max())
						{
							vup_state = sample[var_index];
						}

						M[ei] += msg_rev[lri][vup_state];
						continue;
					}

					// + log q_{v->f}(v_state)
					unsigned int mvar_index = fvars[fvi];
					unsigned int mvar_msg = ltr_var_toroot[mvar_index];
					unsigned int mvar_state =
						ftype->LinearIndexToVariableState(ei, fvi);

					M[ei] += msg[mvar_msg][mvar_state];
				}
			}
			// Normalization is only required for sampling or when there are
			// multiple tree roots
			if (min_sum == false) {
				double log_z_fi = LogSumExp::Compute(M);
				for (unsigned int ei = 0; ei < energies_size; ++ei)
					M[ei] = std::exp(M[ei] - log_z_fi);
			}

			// Sample all adjacent not-yet-sampled variables
			if (sample.empty() == false) {
				unsigned int ei_sample = min_sum ?
					MaximizeConditionalUnnormalized(M)
					: SampleConditionalUnnormalized(M);
				for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
					// Already sampled -> skip
					unsigned int mvar_index = fvars[fvi];
					if (sample[mvar_index] !=
						std::numeric_limits<unsigned int>::max())
						continue;

					// Sample variable
					unsigned int mvar_state =
						ftype->LinearIndexToVariableState(ei_sample, fvi);
					sample[mvar_index] = mvar_state;
				}
			}
		} else {
			// Factor-to-variable message (26.12) in McKay
			//   r_{f->var}(x) = log sum_{x_{m\n}} exp{
			//      -E(x) + sum_{var} loq q_{var->f}(x_var) }
			unsigned int var_index = leaf_to_root[lri].leaf;
			unsigned int factor_index = leaf_to_root[lri].root;

			const Factor* factor = factors[factor_index];
			const FactorType* ftype = factor->Type();

			// Absolute variable index of the upward message variable
			unsigned int msg_from_root = ltr_factor_toroot[factor_index];
			unsigned int var_from_root = leaf_to_root[msg_from_root].root;
			const std::vector<unsigned int> fvars = factor->Variables();
			// fvi_down: relative index of the downward message variable
			unsigned int fvi_down = static_cast<unsigned int>(
				std::find(fvars.begin(), fvars.end(), var_index) -
				fvars.begin());

			// Setup summation on the same table as the energies
			const std::vector<double>& energies = factor->Energies();
			std::vector<double> msum_xn(energies.size());

			// For every setting of x_n, compute message by marginalizing out
			// all other variables of the factor.
			//
			// (26.12) in McKay, but in log-domain.
			size_t energies_size = energies.size();
			std::vector<double> msum_xn_max(card[var_index],
				-std::numeric_limits<double>::infinity());
			for (size_t ei = 0; ei < energies_size; ++ei) {
				msum_xn[ei] = -energies[ei];

				// Sum adjacent leaf-variables of this factor
				for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
					if (fvi == fvi_down)
						continue;

					unsigned int var_index = fvars[fvi];
					if (var_index == var_from_root) {
						if (sample.empty()) {
							// From-above variable
							unsigned int xroot_state =
								ftype->LinearIndexToVariableState(ei, fvi);
							msum_xn[ei] += msg_rev[msg_from_root][xroot_state];
						} else {
							// sampling: condition on root (always sampled)
							assert(sample[var_from_root] !=
								std::numeric_limits<unsigned int>::max());
							msum_xn[ei] +=
								msg_rev[msg_from_root][sample[var_from_root]];
						}
						continue;
					}

					// Find leaf-to-root variable-to-factor message
					unsigned int var_msg = ltr_var_toroot[var_index];
					unsigned int var_state =
						ftype->LinearIndexToVariableState(ei, fvi);
					if (sample.empty() == false) {
						// All adjacent variables must have been sampled
						assert(sample[var_index] !=
							std::numeric_limits<unsigned int>::max());

						// Condition on already-sampled variable.
						// (This only kicks in for higher-order factors.)
						var_state = sample[var_index];
					}

					// + log q_{v->f}(v_state)
					msum_xn[ei] += msg[var_msg][var_state];
				}

				// Compute maximum over state of xn
				unsigned int xn_state =
					ftype->LinearIndexToVariableState(ei, fvi_down);
				if (msum_xn[ei] > msum_xn_max[xn_state])
					msum_xn_max[xn_state] = msum_xn[ei];
			}

			if (min_sum) {
				// Message: maximum negative energy value
				for (unsigned int xn_state = 0; xn_state < card[var_index];
					++xn_state) {
					msg_rev[lri][xn_state] = msum_xn_max[xn_state];
				}
			} else {
				// Log-sum-exp, correctly split along msum_xn
				for (unsigned int ei = 0; ei < energies_size; ++ei) {
					unsigned int xn_state =
						ftype->LinearIndexToVariableState(ei, fvi_down);
					msg_rev[lri][xn_state] +=
						std::exp(msum_xn[ei] - msum_xn_max[xn_state]);
				}
				for (unsigned int xn_state = 0; xn_state < card[var_index];
					++xn_state) {
					msg_rev[lri][xn_state] = msum_xn_max[xn_state] +
						std::log(msg_rev[lri][xn_state]);
				}
			}
		}
	}
}

}

