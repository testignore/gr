
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <cassert>

#include "FactorType.h"
#include "LogSumExp.h"

namespace Grante {

FactorType::FactorType() {
}

FactorType::FactorType(const std::string& name,
	const std::vector<unsigned int>& card, const std::vector<double>& w)
	: name(name), cardinalities(card), is_data_dependent(true), w(w), data_size(0) {
	assert(cardinalities.size() > 0);
	InitializeProdCard();

	// This shadows one (unlikely) use case: data_size=1, in which a single
	// factor-specific value is used to scale global parameters.
	if (w.empty()) {
		data_size = prod_card;
	} else if (w.size() == prod_card) {
		data_size = 0;
		is_data_dependent = false;
	} else {
		assert(w.size() % prod_card == 0);
		data_size = w.size() / prod_card;
	}
}

FactorType::FactorType(const std::string& name,
	const std::vector<unsigned int>& card, const std::vector<double>& w,
	unsigned int data_size)
	: name(name), cardinalities(card), prod_cumcard(card.size()),
		prod_card(1), is_data_dependent(true), w(w), data_size(data_size) {
	assert(cardinalities.size() > 0);

	// Compute linearized length
	for (size_t n = 0; n < cardinalities.size(); ++n) {
		prod_cumcard[n] = static_cast<unsigned int>(prod_card);
		prod_card *= cardinalities[n];
	}
	if (data_size > 0) {
		assert((data_size * prod_card) == w.size());
	} else {
		is_data_dependent = false;
	}
}

FactorType::FactorType(const std::string& name,
	const std::vector<unsigned int>& card, unsigned int data_size)
	: name(name), cardinalities(card), is_data_dependent(true),
	data_size(data_size) {
	assert(cardinalities.size() > 0);
	InitializeProdCard();
}

FactorType::~FactorType() {
}

const std::string& FactorType::Name() const {
	return (name);
}

bool FactorType::IsDataDependent() const {
	return (is_data_dependent);
}

std::vector<double>& FactorType::Weights() {
	return (w);
}

const std::vector<double>& FactorType::Weights() const {
	return (w);
}

unsigned int FactorType::WeightDimension() const {
	return (static_cast<unsigned int>(w.size()));
}

const std::vector<unsigned int>& FactorType::Cardinalities() const {
	return (cardinalities);
}

size_t FactorType::ProdCardinalities() const {
	return (prod_card);
}

unsigned int FactorType::LinearIndexToVariableState(size_t ei,
	size_t var_index) const {
	return ((ei / prod_cumcard[var_index]) % cardinalities[var_index]);
}

size_t FactorType::LinearIndexChangeVariableState(size_t ei,
	unsigned int var_index, unsigned int var_value) const {
	assert(var_value < cardinalities[var_index]);
	return (ei + var_value*prod_cumcard[var_index]
		- LinearIndexToVariableState(ei, var_index)*prod_cumcard[var_index]);
}

unsigned int FactorType::ComputeAbsoluteIndex(
	const Factor* factor, const std::vector<unsigned int>& state) const {
	unsigned int idx = 0;

	unsigned int stride = 1;
	const std::vector<unsigned int>& var_index = factor->Variables();
	for (unsigned int vi = 0; vi < var_index.size(); ++vi) {
		unsigned int cur_var = var_index[vi];
		assert(state[cur_var] < cardinalities[vi]);

		idx += stride * state[cur_var];
		stride *= cardinalities[vi];
	}
	assert(idx < prod_card);
	return (idx);
}

void FactorType::InitializeProdCard() {
	// Compute linearized length
	prod_card = 1;
	prod_cumcard.resize(cardinalities.size());
	for (size_t n = 0; n < cardinalities.size(); ++n) {
		prod_cumcard[n] = static_cast<unsigned int>(prod_card);
		prod_card *= cardinalities[n];
	}
}

void FactorType::ForwardMap(const Factor* factor,
	std::vector<double>& energies) const {
	const std::vector<double>& H = factor->Data();
	const std::vector<unsigned int>& H_index = factor->DataSparseIndex();

	if (H_index.empty()) {
		// Dense or general user-implemented
		ForwardMap(H, energies);
	} else {
		// Sparse
		ForwardMap(H, H_index, energies);
	}
}

// private: canonical dense version
void FactorType::ForwardMap(const std::vector<double>& factor_data,
	std::vector<double>& energies) const {
	assert(energies.size() == prod_card);

	// Perform mode-n vector product (inner product) for all data dimensions
	assert(data_size == factor_data.size());
	if (data_size == 0) {
		// No data dependencies: parameter is the energy table
		//
		// This is an important case for replicating the same identical factor
		// many times.  This case should be rare: normally in this case we do
		// not copy the energies and ForwardMap will never be called.  For
		// some cases (taking weighted subgraphs) it is still necessary.
		assert(prod_card == w.size());
		assert(w.size() == energies.size());
		std::copy(w.begin(), w.end(), energies.begin());
	} else if (w.empty()) {
		// No data dependencies: data is the energy table
		//
		// This is an important case for replicating the same factor structure
		// many times with different parameter-free energies.
		assert(prod_card == data_size);
		assert(factor_data.size() == energies.size());
		std::copy(factor_data.begin(), factor_data.end(), energies.begin());
	} else {
		assert((data_size * prod_card) == w.size());
		assert(energies.size() == prod_card);
		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			double energy_cur = 0.0;
			for (unsigned int di = 0; di < data_size; ++di)
				energy_cur += factor_data[di] * w[di + ei*data_size];

			energies[ei] = energy_cur;
		}
	}
}

// private: canonical sparse version
// Sparse data support: for any i, v[factor_data_idx[i]] = factor_data[i], all
// other elements are zero.  The true length of the vector is
// factor_data_size.
void FactorType::ForwardMap(const std::vector<double>& factor_data,
	const std::vector<unsigned int>& factor_data_idx,
	std::vector<double>& energies) const {
	assert(energies.size() == prod_card);
	assert(factor_data.size() == factor_data_idx.size());
	assert(data_size >= factor_data_idx.size());

	if (data_size == 0) {
		assert(prod_card == w.size());
		std::copy(w.begin(), w.end(), energies.begin());
	} else if (w.empty()) {
		// Unlikely to happen, but still handle this case: energy table is
		// given by sparse vector.  Therefore, we perform a simple
		// sparse-to-dense expansion.
		assert(prod_card == data_size);
		std::fill(energies.begin(), energies.end(), 0.0);
		for (unsigned int n = 0; n < factor_data_idx.size(); ++n)
			energies[factor_data_idx[n]] = factor_data[n];
	} else {
		// Data dependencies, perform inner product
		assert((data_size * prod_card) == w.size());

		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			double energy_cur = 0.0;
			for (unsigned int n = 0; n < factor_data_idx.size(); ++n) {
				energy_cur += factor_data[n] *
					w[factor_data_idx[n] + ei*data_size];
			}
			energies[ei] = energy_cur;
		}
	}
}

void FactorType::BackwardMap(const Factor* factor,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	const std::vector<double>& H = factor->Data();
	const std::vector<unsigned int>& H_index = factor->DataSparseIndex();

	if (H_index.empty()) {
		// Dense canonical
		BackwardMap(H, marginals, parameter_gradient, mult);
	} else {
		// Sparse canonical
		BackwardMap(H, H_index, marginals, parameter_gradient, mult);
	}
}

// private: canonical dense map
void FactorType::BackwardMap(const std::vector<double>& factor_data,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	// Obtain factor data
	assert(data_size == factor_data.size());
	if (data_size == 0) {
		// Parameters are a simple table, gradient is simply the marginal
		assert(prod_card == parameter_gradient.size());

		for (unsigned int ei = 0; ei < prod_card; ++ei)
			parameter_gradient[ei] += mult * marginals[ei];
	} else if (w.empty()) {
		// No parameters
		assert(parameter_gradient.empty());
	} else {
		assert((data_size * prod_card) == parameter_gradient.size());

		// Perform tensor outer product
		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			for (unsigned int di = 0; di < data_size; ++di) {
				// \nabla_w(di,y_1,\dots,y_k) = H(di) marg(y_1,\dots,y_k)
				parameter_gradient[di + ei*data_size] +=
					mult * factor_data[di] * marginals[ei];
			}
		}
	}
}

// private: canonical sparse map
void FactorType::BackwardMap(const std::vector<double>& factor_data,
	const std::vector<unsigned int>& factor_data_idx,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	// The first two cases are the same as for the non-sparse case
	if (data_size == 0) {
		assert(prod_card == parameter_gradient.size());
		for (unsigned int ei = 0; ei < prod_card; ++ei)
			parameter_gradient[ei] += mult * marginals[ei];
	} else if (w.empty()) {
		assert(parameter_gradient.empty());
	} else {
		assert((data_size * prod_card) == parameter_gradient.size());

		// Perform tensor outer product
		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			for (unsigned int n = 0; n < factor_data_idx.size(); ++n) {
				unsigned int di = factor_data_idx[n];
				parameter_gradient[di + ei*data_size] +=
					mult * factor_data[n] * marginals[ei];
			}
		}
	}
}

void FactorType::ComputeBPMessage(const Factor* factor,
	unsigned int vi, unsigned int fvi_to,
	const std::vector<unsigned int>& msglist_for_factor_cur,
	const std::vector<std::vector<double> >& msg_for_factor,
	const std::vector<unsigned int>& msg_for_factor_srcvar,
	std::vector<double>& msg, bool min_sum) const {
	// Obtain basic tables
	const std::vector<double>& energies = factor->Energies();
	size_t energies_size = energies.size();
	std::vector<double> msum_xn(energies_size);
	size_t card_vi = msg.size();
	std::vector<double> msum_xn_max(card_vi,
		-std::numeric_limits<double>::infinity());

	// For x_n
	for (size_t ei = 0; ei < energies_size; ++ei) {
		msum_xn[ei] = -energies[ei];

		// Sum adjacent variables of this factor
		for (size_t fvi = 0; fvi < msglist_for_factor_cur.size(); ++fvi) {
			// The messages are ordered according to the factor
			// variable order, that is fvi is the factor-relative
			// variable index.
			unsigned int var_msg_index = msglist_for_factor_cur[fvi];
			unsigned int var_index = msg_for_factor_srcvar[var_msg_index];

			// Skip over message from this variable
			if (var_index == vi)
				continue;

			// + log q_{v->f}(v_state)
			unsigned int var_state = LinearIndexToVariableState(ei, fvi);
			msum_xn[ei] += msg_for_factor[var_msg_index][var_state];
		}

		// Compute maximum over state of xn for stable log-sum-exp.
		unsigned int xn_state = LinearIndexToVariableState(ei, fvi_to);
		if (msum_xn[ei] > msum_xn_max[xn_state])
			msum_xn_max[xn_state] = msum_xn[ei];
	}

	std::fill(msg.begin(), msg.end(), 0.0);
	if (min_sum) {
		// Message: maximum negative energy value (minimum energy)
		// over domain of xn
		std::copy(msum_xn_max.begin(), msum_xn_max.end(), msg.begin());
	} else {
		// Log-sum-exp (numerically stable), correctly split along msum_xn
		for (size_t ei = 0; ei < energies_size; ++ei) {
			unsigned int xn_state = LinearIndexToVariableState(ei, fvi_to);
			msg[xn_state] += std::exp(msum_xn[ei] - msum_xn_max[xn_state]);
		}
		for (size_t xn_state = 0; xn_state < card_vi; ++xn_state)
			msg[xn_state] = msum_xn_max[xn_state] + std::log(msg[xn_state]);
	}
}

double FactorType::ComputeBPMarginal(const Factor* factor,
	const std::vector<unsigned int>& msglist_for_factor_cur,
	const std::vector<std::vector<double> >& msg_for_factor,
	std::vector<double>& marginal, bool min_sum) const {
	// Energies and marginals
	const std::vector<double>& energies = factor->Energies();
	std::vector<double> M(marginal.size(), 0.0);
	assert(M.size() == energies.size());

	// Compute marginals for target factor:
	//   P_f(x) = exp(-E(x) + sum_{var} loq q_{var->f}(x_var) - log_z)
	size_t energies_size = energies.size();
	double marg_max_diff = -std::numeric_limits<double>::infinity();
	for (size_t ei = 0; ei < energies_size; ++ei) {
		M[ei] = -energies[ei];

		for (size_t mli = 0; mli < msglist_for_factor_cur.size(); ++mli) {
			assert(msglist_for_factor_cur[mli] < msg_for_factor.size());

			// + log q_{v->f}(v_state)
			const std::vector<double>& msg =
				msg_for_factor[msglist_for_factor_cur[mli]];
			unsigned int from_var_state =
				LinearIndexToVariableState(ei, mli);
			M[ei] += msg[from_var_state];
		}
	}

	double z_fi = min_sum ? (std::accumulate(M.begin(), M.end(), 0.0)
		/ static_cast<double>(M.size()))
		: LogSumExp::Compute(M);
	for (size_t ei = 0; ei < energies_size; ++ei) {
		if (min_sum) {
			M[ei] -= z_fi;
		} else {
			M[ei] = std::exp(M[ei] - z_fi);
		}

		// Keep track of the maximum marginal change
		double diff_M = std::fabs(M[ei] - marginal[ei]);
		if (diff_M > marg_max_diff)
			marg_max_diff = diff_M;
		marginal[ei] = M[ei];
	}
	return (marg_max_diff);
}

}

