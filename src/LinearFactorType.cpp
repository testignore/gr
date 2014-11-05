
#include <algorithm>
#include <set>
#include <map>
#include <cassert>

#include "LinearFactorType.h"

namespace Grante {

LinearFactorType::LinearFactorType(const std::string& name,
	const std::vector<unsigned int>& card,
	const std::vector<double>& w, unsigned int data_size,
	const std::vector<int>& A)
	: FactorType(name, card, data_size), A(A), total_a(0) {
	this->w = w;
	assert(w.empty() == false);
	assert(A.size() == ProdCardinalities());

	// Precompute index maps
	origin.resize(prod_card);
	std::set<int> A_ids;
	std::map<int, int> origin_map;
	for (unsigned int ei = 0; ei < A.size(); ++ei) {
		if (A[ei] == -1)
			continue;

		assert(A[ei] >= 0);
		A_ids.insert(A[ei]);
		total_a = std::max(total_a, static_cast<unsigned int>(A[ei]));

		// Keep track of the first ei index with the tying class
		if (origin_map.count(A[ei]) == 0)
			origin_map.insert(std::pair<int, int>(A[ei], ei));
		origin[ei] = origin_map[A[ei]];
	}
	total_a += 1;
	assert(total_a == A_ids.size());

	// Check A and size of w match
	if (data_size == 0) {
		assert(total_a == w.size());
		is_data_dependent = false;
	} else {
		assert(total_a*data_size == w.size());
	}
}

bool LinearFactorType::IsDataDependent() const {
	// This is needed because we need to expand the compact/sparse/tied
	// representation to a full energy table
	return (true);
}

void LinearFactorType::ForwardMap(const Factor* factor,
	std::vector<double>& energies) const {
	const std::vector<double>& H = factor->Data();
	const std::vector<unsigned int>& H_index = factor->DataSparseIndex();

	if (H_index.empty()) {
		// Dense
		ForwardMap(H, energies);
	} else {
		// Sparse
		// TODO
		//ForwardMap(H, H_index, energies);
		assert(0);
	}
}

void LinearFactorType::BackwardMap(const Factor* factor,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	const std::vector<double>& H = factor->Data();
	const std::vector<unsigned int>& H_index = factor->DataSparseIndex();

	if (H_index.empty()) {
		// Dense
		BackwardMap(H, marginals, parameter_gradient, mult);
	} else {
		// Sparse
		// TODO
		//BackwardMap(H, H_index, marginals, parameter_gradient, mult);
		assert(0);
	}
}

// private: dense general linear version
void LinearFactorType::ForwardMap(const std::vector<double>& factor_data,
	std::vector<double>& energies) const {
	assert(energies.size() == prod_card);

	// Perform mode-n vector product (inner product) for all data dimensions
	assert(data_size == factor_data.size());
	assert(w.empty() == false);
	if (data_size == 0) {
		// No data dependencies: parameter is the energy table
		assert(w.size() == total_a);

		// Fill in energies by expanding parameters
		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			// Sparse element?
			if (A[ei] == -1) {
				energies[ei] = 0.0;
			} else {
				energies[ei] = w[A[ei]];
			}
		}
	} else {
		assert((data_size * total_a) == w.size());
		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			// Sparse element?
			if (A[ei] == -1) {
				energies[ei] = 0.0;
				continue;
			}

			// Only need to copy?
			unsigned int origin_ei = origin[ei];
			if (origin_ei != ei) {
				energies[ei] = energies[origin_ei];
				continue;
			}

			// Need to compute
			double energy_cur = 0.0;
			for (unsigned int di = 0; di < data_size; ++di)
				energy_cur += factor_data[di] * w[di + A[ei]*data_size];

			energies[ei] = energy_cur;
		}
	}
}

// private: dense general linear map
void LinearFactorType::BackwardMap(const std::vector<double>& factor_data,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	// Obtain factor data
	assert(data_size == factor_data.size());
	assert(w.empty() == false);
	if (data_size == 0) {
		// Parameters are a simple table, gradient is simply the marginal,
		// again marginalized by projection
		assert(total_a == parameter_gradient.size());

		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			// Sparse elements have no gradient
			if (A[ei] == -1)
				continue;

			assert(static_cast<unsigned int>(A[ei]) < parameter_gradient.size());
			parameter_gradient[A[ei]] += mult * marginals[ei];
		}
	} else {
		assert((data_size * total_a) == parameter_gradient.size());

		// Perform tensor outer product
		for (unsigned int ei = 0; ei < prod_card; ++ei) {
			for (unsigned int di = 0; di < data_size; ++di) {
				// Sparse elements have no gradient
				if (A[ei] == -1)
					continue;

				// \nabla_w(di,y_1,\dots,y_k) = H(di) marg(y_1,\dots,y_k)
				parameter_gradient[di + A[ei]*data_size] +=
					mult * factor_data[di] * marginals[ei];
			}
		}
	}
}


}

