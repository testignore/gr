
#ifndef GRANTE_FACTOR_H
#define GRANTE_FACTOR_H

#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "FactorDataSource.h"

namespace Grante {

class FactorType;

/* One specific factor within the factor graph.  The factor is always of a
 * specific type (FactorType), operates on a fixed set of variables and might
 * have additional data associated with it.
 */
class Factor {
public:
	// data: data-dependent part of this factor, relevant for parameter
	//    learning.  If data.size() == 0, then this factor is not
	//    data-dependent.  Using this calling syntax the data is a dense
	//    vector.
	Factor(const FactorType* ftype, const std::vector<unsigned int>& var_index,
		const std::vector<double>& data);

	// Sparse data support: for each i, data[data_idx[i]] = data_elem[i], with
	// all other elements being zero.
	Factor(const FactorType* ftype, const std::vector<unsigned int>& var_index,
		const std::vector<double>& data_elem,
		const std::vector<unsigned int>& data_idx);

	Factor(const FactorType* ftype, const std::vector<unsigned int>& var_index,
		const FactorDataSource* data_source);

	const FactorType* Type() const;

	// Adjacent variables the factor acts on
	const std::vector<unsigned int>& Variables() const;
	const std::vector<unsigned int>& Cardinalities() const;

	const std::vector<double>& Data() const;
	const std::vector<unsigned int>& DataSparseIndex() const;

	// The energies are in Matlab-linearized order: leftmost indices run
	// by one.
	const std::vector<double>& Energies() const;
	std::vector<double>& Energies();

	// TODO: doc
	void EnergiesAllocate(bool force_copy = false);
	void EnergiesRelease();

	// Evaluate the energy with respect to this factor.
	// state: vector of all model variables (not just this factors').
	double EvaluateEnergy(const std::vector<unsigned int>& state) const;

	unsigned int ComputeAbsoluteIndex(
		const std::vector<unsigned int>& state) const;

	// Compute the state[rel_var_index] from the absolute energy index.
	unsigned int ComputeVariableState(size_t abs_index,
		size_t rel_var_index) const;

	// Return the factor-relative index for the given absolute variable index.
	// The absolute variable index must exist in this factor.
	unsigned int AbsoluteVariableIndexToFactorIndex(
		unsigned int abs_var_index) const;

	// Create a conditional marginal distribution from an observation vector
	// (conditioned on) and a distribution of a single variable (var, with
	// distribution var_marginal).  The full distribution is stored in
	// factor_marginal, which is assumed to be of the correct size and
	// initialized to zero.
	void ExpandVariableMarginalToFactorMarginal(
		const std::vector<unsigned int>& state,
		unsigned int var, const std::vector<double>& var_marginal,
		std::vector<double>& factor_marginal) const;

	// Compute energies from data (H) and parameters of the factor.
	// force_copy: If false, we will not copy energies if not necessary.  If
	//    true, we create redundant copies.
	void ForwardMap(bool force_copy = false);

	// See FactorType::BackwardMap
	void BackwardMap(const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult = 1.0) const;

	// Compute the total correlation of a single given factor as a measure of
	// dependence between multiple variables.  This measure can be used to
	// find a better v-acyclic decomposition of the graph.
	//
	// C(X_1,X_2,...,X_k) = \sum_{i} H(X_i) - H(X_1,X_2,...,X_k).
	//
	// The measure is near zero if the variables are independent and large
	// otherwise.
	//
	// The second method also returns the maximum possible total correlation
	// for this factor, i.e. if rval is returned, then 0 <= rval/max_tc <= 1.
	double TotalCorrelation(void) const;
	double TotalCorrelation(double& max_tc) const;

private:
	const FactorType* factor_type;

	// The indexed variables.  This must match the factor type's
	// cardinalities.
	std::vector<unsigned int> var_index;

	// Energies, energies: (Y_1,\dots,Y_m)
	std::vector<double> energies;

	// If non-NULL, the source of the data for this factor (H, H_index are
	// empty then).
	const FactorDataSource* data_source;

	// Factor-specific data, H: (D_1,\dots,D_k)
	std::vector<double> H;
	// If H_index.empty() == false, then H is specified as sparse vector
	std::vector<unsigned int> H_index;

	Factor();

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & const_cast<FactorType* &>(factor_type);
		ar & var_index;
		ar & energies;
		ar & H;
		ar & H_index;
		// FIXME:
		// ar & data_source;
	}
};

}

#endif

