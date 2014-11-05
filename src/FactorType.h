
#ifndef GRANTE_FACTORTYPE_H
#define GRANTE_FACTORTYPE_H

#include <vector>
#include <string>

#include <boost/serialization/serialization.hpp>

#include "Factor.h"

namespace Grante {

/* One type of factor in the factor graph.  Although there could be many
 * factors of this type, there is only one type object.
 *
 * The FactorType object contains the number and cardinality of the associated
 * variables as well as possible parameters shared by all factors of this
 * type.
 *
 * There are three main use cases of factor types:
 *
 * 1. A parametrized factor type that has identical energies whenever it is
 * used:
 *    a.) create one factor type (data_size = 0) and store energies in w,
 *    b.) instantiate the factor with empty data arrays.
 *    c.) the replicated energies are the parameters and can be learned.
 *
 * 2. A parameter-free factor type that has identical structure but different
 * energies wherever it is used:
 *    a.) create one factor type with empty w and data_size=prod_card,
 *    b.) instantiate the factor with data array equal to the energies.
 *    c.) there are no parameters learnable.
 *
 * 3. A parametrized data-dependent factor type:
 *    a.) use w: (D_1,D_2,...,D_k,Y_1,Y_2,...,Y_ml) in factor type, with
 *        data_size = D_1*D_2*...*D_k,
 *    b.) use different data: (D_1,D_2,...,D_k) for each factor,
 *    c.) the energy assigned to (y_1,y_2,...,y_m) is the inner product of all
 *        D_i dimensions between the data and the parameter.
 *    d.) the parameters can be learned.
 */
class FactorType {
public:
	// Create a new factor type.
	//
	// name: Textual description of the factor type.
	// card: Adjacent variable node cardinalities.
	// w: The value of the parameter.  w can be empty, in that case the factor
	//    type is parameterless (and cannot be learned).
	// data_size: The length of the data vector of factors instantiating this
	//    type.  If no data_size is provided, then it is determined according
	//    to this logic:
	//    If w.empty()
	//       data_size=prod_card,
	//    else if w.size()==prod_card
	//       data_size=0,
	//    else
	//       data_size=w.size()/prod_card.
	//    end.
	//    This covers all common cases except for data_size=1, in which each
	//    factor has a single scalar by which global weights are scaled.
	FactorType(const std::string& name, const std::vector<unsigned int>& card,
		const std::vector<double>& w);
	FactorType(const std::string& name, const std::vector<unsigned int>& card,
		const std::vector<double>& w, unsigned int data_size);

	virtual ~FactorType();

	// A textual identifier describing the factor
	const std::string& Name() const;

	// Return true if the energies of this factor type depend on data.  Then,
	// each factor instantiation needs its own energy table.
	virtual bool IsDataDependent() const;

	// The parameters associated to this factor type.  After using a
	// ParameterEstimationMethod these weights have been updated to fit the
	// training data.
	virtual std::vector<double>& Weights();
	virtual const std::vector<double>& Weights() const;

	// The number of parameters in the weight vector.  This might be zero if
	// this factor type does not have parameters.
	virtual unsigned int WeightDimension() const;

	// Cardinalities of the variables this factor operates on.
	const std::vector<unsigned int>& Cardinalities() const;

	// Size of associated energy and marginal table.  In the canonical case
	// this is the product of all variable cardinalities but in general it is
	// the number of different state space partitions of this factor and
	// therefore it can be smaller than the product of the cardinalities of
	// its adjacent variables.
	virtual size_t ProdCardinalities() const;

	// Convert a linear index used for energies and marginals into the state
	// of a single variable.
	//
	// ei: The linear index.
	// var_index: The variable index of this factor:
	//    0 <= var_index < Cardinalities.size().
	unsigned int LinearIndexToVariableState(size_t ei, size_t var_index) const;

	// Changes a single index dimension to another value.
	//
	// ei: The linear index to be changed.
	// var_index: The variable index to change value.
	// var_value: The new value of the variable.
	//
	// Returned is the new linear index.
	size_t LinearIndexChangeVariableState(size_t ei,
		unsigned int var_index, unsigned int var_value) const;

	// Compute the absolute index into the energy table for a given state.
	// state: vector of all model variables (not just this factors').
	virtual unsigned int ComputeAbsoluteIndex(const Factor* factor,
		const std::vector<unsigned int>& state) const;

	// Forward map:
	//    Compute energy values from parameters for a specific factor
	//
	// An important special case if w is empty.  Then, the energy values are
	// directly specified by the specific factor data.
	//
	// Can be overwritten to tie parameters in a non-trivial way.
	virtual void ForwardMap(const Factor* factor,
		std::vector<double>& energies) const;

	// Backward map:
	//    Compute parameter gradient from marginals and factor data
	//
	// Can be overwritten to tie parameters in a non-trivial way.  In case
	// ForwardMap is overwritten, BackwardMap must be overwritten as well.
	//
	// mult: Multiplier for all additions to parameter_gradient.
	//
	// The gradient from this factor is added to parameter_gradient.
	virtual void BackwardMap(const Factor* factor,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult = 1.0) const;

	// Compute factor-to-variable message vector of the form,
	//    r_{m->n}(x_n) = log sum_{x_m \ n} exp(
	//       -E(x_m) + sum_{n' \in N(m) \ n} q_{n'->m}(x_{n'}) )
	// or
	//    r_{m->n}(x_n) = max_{x_m \ n} [-E(x_m) +
	//       sum_{n' \in N(m) \ n} q_{n'->m}(x_{n'}) ).
	//
	// The input arguments are as follows.
	//
	// factor: the particular factor (m) in the model,
	// vi: the particular variable (n) in the model,
	// fvi_to: the factor-relative variable index the message is directed to,
	// msglist_for_factor_cur: list of variable-to-factor message indices of
	//    the messages q directed to this factor,
	// msg_for_factor: list of all variable-to-factor messages q, indexed by
	//    indices in msglist_for_factor_cur,
	// msg_for_factor_srcvar: array with source variable index for each
	//    variable-to-factor message index,
	// msg: (output) the factor-to-variable message r_{m->n} to be computed,
	//    must be of the correct size,
	// min_sum: if true, compute the min-sum message, if false compute the
	//    log-sum-exp message.
	//
	// The implementation of this method in the factor type class does not
	// cleanly separate the data structure from the message passing algorithm.
	// However, for higher-order factors the messages must be computed in a
	// different way from the basic implementation and this depends on the
	// specific factor type.
	virtual void ComputeBPMessage(const Factor* factor, unsigned int vi,
		unsigned int fvi_to,
		const std::vector<unsigned int>& msglist_for_factor_cur,
		const std::vector<std::vector<double> >& msg_for_factor,
		const std::vector<unsigned int>& msg_for_factor_srcvar,
		std::vector<double>& msg, bool min_sum) const;

	// Compute marginals for target factor, using
	//   P_f(x) = exp(-E(x) + sum_{var} loq q_{var->f}(x_var) - log_z)
	//
	// Input arguments,
	//
	// factor: the particular factor (m) in the model,
	// msglist_for_factor_cur: list of variable-to-factor message indices of
	//    the messages q directed to this factor,
	// msg_for_factor: list of all variable-to-factor messages q, indexed by
	//    indices in msglist_for_factor_cur,
	// marginal: (output) the marginal distribution computed.  Must be
	//    properly sized.  This size can be different from ProdCardinalities
	//    but it must be possible to pass this distribution to BackwardMap to
	//    compute a gradient.
	// min_sum: if true, compute the min-sum 'marginals', if false compute the
	//    normal log-sum-exp marginals.
	//
	// Return maximum change between existing and new marginal.
	virtual double ComputeBPMarginal(const Factor* factor,
		const std::vector<unsigned int>& msglist_for_factor_cur,
		const std::vector<std::vector<double> >& msg_for_factor,
		std::vector<double>& marginal, bool min_sum) const;

protected:
	// Non-public constructor used for serialization
	FactorType();
	FactorType(const std::string& name, const std::vector<unsigned int>& card,
		unsigned int data_size);

	// A textual identifier
	std::string name;

	// The length of this vector is equal to the number of variables this
	// factor operates on, the elements are equal to the cardinalities of the
	// adjacent variables.  prod_card is the product of all cardinalities.
	std::vector<unsigned int> cardinalities;
	std::vector<unsigned int> prod_cumcard;
	size_t prod_card;

	// True when the factor type depends on factor-specific data.
	bool is_data_dependent;

	// Parameter for this factor type.  Canonically, the weights have the
	// following layout: (D_1,\dots,D_k,Y_1,\dots,Y_m), stored linearly in
	// Matlab-like first-index-cycle-fast layout.
	//
	// To compute the forward map, the first k dimensions are computed as
	// inner product with the factor data H: (D_1,\dots,D_k).
	//
	// In case you simply want to learn an energy table (without data
	// dependencies), w has to be the size of ProdCardinalities().
	std::vector<double> w;

	// The size of the data-vector stored in all Factors instantiating this
	// FactorType.  This is needed for dimensionality-checking as well as
	// infering the correct dimension in the presence of sparse vectors.
	size_t data_size;

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & name;
		ar & cardinalities;
		ar & prod_cumcard;
		ar & prod_card;
		ar & w;
		ar & data_size;
	}

	// Initialize prod_card and prod_cumcard
	void InitializeProdCard();

	// canonical dense version
	void ForwardMap(const std::vector<double>& factor_data,
		std::vector<double>& energies) const;
	// canonical sparse version
	void ForwardMap(const std::vector<double>& factor_data,
		const std::vector<unsigned int>& factor_data_idx,
		std::vector<double>& energies) const;

	// canonical dense version
	void BackwardMap(const std::vector<double>& factor_data,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult) const;
	// canonical sparse version
	void BackwardMap(const std::vector<double>& factor_data,
		const std::vector<unsigned int>& factor_data_idx,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult) const;
};

}

#endif

