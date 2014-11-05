
#include <numeric>
#include <iostream>
#include <limits>
#include <cmath>
#include <cassert>

#include "Factor.h"
#include "LogSumExp.h"
#include "FactorType.h"

namespace Grante {

// private
Factor::Factor()
	: factor_type(0), data_source(0) {
}

Factor::Factor(const FactorType* ftype,
	const std::vector<unsigned int>& var_index,
	const std::vector<double>& data)
	: factor_type(ftype), var_index(var_index), data_source(0), H(data) {
	assert(ftype != 0);
	assert(ftype->Cardinalities().size() == var_index.size());

	// If this factor is data-dependent, allocate energies
	// TODO make this lazy
	if (ftype->IsDataDependent())
		energies.resize(ftype->ProdCardinalities());
}

Factor::Factor(const FactorType* ftype,
	const std::vector<unsigned int>& var_index,
	const std::vector<double>& data_elem,
	const std::vector<unsigned int>& data_idx)
	: factor_type(ftype), var_index(var_index), data_source(0),
	H(data_elem), H_index(data_idx) {
	assert(ftype != 0);
	assert(ftype->Cardinalities().size() == var_index.size());
	assert(data_elem.size() == data_idx.size());

	if (ftype->IsDataDependent())
		energies.resize(ftype->ProdCardinalities());
}

Factor::Factor(const FactorType* ftype,
	const std::vector<unsigned int>& var_index,
	const FactorDataSource* data_source)
	: factor_type(ftype), var_index(var_index), data_source(data_source) {
	assert(ftype != 0);
	assert(ftype->Cardinalities().size() == var_index.size());
	assert(data_source != 0);

	if (ftype->IsDataDependent())
		energies.resize(ftype->ProdCardinalities());
}

const FactorType* Factor::Type() const {
	return (factor_type);
}

const std::vector<unsigned int>& Factor::Variables() const {
	return (var_index);
}

const std::vector<unsigned int>& Factor::Cardinalities() const {
	return (factor_type->Cardinalities());
}

const std::vector<double>& Factor::Data() const {
	if (data_source != 0)
		return (data_source->Data());

	return (H);
}

const std::vector<unsigned int>& Factor::DataSparseIndex() const {
	if (data_source != 0)
		return (data_source->DataSparseIndex());

	return (H_index);
}

const std::vector<double>& Factor::Energies() const {
	// For "parametrized factor type that has identical energies whenever it
	// is used": pass factor type energies
	if (factor_type->IsDataDependent() == false && energies.empty()) {
		const std::vector<double>& ft_energies = factor_type->Weights();
		assert(ft_energies.size() == factor_type->ProdCardinalities());
		return (ft_energies);
	}
	return (energies);
}

std::vector<double>& Factor::Energies() {
	// Data-independent factors: return factor type energies
	if (factor_type->IsDataDependent() == false && energies.empty()) {
		std::vector<double>& ft_energies =
			const_cast<FactorType*>(factor_type)->Weights();
		assert(ft_energies.size() == factor_type->ProdCardinalities());
		return (ft_energies);
	}
	return (energies);
}

void Factor::EnergiesAllocate(bool force_copy) {
	if (factor_type->IsDataDependent() || force_copy)
		energies.resize(factor_type->ProdCardinalities());
}

void Factor::EnergiesRelease() {
	energies.clear();
}

double Factor::EvaluateEnergy(const std::vector<unsigned int>& state) const {
	return (Energies()[factor_type->ComputeAbsoluteIndex(this, state)]);
}

unsigned int Factor::ComputeAbsoluteIndex(
	const std::vector<unsigned int>& state) const {
	return (factor_type->ComputeAbsoluteIndex(this, state));
}

unsigned int Factor::ComputeVariableState(size_t abs_index,
	size_t rel_var_index) const {
	assert(abs_index < Energies().size());
	assert(rel_var_index < var_index.size());

	unsigned int stride = 1;
	const std::vector<unsigned int>& card =
		factor_type->Cardinalities();
	for (unsigned int vi = 0; vi < rel_var_index; ++vi)
		stride *= card[vi];

	abs_index /= stride;
	abs_index %= card[rel_var_index];

	return (static_cast<unsigned int>(abs_index));
}

unsigned int Factor::AbsoluteVariableIndexToFactorIndex(
	unsigned int abs_var_index) const {
	for (unsigned int vi = 0; vi < var_index.size(); ++vi) {
		if (var_index[vi] == abs_var_index)
			return (vi);
	}
	assert(0);
	return (std::numeric_limits<unsigned int>::max());
}

void Factor::ExpandVariableMarginalToFactorMarginal(
	const std::vector<unsigned int>& state,
	unsigned int abs_var, const std::vector<double>& var_marginal,
	std::vector<double>& factor_marginal) const {
	const std::vector<unsigned int>& card =
		factor_type->Cardinalities();
	assert(factor_marginal.size() == factor_type->ProdCardinalities());

	// Condition on state, except for the variable var
	for (unsigned int var_state = 0; var_state < var_marginal.size();
		++var_state) {
		unsigned int stride = 1;
		unsigned int idx = 0;
		for (unsigned int vi = 0; vi < var_index.size(); ++vi) {
			unsigned int cur_var = var_index[vi];
			if (cur_var == abs_var)
				idx += stride * var_state;
			else
				idx += stride * state[cur_var];

			stride *= card[vi];
		}
		factor_marginal[idx] = var_marginal[var_state];
	}
}

void Factor::ForwardMap(bool force_copy) {
	// If the factor does not depend on data, then we do not need to evaluate
	// the energies
	if (factor_type->IsDataDependent() == false && force_copy == false) {
		assert(energies.empty());
		return;
	}

	// For some factor types the size of the energy table is determined only
	// after an initialization step from training data.
	if (energies.empty())
		energies.resize(factor_type->ProdCardinalities());

	factor_type->ForwardMap(this, energies);
}

void Factor::BackwardMap(const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	factor_type->BackwardMap(this, marginals, parameter_gradient, mult);
}

double Factor::TotalCorrelation(void) const {
	double max_tc = 0.0;
	return (TotalCorrelation(max_tc));
}

double Factor::TotalCorrelation(double& max_tc) const {
	const FactorType* ftype = Type();
	const std::vector<double>& E = Energies();
	assert(E.size() == ftype->ProdCardinalities());
	double log_z_fac = LogSumExp::ComputeNeg(E);
	const std::vector<unsigned int>& fac_vars = Variables();

	// -p(y) log p(y)
	double Hjoint = 0.0;
	std::vector<std::vector<double> > Psep(fac_vars.size());
	for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
		Psep[fvi].resize(ftype->Cardinalities()[fvi]);
		std::fill(Psep[fvi].begin(), Psep[fvi].end(), 0.0);
	}

	for (unsigned int ei = 0; ei < E.size(); ++ei) {
		// Joint entropy
		Hjoint += - std::exp(-E[ei] - log_z_fac) * (-E[ei]-log_z_fac);

		// Accumulate marginal probabilities
		for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
			unsigned int var_state =
				ftype->LinearIndexToVariableState(ei, fvi);
			Psep[fvi][var_state] += std::exp(-E[ei] - log_z_fac);
		}
	}

	// Compute and sum marginal entropies
	double Hsep = 0.0;
	max_tc = 0.0;
	for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
		double Hsep_cur = 0.0;
		for (unsigned int var_state = 0; var_state < Psep[fvi].size();
			++var_state) {
			assert(Psep[fvi][var_state] > 0.0);
			Hsep_cur += -Psep[fvi][var_state] * std::log(Psep[fvi][var_state]);
		}
		if (Hsep_cur > max_tc)
			max_tc = Hsep_cur;

		Hsep += Hsep_cur;
	}
	assert(Hsep >= 0.0);
	assert(Hjoint >= 0.0);
	max_tc = Hsep - max_tc;
	if ((Hsep - Hjoint) > max_tc) {
		for (unsigned int ei = 0; ei < E.size(); ++ei)
			std::cout << E[ei] << " ";
		std::cout << std::endl;
		std::cout << "Hsep " << Hsep << ", Hjoint " << Hjoint
			<< ", max_tc " << max_tc << std::endl;
	}
	assert((Hsep - Hjoint) <= max_tc);

	return (Hsep - Hjoint);
}


}

