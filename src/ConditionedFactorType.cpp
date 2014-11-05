
#include <algorithm>
#include <cassert>

#include "ConditionedFactorType.h"
#include "FactorConditioningTable.h"

namespace Grante {

ConditionedFactorType::ConditionedFactorType(const FactorType* base_ft,
	const std::vector<unsigned int>& cond_var_index,
	const FactorConditioningTable* fcond_data)
	: base_ft(base_ft), cond_var_index(cond_var_index), fcond_data(fcond_data) {
	// Initialize everything so this is compatible with the FactorType
	// interface.  Infact, this needs to operate like any normal factor.
	// The conditional factor retains the name of the unconditional factor.
	// This is needed for parameter learning.
	name = base_ft->Name();

	const std::vector<unsigned int>& base_card = base_ft->Cardinalities();
	assert(base_card.size() > cond_var_index.size());
	cardinalities.reserve(base_card.size()-cond_var_index.size());
	prod_cumcard.reserve(base_card.size()-cond_var_index.size());

	unsigned int cvi = 0;
	prod_card = 1;
	for (unsigned int vi = 0; vi < base_card.size(); ++vi) {
		if (cvi > 0 && cvi < cond_var_index.size()) {
			// Conditioning variable indices must be ordered
			assert(cond_var_index[cvi-1] < cond_var_index[cvi]);
		}
		if (cvi < cond_var_index.size() && cond_var_index[cvi] == vi) {
			// Conditioned variable
			cvi += 1;
			continue;
		}

		// Unconditioned variable
		cardinalities.push_back(base_card[vi]);
		prod_cumcard.push_back(static_cast<unsigned int>(prod_card));
		prod_card *= base_card[vi];
	}
	assert(cardinalities.size() == (base_card.size()-cond_var_index.size()));

	// No weight vector is ever stored in conditional factor types
	w.clear();
	data_size = 0;
}

ConditionedFactorType::~ConditionedFactorType() {
}

bool ConditionedFactorType::operator==(const ConditionedFactorType& cft2) const
{
	if (base_ft != cft2.base_ft)
		return (false);

	if (cond_var_index.size() != cft2.cond_var_index.size())
		return (false);

	// Condition on the same subset of variables?
	return (std::equal(cond_var_index.begin(), cond_var_index.end(),
		cft2.cond_var_index.begin()));
}

bool ConditionedFactorType::operator!=(const ConditionedFactorType& cft2) const
{
	return (!(*this == cft2));
}

const FactorType* ConditionedFactorType::BaseType() const {
	return (base_ft);
}

const std::vector<unsigned int>&
ConditionedFactorType::ConditionedVariableIndices() const {
	return (cond_var_index);
}

bool ConditionedFactorType::IsDataDependent() const {
	return (true);
}

std::vector<double>& ConditionedFactorType::Weights() {
	return (const_cast<FactorType*>(base_ft)->Weights());
}

const std::vector<double>& ConditionedFactorType::Weights() const {
	return (const_cast<FactorType*>(base_ft)->Weights());
}

unsigned int ConditionedFactorType::WeightDimension() const {
	return (base_ft->WeightDimension());
}

void ConditionedFactorType::ForwardMap(const Factor* factor,
	std::vector<double>& energies) const {
	// 1. Obtain original factor and invoke forward map
	Factor* orig_factor = fcond_data->OriginalFactor(factor);
	assert(orig_factor != 0);
	orig_factor->ForwardMap();

	// 2. Translate energies from original Factor to conditioned Factor
	const std::vector<double>& orig_energies = orig_factor->Energies();
	fcond_data->ConditionEnergies(factor, orig_energies, energies);
}

void ConditionedFactorType::BackwardMap(const Factor* factor,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	// 1. Extend conditional marginals to full marginals
	std::vector<double> ext_marginals(base_ft->ProdCardinalities(), 0.0);
	fcond_data->ExtendMarginals(factor, marginals, ext_marginals);

	// 2. Perform backward map on extended marginals using original factor
	// type
	const Factor* orig_factor = fcond_data->OriginalFactor(factor);
	assert(orig_factor != 0);
	base_ft->BackwardMap(orig_factor, ext_marginals, parameter_gradient, mult);
}

}

