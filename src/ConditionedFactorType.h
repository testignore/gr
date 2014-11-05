
#ifndef GRANTE_CONDITIONEDFACTORTYPE_H
#define GRANTE_CONDITIONEDFACTORTYPE_H

#include <vector>
#include <functional>

#include "FactorType.h"

namespace Grante {

class FactorConditioningTable;

/* Factor type derived by conditioning another factor type.
 * A factor is 'conditioned' by marginalization over a distribution defined
 * for a strict non-empty subset of its adjacent variables.
 *
 * Right now, conditioning is only supported for discrete partial
 * observations, that is, fixing a subset of the factors' variables to known
 * values.
 *
 * Note: The information what relative variables are conditioned is stored
 * in this factor type, the actual value it is conditioned on is stored in the
 * factor instantiating this type.
 */
class ConditionedFactorType : public FactorType {
public:
	/* base_ft: The base factor type that is being conditioned
	 * cond_var_index_set: An ordered subset of indices
	 *    [0,...,base_ft->Cardinalities.size()-1].
	 * fcond_data: conditioning data (what adjacent variable of a factor is
	 *    set to which value).
	 *
	 * We do not take ownership of any of the objects and all of them except
	 * cond_var_index need to remain valid throughout the lifetime of this
	 * object.
	 */
	ConditionedFactorType(const FactorType* base_ft,
		const std::vector<unsigned int>& cond_var_index,
		const FactorConditioningTable* fcond_data);

	virtual ~ConditionedFactorType();

	// Two conditioned factor types are equal if they condition the same
	// factor type and condition the same relative variables.
	bool operator==(const ConditionedFactorType& cft2) const;
	bool operator!=(const ConditionedFactorType& cft2) const;

	const FactorType* BaseType() const;
	const std::vector<unsigned int>& ConditionedVariableIndices() const;

	virtual bool IsDataDependent() const;

	// Pass through to base type
	virtual std::vector<double>& Weights();
	virtual const std::vector<double>& Weights() const;
	virtual unsigned int WeightDimension() const;

	/// Overwrite the forward/backward map operations
	// ForwardMap: translate energies from the original factor to the
	//    reduced factor.  We do not assume the original base factor graph has
	//    been forward mapped when this ForwardMap is called and instead
	//    invoke the forward map first.
	virtual void ForwardMap(const Factor* factor,
		std::vector<double>& energies) const;
	virtual void BackwardMap(const Factor* factor,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult = 1.0) const;

	struct condfac_tp_hash :
		public std::unary_function<ConditionedFactorType*, size_t>
	{
		size_t operator()(ConditionedFactorType* cft) const {
			size_t h1 = reinterpret_cast<size_t>(cft->base_ft);
			size_t h2 = 1;
			for (std::vector<unsigned int>::const_iterator cvi =
				cft->cond_var_index.begin(); cvi != cft->cond_var_index.end();
				++cvi)
			{
				h2 *= 0x7f;
				h2 += *cvi;
			}
			return (h1 ^ ~h2);
		}
	};

private:
	// The original factor type we condition
	const FactorType* base_ft;

	// The factortype-relative indices that are conditioned
	const std::vector<unsigned int> cond_var_index;

	// Actual data being conditioned on.  This data is external and can be
	// changed.
	const FactorConditioningTable* fcond_data;
};

}

#endif

