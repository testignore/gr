
#ifndef GRANTE_LINEARFTYPE_H
#define GRANTE_LINEARFTYPE_H

#include "FactorType.h"

namespace Grante {

/* General linear factor type supporting tying and sparsity.  Linear means
 * that it depends linearly on its learnable parameters so that the overall
 * model remains log-linear.
 */
class LinearFactorType : public FactorType {
public:
	// name, card, data_size: As for FactorType,
	// A: vector of length prod_card (card[0]*...*card[card.size()-1]), that
	//    contains indices 0,1,... tying elements in the energy table.  The
	//    special element -1 is reserved for fixed zero values.  Let total_a
	//    be the total number of unique indices >=0.  The indices must be
	//    gap-free, that is, they must start with zero and in increments of
	//    one.
	// w: Must be non-empty.  If data_size==0, then w.size()==total_a.  If
	//    data_size>=1, then w.size()==total_a*data_size.
	LinearFactorType(const std::string& name,
		const std::vector<unsigned int>& card,
		const std::vector<double>& w, unsigned int data_size,
		const std::vector<int>& A);

	virtual bool IsDataDependent() const;

	virtual void ForwardMap(const Factor* factor,
		std::vector<double>& energies) const;

	virtual void BackwardMap(const Factor* factor,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult = 1.0) const;

private:
	// The sparsity/tying pattern matrix A with prod_card elements.
	// Has elements -1, 0, ..., total_a-1, where an element of -1 means this
	// energy is always zero, and an element >=0 is a tying pattern index.
	// For example, [-1,0,0,-1] would correspond to the set of symmetric
	// matrices
	//   0 a
	//   a 0,
	// where a could be a parameter or data-dependent energy.  The matrix
	// [-1,0,1,0,-1,0,2,0,-1] is the asymmetric matrix with three free
	// parameters:
	//   0 a c
	//   a 0 a
	//   b a 0.
	// The original FactorType corresponds to A=[0,1,2,...,prod_card-1].
	// Symmetry can be enforced using a pattern such as this:
	//   0 1 2 3
	//   1 4 5 6
	//   2 5 8 9
	//   3 6 9 10,
	// where the lower-triangular part is a flipped version of the
	// upper-triangular part.
	std::vector<int> A;

	// The total number of tying patterns.
	unsigned int total_a;

	// origin[ei] is either ==ei or an index ei2<ei.  This is used to avoid
	// recomputing redundant elements.
	std::vector<unsigned int> origin;

	void ForwardMap(const std::vector<double>& factor_data,
		std::vector<double>& energies) const;
	void BackwardMap(const std::vector<double>& factor_data,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult) const;
};

}

#endif

