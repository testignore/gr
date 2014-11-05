
#ifndef GRANTE_TREECOVERDECOMPOSITION_H
#define GRANTE_TREECOVERDECOMPOSITION_H

#include <vector>
#include <functional>

#include "FactorGraph.h"

namespace Grante {

class TreeCoverDecomposition {
public:
	explicit TreeCoverDecomposition(const FactorGraph* fg);

	// Cover the input factor graph with a set of spanning trees
	//
	// tree_factor_indices: (output) the .size() is the number of trees, and
	//    each tree_factor_indices[t] contains the set of factor indices in
	//    that tree.  (A factor is always completely in or completely out the
	//    tree and never conditioned.)  The tree is spanning and thus all
	//    variables of the original factor graph are used.  This argument is
	//    resized accordingly.
	// factor_cover_count: (output) for each factor index fi, the value of
	//    factor_cover_count[fi] denotes the number of times the factor was
	//    covered.
	void ComputeDecompositionGreedy(
		std::vector<std::vector<unsigned int> >& tree_factor_indices,
		std::vector<unsigned int>& factor_cover_count) const;

private:
	const FactorGraph* fg;

	struct CountOrdering :
		public std::binary_function<unsigned int, unsigned int, bool> {
	public:
		CountOrdering(const std::vector<unsigned int>& counts)
			: counts(counts) {
		}

		bool operator()(unsigned int i1, unsigned int i2) const {
			return (counts[i1] < counts[i2]);
		}

	private:
		const std::vector<unsigned int>& counts;
	};

	// Find a new tree to add
	void AddTree(const std::vector<unsigned int>& factor_cover_count,
		std::vector<unsigned int>& tree_fidx) const;
};

}

#endif

