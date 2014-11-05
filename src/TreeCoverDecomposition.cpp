
#include <algorithm>

#include "DisjointSet.h"
#include "TreeCoverDecomposition.h"

namespace Grante {

TreeCoverDecomposition::TreeCoverDecomposition(const FactorGraph* fg)
	: fg(fg) {
}

void TreeCoverDecomposition::ComputeDecompositionGreedy(
	std::vector<std::vector<unsigned int> >& tree_factor_indices,
	std::vector<unsigned int>& factor_cover_count) const {
	const std::vector<Factor*>& factors = fg->Factors();
	size_t factor_count = factors.size();
	factor_cover_count.resize(factor_count);
	std::fill(factor_cover_count.begin(), factor_cover_count.end(), 0);

	// Keep adding trees until all factors are covered
	while (std::find(factor_cover_count.begin(), factor_cover_count.end(), 0)
		!= factor_cover_count.end()) {

		// Find a single new tree to add
		std::vector<unsigned int> tree_fidx;
		AddTree(factor_cover_count, tree_fidx);
		tree_factor_indices.push_back(tree_fidx);

		// Update factor cover counts
		for (std::vector<unsigned int>::const_iterator ti = tree_fidx.begin();
			ti != tree_fidx.end(); ++ti) {
			factor_cover_count[*ti] += 1;
		}
	}
}

void TreeCoverDecomposition::AddTree(
	const std::vector<unsigned int>& factor_cover_count,
	std::vector<unsigned int>& tree_fidx) const {
	// 1. Order factors ascendingly by cover count
	const std::vector<Factor*>& factors = fg->Factors();
	std::vector<unsigned int> facs_order(factors.size());
	for (unsigned int fi = 0; fi < facs_order.size(); ++fi)
		facs_order[fi] = fi;
	std::sort(facs_order.begin(), facs_order.end(),
		CountOrdering(factor_cover_count));

	DisjointSet dset(fg->Cardinalities().size());

	// For all factor nodes
	for (unsigned int fi = 0; fi < facs_order.size(); ++fi) {
		// Factor to be considered for addition
		const Factor* fac = factors[facs_order[fi]];

		// For all variables adjacent to the factor node
		const std::vector<unsigned int>& vars = fac->Variables();
		unsigned int v0_set = dset.FindSet(vars[0]);
		bool can_be_added = true;
		for (unsigned int vi = 1; vi < vars.size(); ++vi) {
			unsigned int vi_set = dset.FindSet(vars[vi]);
			if (v0_set == vi_set) {
				can_be_added = false;
				break;
			}
		}
		// Unaries can always be added, pairwise only if no cycle
		if (can_be_added == false)
			continue;

		// Add factor: merge all variables in the two disjoint sets
		for (unsigned int vi = 1; vi < vars.size(); ++vi) {
			unsigned int vi_set = dset.FindSet(vars[vi]);
			v0_set = dset.Link(v0_set, vi_set);
		}
		tree_fidx.push_back(facs_order[fi]);
	}
}

}

