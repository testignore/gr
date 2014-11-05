
#ifndef GRANTE_DISJOINTSET_H
#define GRANTE_DISJOINTSET_H

#include <vector>

namespace Grante {

/* Disjoint-set-union-rank data structure for efficient lookup and merging of
 * disjoint sets.
 */
class DisjointSet {
public:
	explicit DisjointSet(size_t number_of_elements);

	// Given an element index, find the representer element of its set.
	// This performs path compression but does not change the semantics of
	// DisjointSet.
	//
	// Return the set-representing element index.
	// Complexity is O(\alpha(n)).
	unsigned int FindSet(unsigned int element_index) const;

	// Link two sets represented by root1 and root2.  This uses union-rank.
	// root1 and root2 must be different.
	// Return the new root of the merged tree.  Complexity is O(1).
	unsigned int Link(unsigned int root1, unsigned int root2);

	// Label all elements uniquely.  The vector out_labeling will be properly
	// resized.
	// Return the number of modes.
	unsigned int UniqueLabeling(std::vector<unsigned int>& out_labeling) const;

	// Return the number of disjoint sets.
	unsigned int NumberOfDisjointSets() const;

private:
	size_t number_of_elements;
	mutable std::vector<unsigned int> parent;
	std::vector<unsigned int> rank;
};

}

#endif

