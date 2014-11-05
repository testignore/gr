
#include <algorithm>
#include <map>
#include <cassert>

#include "DisjointSet.h"

namespace Grante {

DisjointSet::DisjointSet(size_t number_of_elements)
	: number_of_elements(number_of_elements), parent(number_of_elements),
		rank(number_of_elements) {
	for (size_t n = 0; n < number_of_elements; ++n)
		parent[n] = static_cast<unsigned int>(n);
	std::fill(rank.begin(), rank.end(), 0);
}

unsigned int DisjointSet::FindSet(unsigned int element_index) const {
	assert(element_index >= 0 && element_index < number_of_elements);
	unsigned int ei = element_index;
	while (parent[ei] != ei)
		ei = parent[ei];

	// Path-compression
	while (parent[element_index] != ei) {
		unsigned int next_parent = parent[element_index];
		parent[element_index] = ei;
		element_index = next_parent;
	}

	return (ei);
}

unsigned int DisjointSet::Link(unsigned int root1, unsigned int root2) {
	// Preconditions
	assert(root1 >= 0 && root1 < number_of_elements);
	assert(root2 >= 0 && root2 < number_of_elements);
	assert(parent[root1] == root1);
	assert(parent[root2] == root2);
	assert(root1 != root2);

	// Keep the resulting tree flat by considering the rank
	if (rank[root1] >= rank[root2]) {
		parent[root2] = root1;
		rank[root1] += 1;
		return (root1);
	} else {
		parent[root1] = root2;
		rank[root2] += 1;
		return (root2);
	}
}

unsigned int DisjointSet::UniqueLabeling(
	std::vector<unsigned int>& out_labeling) const {
	out_labeling.resize(number_of_elements);
	std::map<unsigned int, unsigned int> parent_to_unique;
	unsigned int unique_label = 0;
	for (unsigned int ei = 0; ei < number_of_elements; ++ei) {
		unsigned int parent = FindSet(ei);
		// If this parent already exist, skip
		if (parent_to_unique.find(parent) != parent_to_unique.end())
			continue;

		// Does not exist yet, assign it a unique label
		parent_to_unique[parent] = unique_label;
		unique_label += 1;
	}

	for (unsigned int ei = 0; ei < number_of_elements; ++ei)
		out_labeling[ei] = parent_to_unique[FindSet(ei)];

	return (unique_label);
}

unsigned int DisjointSet::NumberOfDisjointSets() const {
	std::map<unsigned int, unsigned int> parent_to_unique;
	unsigned int unique_label = 0;
	for (unsigned int ei = 0; ei < number_of_elements; ++ei) {
		unsigned int parent = FindSet(ei);
		if (parent_to_unique.find(parent) != parent_to_unique.end())
			continue;

		parent_to_unique[parent] = unique_label;
		unique_label += 1;
	}
	return (unique_label);
}

}

