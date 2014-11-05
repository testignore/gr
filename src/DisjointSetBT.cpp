
#include <algorithm>
#include <map>
#include <cassert>

#include "DisjointSetBT.h"

namespace Grante {

DisjointSetBT::DisjointSetBT(size_t number_of_elements)
	: number_of_elements(number_of_elements),
		rank(number_of_elements), current_time(1), union_stack_ec(0) {
	std::fill(rank.begin(), rank.end(), 0);

	node_stacks.resize(number_of_elements);
}

unsigned int DisjointSetBT::Find(unsigned int element_index) const {
	assert(element_index >= 0 && element_index < number_of_elements);
	unsigned int ei = element_index;

	CleanupNodeStack(ei);
	while (node_stacks[ei].empty() == false) {
		// Follow live link
		const node_stack_elem_t& top = node_stacks[ei].top();
		unsigned int ei_parent = top.parent;

		CleanupNodeStack(ei_parent);

		// Path-splitting (due to van Leeuwen and van der Weide)
		if (node_stacks[ei_parent].empty() == false) {
			// Transform
			//       ei --> ei_parent --> super_parent
			// to
			//       ei ----------------> super_parent
			node_stacks[ei].push(node_stacks[ei_parent].top());
		}

		// Move up
		ei = ei_parent;
	}

	return (ei);
}

unsigned int DisjointSetBT::Union(unsigned int root1, unsigned int root2) {
	// Preconditions
	assert(root1 >= 0 && root1 < number_of_elements);
	assert(root2 >= 0 && root2 < number_of_elements);
	assert(root1 != root2);
	CleanupNodeStack(root1);
	CleanupNodeStack(root2);
	assert(node_stacks[root1].empty());	// is indeed a root node
	assert(node_stacks[root2].empty());

	// UNION stack element: timestamps this union operation and contains the
	// node that is made non-root.  This is sufficient information to reverse
	// the union.
	union_stack_elem_t us;
	us.timestamp = current_time;

	// NODE stack element: contains a link to the corresponding union
	// operation on the node stack as well as the same timestamp, allowing us
	// to determine whether the link is still live.  The top-most live node
	// stack element on the node stack is the one that determines the actual
	// link.
	node_stack_elem_t ne;
	ne.timestamp = current_time;
	ne.node_stack_L = union_stack_ec;

	current_time += 1;

	// Keep the resulting tree flat by considering the rank
	unsigned int new_root = root1;
	unsigned int new_child = root2;
	if (rank[root1] < rank[root2]) {
		new_root = root2;
		new_child = root1;
	}
	us.prev_root = new_child;	// the node made non-root
	ne.parent = new_root;
	rank[new_root] += 1;

	// Push LINK change onto node stack
	node_stacks[new_child].push(ne);

	// Push UNION on the global union stack
	union_stack_ec += 1;
	if (union_stack.size() < union_stack_ec) {
		union_stack.push_back(us);
	} else {
		union_stack[union_stack_ec-1] = us;
	}

	return (new_root);
}

void DisjointSetBT::Deunion(void) {
	assert(union_stack_ec >= 1);

	// Undo union
	//
	// The rank is easy to update by observing that the parent will always be
	// a root node.
	union_stack_elem_t& top = union_stack[union_stack_ec-1];

	CleanupNodeStack(top.prev_root);
	const node_stack_elem_t& prev_root_ns = node_stacks[top.prev_root].top();

	unsigned int cur_parent = prev_root_ns.parent;
	rank[cur_parent] -= 1;	// undo rank update

	// Invalidate this element in the union stack and semi-pop it off the
	// stack
	top.timestamp = std::numeric_limits<unsigned long>::max();
	union_stack_ec -= 1;
}

unsigned int DisjointSetBT::UniqueLabeling(
	std::vector<unsigned int>& out_labeling) const {
	out_labeling.resize(number_of_elements);
	std::map<unsigned int, unsigned int> parent_to_unique;
	unsigned int unique_label = 0;
	for (unsigned int ei = 0; ei < number_of_elements; ++ei) {
		unsigned int parent = Find(ei);
		// If this parent already exist, skip
		if (parent_to_unique.find(parent) != parent_to_unique.end())
			continue;

		// Does not exist yet, assign it a unique label
		parent_to_unique[parent] = unique_label;
		unique_label += 1;
	}

	for (unsigned int ei = 0; ei < number_of_elements; ++ei)
		out_labeling[ei] = parent_to_unique[Find(ei)];

	return (unique_label);
}

unsigned int DisjointSetBT::NumberOfDisjointSets() const {
	std::map<unsigned int, unsigned int> parent_to_unique;
	unsigned int unique_label = 0;
	for (unsigned int ei = 0; ei < number_of_elements; ++ei) {
		unsigned int parent = Find(ei);
		if (parent_to_unique.find(parent) != parent_to_unique.end())
			continue;

		parent_to_unique[parent] = unique_label;
		unique_label += 1;
	}
	return (unique_label);
}

void DisjointSetBT::CleanupNodeStack(unsigned int node_id) const {
	node_stack_t& ns = node_stacks[node_id];
	if (ns.empty())
		return;

	// Remove all dead links
	do {
		const node_stack_elem_t& top = ns.top();
		if (union_stack[top.node_stack_L].timestamp == top.timestamp)
			return;	// Done, this link is live

		// Remove link
		ns.pop();
	} while (ns.empty() == false);
}

}

