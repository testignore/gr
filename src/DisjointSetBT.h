
#ifndef GRANTE_DISJOINTSETBT_H
#define GRANTE_DISJOINTSETBT_H

#include <vector>
#include <stack>

namespace Grante {

/* Disjoint set data structure with backtracking.
 *
 * The data structure maintains a collection of disjoint set and supports
 * three operations,
 *
 * 1. Union(i,j), combine the set containing i and the set containing j into
 *    one.  The worst-case complexity is O(1).
 * 2. Find(i), find the set containing i.  Amortized complexity is
 *    O(log log N), where N is the total number of elements).
 * 3. Deunion(), undo the last not-yet-undone Union operation.
 *    The worst-case complexity is O(1).
 *
 * This particular implementation has been proposed in [Mannila1986] and for
 * the above operations and arbitrary sequences of operations, this data
 * structure has optimal amortized complexity, see [Galil1991].
 *
 * Reference
 * [Mannila1986] Heikki Mannila and Esko Ukkonen,
 *    "The Set Union Problem with Backtracking",
 *    Lecture Notes in Computer Science, Vol. 226, pages 236-243, 1986.
 * [Galil1991] Ziv Galil and Giuseppe F. Italiano,
 *    "Data Structures and Algorithms for Disjoint Set Union Problems",
 *    ACM Computing Surveys, Vol. 23, No. 3, 1991.
 */
class DisjointSetBT {
public:
	explicit DisjointSetBT(size_t number_of_elements);

	// Given an element index, find the representer element of its set.
	//
	// Return the set-representing element index.
	// Amortized complexity is O(log log N).
	unsigned int Find(unsigned int element_index) const;

	// Link two sets represented by root1 and root2.  This uses union-rank.
	// root1 and root2 must be different.
	//
	// Return the new root of the merged tree.  Complexity is O(1).
	unsigned int Union(unsigned int root1, unsigned int root2);

	// Undo the last not-yet-undone link operation.
	void Deunion(void);

	// Label all elements uniquely.  The vector out_labeling will be properly
	// resized.
	//
	// Return the number of modes.  Complexity is O(N log N).
	unsigned int UniqueLabeling(std::vector<unsigned int>& out_labeling) const;

	// Return the number of disjoint sets.  Complexity is O(N log N).
	unsigned int NumberOfDisjointSets() const;

private:
	size_t number_of_elements;	// universe size
	std::vector<unsigned int> rank;

	///
	/// UNION STACK
	///

	// Incremental time stamp
	unsigned long current_time;
	unsigned int union_stack_ec;	// number of elements in the union_stack
	struct union_stack_elem_t {
		unsigned long timestamp;	// union event time
		unsigned int prev_root;		// root made non-root
	};
	typedef struct union_stack_elem_t union_stack_elem_t;
	std::vector<union_stack_elem_t> union_stack;

	///
	/// NODE STACKS
	///
	struct node_stack_elem_t {
		unsigned long timestamp;	// UNION link timestamp
		unsigned int node_stack_L;	// UNION corresponding to this link
		unsigned int parent;	// parent pointer
	};
	typedef struct node_stack_elem_t node_stack_elem_t;
	typedef std::stack<node_stack_elem_t, std::vector<node_stack_elem_t> >
		node_stack_t;
	mutable std::vector<node_stack_t> node_stacks;

	// Lazy cleanup of a node stack
	void CleanupNodeStack(unsigned int node_id) const;
};

}

#endif


