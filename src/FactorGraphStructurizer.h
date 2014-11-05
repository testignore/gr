
#ifndef GRANTE_FACTORGRAPHSTRUCT_H
#define GRANTE_FACTORGRAPHSTRUCT_H

#include <vector>
#include <utility>
#include <tr1/unordered_set>

#include "FactorGraph.h"

namespace Grante {

/* Analyzing, preprocessing, decomposing factor graphs.
 *
 * This class can answer basic questions such as whether the factor graph is
 * tree structured.
 */
class FactorGraphStructurizer {
public:
	// Returns true if fg is forest-structured, false otherwise.
	// The complexity is O(A(E) E), where E is the number of factor graph
	// edges and A(E) is the inverse Ackermann function (which is <5 for all
	// reasonable numbers).
	// Note: this function does not test whether the graph is connected.
	static bool IsForestStructured(const FactorGraph* fg);

	// Returns true if fg is connected
	// Optional: return representing roots of disconnected components
	static bool IsConnected(const FactorGraph* fg);
	static bool IsConnected(const FactorGraph* fg,
		std::tr1::unordered_set<unsigned int>* roots);

	// Returns true if fg is connected and forest structured
	static bool IsTreeStructured(const FactorGraph* fg);

	// Return true if fg is a 4-grid structured graph.  This can identify not
	// all grid structured graphs but only graphs whose variable ordering
	// follows already a column-major or row-major layout.
	static bool IsOrderedPairwiseGridStructured(const FactorGraph* fg);
	// If true, additionally return partitioning of the graph into rows and
	// columns, not necessarily in any particular order
	static bool IsOrderedPairwiseGridStructured(const FactorGraph* fg,
		std::vector<std::vector<unsigned int> >& var_rows,
		std::vector<std::vector<unsigned int> >& var_cols);

	// Label the vertices of the factor graph into their connected components.
	// Return the number of connected components in the factor graph.
	static unsigned int ConnectedComponents(const FactorGraph* fg,
		const std::vector<bool>& factor_is_removed,
		std::vector<unsigned int>& var_label);
	static unsigned int ConnectedComponents(const FactorGraph* fg,
		std::vector<unsigned int>& var_label);

	enum OrderStepType {
		LeafIsFactorNode = 0,
		LeafIsVariableNode,
	};
	struct OrderStep {
		OrderStepType steptype;

		// 'leaf' and 'root' are always of different types:
		//    if steptype == LeafIsFactorNode, then root is a variable node,
		//    if steptype == LeafIsVariableNode, then root is a factor node.
		//
		// The number in leaf and root denotes either the variable or factor
		// node index (the index into in fg->Factors()).
		unsigned int leaf;
		unsigned int root;

		OrderStep(OrderStepType steptype, unsigned int leaf, unsigned int root);

		// Returns the variable index, irrespectible of the step type
		unsigned int VariableNode() const;
		// Returns the factor index, irrespectible of the step type
		unsigned int FactorNode() const;
	};

	// Compute a leaf-to-root order of variable- and factor-nodes for
	// sum-product computation.
	// tree_roots: Set of component variable node indices that has been made
	// the root of the tree.
	// Return the number of trees in this factor graph (one or more).
	static unsigned int ComputeTreeOrder(const FactorGraph* fg,
		std::vector<OrderStep>& order,
		std::tr1::unordered_set<unsigned int>& tree_roots);

	static void PrintOrder(const std::vector<OrderStep>& order);

	// Find an Eulerian trail in a digraph.  No checking is performed as to
	// whether the graph is Eulerian, that is, whether a trail exists.
	//
	// vertex_count: Number of vertices.
	// arcs: Array of directed edges, .first is originating, .second is end of
	//    the arc.
	//
	// After this method returns, trail[ei] contains the successor edge of
	// edge ei.  A trail can be extracted starting from any edge e0:
	//    ei = e0;
	//    do {
	//        ei = trail[ei];
	//    } while (ei != e0);
	static void ComputeEulerianTrail(unsigned int vertex_count,
		const std::vector<std::pair<unsigned int, unsigned int> >& arcs,
		std::vector<unsigned int>& trail);
	static void ComputeEulerianMessageTrail(const FactorGraph* fg,
		std::vector<OrderStep>& order);
};

}

#endif

