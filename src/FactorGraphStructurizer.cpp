
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>
#include <tr1/unordered_map>

#include "FactorGraphStructurizer.h"
#include "FactorGraphUtility.h"
#include "DisjointSet.h"

namespace Grante {

bool FactorGraphStructurizer::IsForestStructured(const FactorGraph* fg) {
	DisjointSet dset(fg->Cardinalities().size());

	// For all factor nodes
	const std::vector<Factor*>& factors = fg->Factors();
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		// For all variables adjacent to the factor node
		const std::vector<unsigned int>& vars = (*fi)->Variables();
		unsigned int v0_set = dset.FindSet(vars[0]);
		for (unsigned int vi = 1; vi < vars.size(); ++vi) {
			unsigned int vi_set = dset.FindSet(vars[vi]);
			if (v0_set == vi_set)
				return (false);	// cycle detected

			// Merge all variables in the two disjoint sets
			v0_set = dset.Link(v0_set, vi_set);
		}
	}
	return (true);
}

bool FactorGraphStructurizer::IsConnected(const FactorGraph* fg) {
	return (IsConnected(fg, 0));
}

bool FactorGraphStructurizer::IsConnected(const FactorGraph* fg,
	std::tr1::unordered_set<unsigned int>* roots) {
	size_t var_count = fg->Cardinalities().size();
	DisjointSet dset(var_count);
	if (roots != 0)
		roots->clear();

	// For all factor nodes, connect
	const std::vector<Factor*>& factors = fg->Factors();
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		const std::vector<unsigned int>& vars = (*fi)->Variables();
		unsigned int v0_set = dset.FindSet(vars[0]);
		for (size_t vi = 1; vi < vars.size(); ++vi)
			v0_set = dset.Link(v0_set, dset.FindSet(vars[vi]));
	}
	if (roots != 0) {
		for (unsigned int vi = 0; vi < static_cast<unsigned int>(var_count);
			++vi) {
			roots->insert(dset.FindSet(vi));
		}
	}
	return (dset.NumberOfDisjointSets() == 1);
}

bool FactorGraphStructurizer::IsTreeStructured(const FactorGraph* fg) {
	return (IsConnected(fg) && IsForestStructured(fg));
}

bool FactorGraphStructurizer::IsOrderedPairwiseGridStructured(
	const FactorGraph* fg) {
	std::vector<std::vector<unsigned int> > var_rows;
	std::vector<std::vector<unsigned int> > var_cols;
	return (IsOrderedPairwiseGridStructured(fg, var_rows, var_cols));
}

bool FactorGraphStructurizer::IsOrderedPairwiseGridStructured(
	const FactorGraph* fg,
	std::vector<std::vector<unsigned int> >& var_rows,
	std::vector<std::vector<unsigned int> >& var_cols) {
	// Count variable degrees
	std::vector<unsigned int> var_pw_degree(fg->Cardinalities().size(), 0);
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const std::vector<unsigned int>& fac_vars = factors[fi]->Variables();

		// Higher-order factors -> not pairwise
		if (fac_vars.size() > 2)
			return (false);

		// Do not count unaries
		if (fac_vars.size() == 1)
			continue;

		for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
			assert(fac_vars[fvi] < var_pw_degree.size());
			var_pw_degree[fac_vars[fvi]] += 1;
		}
	}

	// Has 4 factors of pairwise-degree two
	unsigned int deg2_count = 0;
	unsigned int deg3_count = 0;
	unsigned int deg4_count = 0;
	for (unsigned int vi = 0; vi < var_pw_degree.size(); ++vi) {
		if (var_pw_degree[vi] == 0 || var_pw_degree[vi] > 4)
			return (false);

		switch (var_pw_degree[vi]) {
		case (2):
			deg2_count += 1;
			break;
		case (3):
			deg3_count += 1;
			break;
		case (4):
			deg4_count += 1;
			break;
		default:
			return (false);
			break;
		}
	}
	// Four corners
	if (deg2_count != 4)
		return (false);

	assert(((deg3_count * deg3_count)/16) >= deg4_count);
	double tde = std::sqrt(0.0625*(deg3_count * deg3_count)
		- static_cast<double>(deg4_count));
	unsigned int d1 = static_cast<unsigned int>(
		0.25*static_cast<double>(deg3_count)
		+ tde + 2.0 + 1.0e-3);
	unsigned int d2 = static_cast<unsigned int>(
		0.25*static_cast<double>(deg3_count)
		- tde + 2.0 + 1.0e-3);
	if (d1*d2 != var_pw_degree.size())
		return (false);

	// Find variable ordering (column-major or row-major)
	if (var_pw_degree[0] != 2 || *var_pw_degree.rbegin() != 2)
		return (false);
	unsigned int bi = 1;
	while (var_pw_degree[bi] != 2)
		bi += 1;
	bi += 1;

	// Make d1 the number of columns
	if (d1 != bi) {
		unsigned int temp_d = d1;
		d1 = d2;
		d2 = temp_d;
	}

#if 0
	std::cout << "Identified a " << d1 << "-by-" << d2 << " grid"
		<< std::endl;
#endif

	// Provide row and column ordering
	var_cols.clear();
	var_cols.resize(d1);
	var_rows.clear();
	var_rows.resize(d2);
	for (unsigned int vi = 0; vi < var_pw_degree.size(); ++vi) {
		unsigned int col = vi % d1;
		unsigned int row = vi / d1;
		var_cols[col].push_back(vi);
		var_rows[row].push_back(vi);
		if (col == 0 || col == (d1-1)) {
			assert(var_pw_degree[vi] < 4);
		}
		if (row == 0 || row == (d2-1)) {
			assert(var_pw_degree[vi] < 4);
		}
	}
	return (true);
}

unsigned int FactorGraphStructurizer::ConnectedComponents(
	const FactorGraph* fg, std::vector<unsigned int>& var_label) {
	std::vector<bool> factor_is_removed_dummy;
	return (ConnectedComponents(fg, factor_is_removed_dummy, var_label));
}

unsigned int FactorGraphStructurizer::ConnectedComponents(
	const FactorGraph* fg, const std::vector<bool>& factor_is_removed,
	std::vector<unsigned int>& var_label) {
	// For all factor nodes
	const std::vector<Factor*>& factors = fg->Factors();
	assert(factor_is_removed.empty() ||
		factor_is_removed.size() == factors.size());
	DisjointSet dset(fg->Cardinalities().size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		if (factor_is_removed.empty() == false && factor_is_removed[fi])
			continue;

		// For all variables adjacent to the factor node
		const std::vector<unsigned int>& vars = factors[fi]->Variables();
		unsigned int v0_set = dset.FindSet(vars[0]);
		for (unsigned int vi = 1; vi < vars.size(); ++vi) {
			unsigned int vi_set = dset.FindSet(vars[vi]);
			if (v0_set == vi_set)
				continue;	// already linked

			// Merge all variables in the two sets
			v0_set = dset.Link(v0_set, dset.FindSet(vars[vi]));
		}
	}
	return (dset.UniqueLabeling(var_label));
}

unsigned int FactorGraphStructurizer::ComputeTreeOrder(const FactorGraph* fg,
	std::vector<OrderStep>& order,
	std::tr1::unordered_set<unsigned int>& tree_roots) {
	// Variable to factor lookup table: all factors containing this variable.
	typedef std::tr1::unordered_multimap<unsigned int, unsigned int> v2f_map_type;
	v2f_map_type v2f;
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const std::vector<unsigned int>& vars = factors[fi]->Variables();
		for (unsigned int vi = 0; vi < vars.size(); ++vi)
			v2f.insert(v2f_map_type::value_type(vars[vi], fi));
	}

	// Fix arbitrary variables roots of the connected components (tree) of the
	// factor graph
	IsConnected(fg, &tree_roots);
	assert(tree_roots.size() >= 1);

	std::vector<unsigned int> gnode_stack;
	std::vector<bool> gtype_stack;	// 'is variable'
	std::vector<unsigned int> gcamefrom_stack;
	for (std::tr1::unordered_set<unsigned int>::const_iterator
		tri = tree_roots.begin(); tri != tree_roots.end(); ++tri) {
		gnode_stack.push_back(*tri);
		gtype_stack.push_back(true);
		gcamefrom_stack.push_back(std::numeric_limits<unsigned int>::max());
	}

	while (gnode_stack.empty() == false) {
		unsigned int cur_node = *gnode_stack.rbegin();
		gnode_stack.pop_back();
		bool is_variable = *gtype_stack.rbegin();
		gtype_stack.pop_back();
		unsigned int came_from = *gcamefrom_stack.rbegin();
		gcamefrom_stack.pop_back();

		if (is_variable) {
			// Parent: variable node,
			// Children: factor nodes.
			std::pair<v2f_map_type::const_iterator, v2f_map_type::const_iterator>
				adj_factors = v2f.equal_range(cur_node);
			for (v2f_map_type::const_iterator mi = adj_factors.first;
				mi != adj_factors.second; ++mi) {
				unsigned int adj_factor_id = (*mi).second;
				if (adj_factor_id == came_from)
					continue;

				order.push_back(OrderStep(LeafIsFactorNode,
					adj_factor_id, cur_node));

				// Add adj_factor_id to stack
				gnode_stack.push_back(adj_factor_id);
				gtype_stack.push_back(false);
				gcamefrom_stack.push_back(cur_node);
			}
		} else {
			// Parent: factor node,
			// Children: variable nodes.
			const std::vector<unsigned int>& adj_vars =
				factors[cur_node]->Variables();
			for (std::vector<unsigned int>::const_iterator
				vi = adj_vars.begin(); vi != adj_vars.end(); ++vi) {
				if (came_from == *vi)
					continue;

				order.push_back(OrderStep(LeafIsVariableNode, *vi, cur_node));

				// Add *vi to stack
				gnode_stack.push_back(*vi);
				gtype_stack.push_back(true);
				gcamefrom_stack.push_back(cur_node);
			}
		}
	}
	// Orient from leafs to root
	std::reverse(order.begin(), order.end());

	return (static_cast<unsigned int>(tree_roots.size()));
}

FactorGraphStructurizer::OrderStep::OrderStep(
	FactorGraphStructurizer::OrderStepType steptype,
	unsigned int leaf, unsigned int root)
	: steptype(steptype), leaf(leaf), root(root) {
}

unsigned int FactorGraphStructurizer::OrderStep::VariableNode() const {
	if (steptype == LeafIsVariableNode)
		return (leaf);

	return (root);
}
unsigned int FactorGraphStructurizer::OrderStep::FactorNode() const {
	if (steptype == LeafIsVariableNode)
		return (root);

	return (leaf);
}

void FactorGraphStructurizer::PrintOrder(
	const std::vector<OrderStep>& order) {
	for (unsigned int lri = 0; lri < order.size(); ++lri) {
		std::cout << "Step " << lri << ": ";
		if (order[lri].steptype ==
			FactorGraphStructurizer::LeafIsFactorNode) {
			std::cout << "factor " << order[lri].leaf << " to variable "
				<< order[lri].root << std::endl;
		} else {
			std::cout << "variable " << order[lri].leaf << " to factor "
				<< order[lri].root << std::endl;
		}
	}
}

// Implementation following the linear time (in the number of edges) algorithm
// described in (and attributed to Hierholzer):
//
// Juergen Ebert, "Computing Eulerian Trails", Information Processing Letters,
// Vol. 28, pages 93--97, 1988.
void FactorGraphStructurizer::ComputeEulerianTrail(unsigned int vertex_count,
	const std::vector<std::pair<unsigned int, unsigned int> >& arcs,
	std::vector<unsigned int>& trail) {
	trail.resize(arcs.size());
	std::fill(trail.begin(), trail.end(),	// successor array
		std::numeric_limits<unsigned int>::max());
	std::vector<unsigned int> tnew(vertex_count,	// next unprocessed arc
		std::numeric_limits<unsigned int>::max());
	// next_in[ei] is the first arc following ei and going into dest(ei)
	std::vector<unsigned int> next_in(arcs.size(),
		std::numeric_limits<unsigned int>::max());
	std::vector<unsigned int> prev_in(vertex_count,
		std::numeric_limits<unsigned int>::max());

	// initialize
	for (unsigned int ei = 0; ei < arcs.size(); ++ei) {
		unsigned int dest = arcs[ei].second;
		assert(dest < vertex_count);

		// next_in
		if (prev_in[dest] != std::numeric_limits<unsigned int>::max())
			next_in[prev_in[dest]] = ei;
		prev_in[dest] = ei;

		// tnew
		if (tnew[dest] == std::numeric_limits<unsigned int>::max())
			tnew[dest] = ei;
	}
	prev_in.clear();
	for (unsigned int vi = 0; vi < vertex_count; ++vi) {
		assert(tnew[vi] != std::numeric_limits<unsigned int>::max());
	}

	// find_first_cycle
	unsigned int last_arc = tnew[0];	// start vertex
	unsigned int out_arc = std::numeric_limits<unsigned int>::max();
	unsigned int u = 0;
	while (tnew[u] != std::numeric_limits<unsigned int>::max()) {
		unsigned int ei = tnew[u];
		tnew[u] = next_in[ei];
		trail[ei] = out_arc;
		out_arc = ei;
		u = arcs[ei].first;
	}

	// add_further_cycles
	unsigned int first_arc = out_arc;
	for (unsigned int ei = out_arc; ei != last_arc; ei = trail[ei]) {
		out_arc = trail[ei];
		unsigned int v = arcs[out_arc].first;

		// TRACE(v)
		u = v;
		while (tnew[u] != std::numeric_limits<unsigned int>::max()) {
			unsigned int edi = tnew[u];
			tnew[u] = next_in[edi];
			trail[edi] = out_arc;
			out_arc = edi;
			u = arcs[edi].first;
		}

		trail[ei] = out_arc;
	}
	trail[last_arc] = first_arc;
}

void FactorGraphStructurizer::ComputeEulerianMessageTrail(const FactorGraph* fg,
	std::vector<OrderStep>& order) {
	// Treat each connected component individually
	std::vector<unsigned int> var_label;
	unsigned int cc = ConnectedComponents(fg, var_label);
	FactorGraphUtility fgu(fg);

	for (unsigned int ci = 0; ci < cc; ++ci) {
		// Collect all variables and factors in this component
		// cc_vi_map[vi] = variable index in connected subgraph
		std::tr1::unordered_map<unsigned int, unsigned int> cc_vi_map;
		std::tr1::unordered_map<unsigned int, unsigned int> cc_vi_rmap;
		unsigned int cc_vi = 0;
		for (unsigned int vi = 0; vi < var_label.size(); ++vi) {
			if (var_label[vi] != ci)
				continue;

			cc_vi_map[vi] = cc_vi;
			cc_vi_rmap[cc_vi] = vi;
			cc_vi += 1;
		}

		std::tr1::unordered_map<unsigned int, unsigned int> cc_fi_map;
		std::tr1::unordered_map<unsigned int, unsigned int> cc_fi_rmap;
		unsigned int cc_fi = 0;
		for (std::tr1::unordered_map<unsigned int, unsigned int>::const_iterator
			vmi = cc_vi_map.begin(); vmi != cc_vi_map.end(); ++vmi) {
			unsigned int vi = vmi->first;
			const std::set<unsigned int>& adj_fac = fgu.AdjacentFactors(vi);
			for (std::set<unsigned int>::const_iterator aji = adj_fac.begin();
				aji != adj_fac.end(); ++aji) {
				unsigned int fi = *aji;
				if (cc_fi_map.find(fi) != cc_fi_map.end())
					continue;	// already included

				cc_fi_map[fi] = cc_fi;
				cc_fi_rmap[cc_fi] = fi;
				cc_fi += 1;
			}
		}
		// Variable order:
		//   {0,...,cc_vi-1} variables,
		//   {cc_vi,...,cc_vi+cc_fi-1} factors.

		// Collect all factor graph message arcs in the current component
		std::vector<std::pair<unsigned int, unsigned int> > arcs;
		for (std::tr1::unordered_map<unsigned int, unsigned int>::const_iterator
			ctvi = cc_vi_map.begin(); ctvi != cc_vi_map.end(); ++ctvi) {
			unsigned int vi = ctvi->first;
			unsigned int cvi = ctvi->second;

			const std::set<unsigned int>& adj_fac = fgu.AdjacentFactors(vi);
			for (std::set<unsigned int>::const_iterator aji = adj_fac.begin();
				aji != adj_fac.end(); ++aji) {
				unsigned int fi = *aji;
				std::tr1::unordered_map<unsigned int, unsigned int>::const_iterator
					cc_fmi = cc_fi_map.find(fi);
				assert(cc_fmi != cc_fi_map.end());
				unsigned int cfi = cc_fmi->second;

				// Add edges: vi -> fi, fi -> vi
				arcs.push_back(std::pair<unsigned int, unsigned int>(
					cvi, cc_vi + cfi));
				arcs.push_back(std::pair<unsigned int, unsigned int>(
					cc_vi + cfi, cvi));
			}
		}

		// Compute Eulerian trail
		std::vector<unsigned int> trail;
		ComputeEulerianTrail(cc_vi + cc_fi, arcs, trail);

		// Reconstruct message order
		unsigned int t0 = 0;
		unsigned int ti = t0;
		do {
			unsigned int from = arcs[trail[ti]].first;
			unsigned int to = arcs[trail[ti]].second;
			if (from < to) {
				order.push_back(OrderStep(LeafIsVariableNode,
					cc_vi_rmap[from] /*leaf*/, cc_fi_rmap[to-cc_vi] /*root*/));
			} else {
				order.push_back(OrderStep(LeafIsFactorNode,
					cc_fi_rmap[from-cc_vi] /*leaf*/, cc_vi_rmap[to] /*root*/));
			}

			// Advance
			ti = trail[ti];
		} while (ti != t0);
	}
}

}

