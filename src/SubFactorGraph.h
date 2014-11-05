
#ifndef GRANTE_SUBFACTORGRAPH_H
#define GRANTE_SUBFACTORGRAPH_H

#include <vector>
#include <tr1/unordered_map>

#include "FactorGraph.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* A subgraph of the original factor graph, derived by taking a subset of the
 * factors.  Each taken factor has a scalar weight associated with it, scaling
 * the original factor.
 */
class SubFactorGraph {
public:
	// base_fg: The larger factor graph this graph is a subgraph of.
	// f_set: The factor subset.  The factors will be added in exactly this
	//    order.
	// f_scale: Scalar multipliers for derived factor energies.
	SubFactorGraph(const FactorGraph* base_fg, std::vector<unsigned int>& f_set,
		std::vector<double>& f_scale);

	~SubFactorGraph();

	// Map energies from base_fg to the subgraph.  This does not assume
	// base_fg has up-to-date energies, that is, the base ForwardMap will be
	// called.
	void ForwardMap();

	// Maps parameter gradient from the subgraph to the original parameters
	void BackwardMap(const std::vector<std::vector<double> >& marginals,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient) const;

	// Return the subgraph instance (for inference)
	FactorGraph* FG();

	// Construct an observation on the subgraph from an observation on the
	// original graph.  The returned object is owned by the caller.
	// FIXME: only discrete observations supported for now
	FactorGraphObservation* ConstructSubObservation(
		const FactorGraphObservation* full_obs);

private:
	// The factor graph from which this one is a derived subgraph
	const FactorGraph* base_fg;

	// The actual instance of the factor graph, to be used for inference.
	FactorGraph* fg;

	// f_set[i] is the original index in base_fg of the i'th factor
	std::vector<unsigned int> f_set;
	std::vector<double> f_scale;
	// sub_var[vi] is the original variable index of variable vi
	std::vector<unsigned int> sub_var;
};

}

#endif

