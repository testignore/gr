
#include <algorithm>
#include <set>
#include <tr1/unordered_map>
#include <cassert>

#include <boost/lambda/lambda.hpp>

#include "SubFactorGraph.h"

using namespace boost::lambda;

namespace Grante {

SubFactorGraph::SubFactorGraph(const FactorGraph* base_fg,
	std::vector<unsigned int>& f_set, std::vector<double>& f_scale)
	: base_fg(base_fg), fg(0), f_set(f_set), f_scale(f_scale) {
	assert(f_set.size() == f_scale.size());

	// Obtain set of all variables in this subgraph
	std::set<unsigned int> var_set;
	const std::vector<Factor*>& facs = base_fg->Factors();
	for (std::vector<unsigned int>::const_iterator fsi = f_set.begin();
		fsi != f_set.end(); ++fsi) {
		var_set.insert(facs[*fsi]->Variables().begin(),
			facs[*fsi]->Variables().end());
	}

	// Set of variables taking part in the subgraph
	sub_var.resize(var_set.size());
	std::copy(var_set.begin(), var_set.end(), sub_var.begin());
	std::sort(sub_var.begin(), sub_var.end());

	// Cardinalities of variables
	std::vector<unsigned int> sub_card(sub_var.size());
	const std::vector<unsigned int>& base_card = base_fg->Cardinalities();
	for (unsigned int si = 0; si < sub_var.size(); ++si)
		sub_card[si] = base_card[sub_var[si]];

	// Translation table for old variable indices to new indices
	std::tr1::unordered_map<unsigned int, unsigned int> old_to_new;
	for (unsigned int vsi = 0; vsi < sub_var.size(); ++vsi)
		old_to_new[sub_var[vsi]] = vsi;

	// Create sub-factor graph
	fg = new FactorGraph(base_fg->Model(), sub_card);

	// Insert factors.  We do not copy any data (the data remains in the
	// original factor graph).
	std::vector<double> data_dummy;
	for (unsigned int fi = 0; fi < f_set.size(); ++fi) {
		Factor* old_fac = facs[f_set[fi]];

		// Translate variable indices
		const std::vector<unsigned int>& old_vars = old_fac->Variables();
		std::vector<unsigned int> new_index(old_vars.size());
		for (unsigned int n = 0; n < old_vars.size(); ++n)
			new_index[n] = old_to_new[old_vars[n]];

		Factor* new_fac = new Factor(old_fac->Type(), new_index, data_dummy);
		fg->AddFactor(new_fac);
	}
}

SubFactorGraph::~SubFactorGraph() {
	delete (fg);
}

void SubFactorGraph::ForwardMap() {
	const std::vector<Factor*>& facs = base_fg->Factors();
	const std::vector<Factor*>& new_facs = fg->Factors();
	for (unsigned int fi = 0; fi < f_set.size(); ++fi) {
		Factor* old_fac = facs[f_set[fi]];
		Factor* new_fac = new_facs[fi];

		// Subgraph energies: scaled base-graph energies
		old_fac->ForwardMap(true);
		// Force explicit allocation in new factor
		new_fac->EnergiesAllocate(true);
		std::transform(old_fac->Energies().begin(), old_fac->Energies().end(),
			new_fac->Energies().begin(), f_scale[fi] * _1);
	}
}

void SubFactorGraph::BackwardMap(
	const std::vector<std::vector<double> >& marginals,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const {
	const std::vector<Factor*>& facs = fg->Factors();
	assert(facs.size() == marginals.size());
	for (unsigned int fi = 0; fi < f_set.size(); ++fi) {
		facs[fi]->BackwardMap(marginals[fi],
			parameter_gradient[facs[fi]->Type()->Name()],
			f_scale[fi]);
	}
}

FactorGraph* SubFactorGraph::FG() {
	return (fg);
}

FactorGraphObservation* SubFactorGraph::ConstructSubObservation(
	const FactorGraphObservation* full_obs) {
	if (full_obs->Type() == FactorGraphObservation::DiscreteLabelingType) {
		// Discrete observation: map variables
		std::vector<unsigned int> sobs_s(sub_var.size());
		const std::vector<unsigned int>& obs_s = full_obs->State();
		for (unsigned int vi = 0; vi < sub_var.size(); ++vi)
			sobs_s[vi] = obs_s[sub_var[vi]];

		return (new FactorGraphObservation(sobs_s));
	} else if (full_obs->Type() == FactorGraphObservation::ExpectationType) {
		// Expectation observation: copy and reorder factor observation
		assert(0);
#if 0
		const std::vector<std::vector<double> >& obs_e =
			full_obs->Expectation();
		std::vector<std::vector<double> > sobs_e;
		sobs_e.reserve(f_set.size());
		for (unsigned int fi = 0; fi < f_set.size(); ++fi) {
			// TODO: remap variables within expectation
			std::vector<unsigned int> f_set;
			sobs_e.push_back(obs_e[fi]);
		}
		return (new FactorGraphObservation(sobs_e));
#endif
	}
	assert(0);
	return (0);
}

}

