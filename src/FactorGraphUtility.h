
#ifndef GRANTE_FACTORGRAPHUTILITY_H
#define GRANTE_FACTORGRAPHUTILITY_H

#include <vector>
#include <set>
#include <tr1/unordered_map>

#include "FactorGraph.h"

namespace Grante {

/* Efficient utility functions for factor graphs that incur an additional
 * memory overhead.
 */
class FactorGraphUtility {
public:
	explicit FactorGraphUtility(const FactorGraph* fg);

	// Compute an unnormalized conditional distribution for a single variable
	// (site).
	//
	// test_state: State of all variables, where test_state[var_index] will be
	//    ignored.  The vector is left unchanged.
	// var_index: The site index to compute the conditional distribution of.
	// cond_dist_unnorm: (output) unnormalized distribution.
	// temp: optional temperature.
	//
	// The return value is the normalization constant for cond_dist_unnorm.
	double ComputeConditionalSiteDistribution(
		std::vector<unsigned int>& test_state,
		unsigned int var_index, std::vector<double>& cond_dist_unnorm,
		double temp = 1.0) const;

	// Compute energy difference E(new)-E(old) that is incurred by changing
	// the single given variable from old_state to new_state.
	double ComputeEnergyChange(std::vector<unsigned int>& state,
		unsigned int var_index,
		unsigned int old_state, unsigned int new_state) const;

	// Return the set of factor indices that is adjacent to a variable.
	const std::set<unsigned int>& AdjacentFactors(
		unsigned int var_index) const;

private:
	const FactorGraph* fg;

	// [var_index] = { factor indices variable appears in }
	std::tr1::unordered_map<unsigned int, std::set<unsigned int> >
		var_to_factorset;
};

}

#endif

