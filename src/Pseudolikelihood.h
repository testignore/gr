
#ifndef GRANTE_PSEUDOLIKELIHOOD_H
#define GRANTE_PSEUDOLIKELIHOOD_H

#include <vector>
#include <string>
#include <tr1/unordered_map>

#include "FactorGraphModel.h"
#include "FactorGraph.h"
#include "FactorGraphUtility.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* Log-Pseudolikelihood and gradient of log-pseudolikelihood computation.
 *
 * This class is very similar to the Likelihood class, however only
 * the conditional marginal distributions of each variable is needed, not the
 * full joint marginal distribution of each factor.
 */
class Pseudolikelihood {
public:
	explicit Pseudolikelihood(const FactorGraphModel* fg_model);

	// cond_site_marginals: [var_index] is a normalized distribution over the
	//    site states.
	// cond_log_z_sum: log-partition function of conditional p(y_i|y_{V\{i}})
	//    distributions, summed over all i in V.
	double ComputeNegLogPseudolikelihood(const FactorGraph* fg,
		const FactorGraphUtility* fgu,
		const FactorGraphObservation* obs,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient) const;

private:
	const FactorGraphModel* fg_model;

	double ComputeNegLogPseudolikelihood(const FactorGraph* fg,
		const FactorGraphUtility* fgu,
		const std::vector<unsigned int>& observed_state,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient) const;

	double ComputeNegLogPseudolikelihood(const FactorGraph* fg,
		const FactorGraphUtility* fgu,
		const std::vector<std::vector<double> >& observed_expectations,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient) const;
};

}

#endif

