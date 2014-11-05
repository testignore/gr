
#ifndef GRANTE_LIKELIHOOD_H
#define GRANTE_LIKELIHOOD_H

#include <vector>
#include <string>
#include <tr1/unordered_map>

#include "FactorGraphModel.h"
#include "FactorGraph.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* Log-likelihood and gradient of log-likelihood computations.
 *
 * Given a FactorGraphModel, a factor graph instantiating this model, and a
 * fully-observed instance, compute the log-likelihood and the parameter
 * gradient of the log-likelihood for this instance.
 *
 * Observations can be given in two forms through FactorGraphObservation,
 *    1. As discrete states (labeling), or
 *    2. As a set of locally-consistent distributions defined for each factor
 *       (expectations).
 */
class Likelihood {
public:
	explicit Likelihood(const FactorGraphModel* fg_model);

	// Compute the negative log-likelihood of an observation.
	//
	// fg: The factorgraph realizing the factor graph model.  Note that the
	//    energies of this factor graph must be up-to-date for the
	//    computations to be correct.
	// obs: A factor graph observation (either discrete labeling or marginal
	//    expected label distribution).
	// marginals: Exact or approximate marginals for this factor graph.
	// log_z: Exact or approximate log partition function.  Only used for
	//    computing the objective.
	// parameter_gradient: (out) a properly sized set of vectors to which the
	//    gradient of the negative log-likelihood will be added.
	//
	// Returns the negative log-likelihood.
	double ComputeNegLogLikelihood(const FactorGraph* fg,
		const FactorGraphObservation* obs,
		const std::vector<std::vector<double> >& marginals, double log_z,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient) const;

	// Compute the energy of an observation and its gradient.  This is the
	// first term of the negative log-likelihood objective.
	//
	// All objective and parameter-gradient computations are scaled with the
	// given scalar.
	double ComputeObservationEnergy(const FactorGraph* fg,
		const FactorGraphObservation* obs,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient, double scale = 1.0) const;

	// Compute the observation energy (first term of the negative
	// log-likelihood) of an observation based on a discrete labeling.
	double ComputeObservationEnergy(const FactorGraph* fg,
		const std::vector<unsigned int>& observed_state,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient, double scale) const;

	// Compute the observation energy (first term of the negative
	// log-likelihood) of an observation given as expectation.
	//
	// Same as the previous method but the observation is given as a
	// distribution over states, one per factor of the model.  This is useful
	// at least for two applications,
	//    1. In case the true observation is a distribution over states, and
	//    2. For algorithms such as EM that depend on likelihood computations
	//       with expectations.
	double ComputeObservationEnergy(const FactorGraph* fg,
		const std::vector<std::vector<double> >& observed_expectations,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient, double scale) const;

private:
	const FactorGraphModel* fg_model;

	// Compute -log Z and its gradient
	double ComputeNegLogLikelihoodNegLogZTerm(const FactorGraph* fg,
		const std::vector<std::vector<double> >& marginals, double log_z,
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient) const;
};

}

#endif

