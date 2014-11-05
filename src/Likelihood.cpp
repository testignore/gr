
#include <iostream>
#include <numeric>
#include <cmath>
#include <cassert>

#include "Likelihood.h"

namespace Grante {

Likelihood::Likelihood(const FactorGraphModel* fg_model)
	: fg_model(fg_model) {
}

double Likelihood::ComputeNegLogLikelihood(const FactorGraph* fg,
	const FactorGraphObservation* obs,
	const std::vector<std::vector<double> >& marginals, double log_z,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const {

	// PART 1: E(y_n;x_n,w) term
	double nloglikelihood = ComputeObservationEnergy(fg, obs,
		parameter_gradient);

	// PART 2: -log Z term
	nloglikelihood += ComputeNegLogLikelihoodNegLogZTerm(fg, marginals,
		log_z, parameter_gradient);

	return (nloglikelihood);
}

double Likelihood::ComputeObservationEnergy(const FactorGraph* fg,
	const FactorGraphObservation* obs,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, double scale) const {
	if (obs->Type() == FactorGraphObservation::DiscreteLabelingType) {
		return (ComputeObservationEnergy(fg, obs->State(),
			parameter_gradient, scale));
	} else {
		assert(obs->Type() == FactorGraphObservation::ExpectationType);
		return (ComputeObservationEnergy(fg, obs->Expectation(),
			parameter_gradient, scale));
	}
}

double Likelihood::ComputeObservationEnergy(const FactorGraph* fg,
	const std::vector<unsigned int>& observed_state,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, double scale) const {
	// PART 1: Compute energy gradient of observations
	assert(observed_state.size() == fg->Cardinalities().size());

	// Setup temporary marginal distributions (key is size)
	std::tr1::unordered_map<size_t, std::vector<double> > temp_marginals;
	const std::vector<Factor*>& factors = fg->Factors();
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		const FactorType* fti = (*fi)->Type();
		if (temp_marginals.find(fti->ProdCardinalities()) !=
			temp_marginals.end())
			continue;
		temp_marginals[fti->ProdCardinalities()] =
			std::vector<double>(fti->ProdCardinalities(), 0.0);
	}

	// Compute energy gradient for factor
	double nloglikelihood = 0.0;
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		const Factor* factor = *fi;
		std::vector<double>& temp_f_m =
			temp_marginals[factor->Type()->ProdCardinalities()];

		// Set to 1, evaluate, set to 0
		// FIXME: for regtree factors this needs to map to the cell id
		unsigned int ei = factor->ComputeAbsoluteIndex(observed_state);
		assert(ei < temp_f_m.size());
		temp_f_m[ei] = 1.0;
		factor->BackwardMap(temp_f_m,
			parameter_gradient[factor->Type()->Name()], scale);
		temp_f_m[ei] = 0.0;

		nloglikelihood += factor->Energies()[ei];
	}
	return (scale * nloglikelihood);
}

double Likelihood::ComputeObservationEnergy(const FactorGraph* fg,
	const std::vector<std::vector<double> >& observed_expectations,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, double scale) const {
	assert(fg->Factors().size() == observed_expectations.size());

	// PART 1: Compute energy from the expectation
	double nloglikelihood = 0.0;
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		assert(std::fabs(std::accumulate(observed_expectations[fi].begin(),
			observed_expectations[fi].end(), 0.0) - 1.0) < 1e-5);
		assert(observed_expectations[fi].size() ==
			factors[fi]->Energies().size());

		factors[fi]->BackwardMap(observed_expectations[fi],
			parameter_gradient[factors[fi]->Type()->Name()], scale);

		nloglikelihood += std::inner_product(
			observed_expectations[fi].begin(), observed_expectations[fi].end(),
			factors[fi]->Energies().begin(), 0.0);
	}
	return (scale * nloglikelihood);
}

double Likelihood::ComputeNegLogLikelihoodNegLogZTerm(
	const FactorGraph* fg,
	const std::vector<std::vector<double> >& marginals, double log_z,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const {
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// \nabla_w: - \expect_{y~p(y|x,w)}[ \nabla_w E(y;x,w) ]
		factors[fi]->BackwardMap(marginals[fi],
			parameter_gradient[factors[fi]->Type()->Name()], -1.0);
	}
	return (log_z);
}

}

