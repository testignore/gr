
#include <cmath>
#include <cassert>

#include "Pseudolikelihood.h"

namespace Grante {

Pseudolikelihood::Pseudolikelihood(const FactorGraphModel* fg_model)
	: fg_model(fg_model) {
}

double Pseudolikelihood::ComputeNegLogPseudolikelihood(const FactorGraph* fg,
	const FactorGraphUtility* fgu, const FactorGraphObservation* obs,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const
{
	if (obs->Type() == FactorGraphObservation::DiscreteLabelingType) {
		return (ComputeNegLogPseudolikelihood(fg, fgu, obs->State(),
			parameter_gradient));
	} else {
		assert(obs->Type() == FactorGraphObservation::ExpectationType);
		return (ComputeNegLogPseudolikelihood(fg, fgu, obs->Expectation(),
			parameter_gradient));
	}
}

// Evaluate negative log-pseudolikelihood nlp(w) and its parameter gradient
// nlp(w) = (1/N) sum_i nlp_i(w)
// nlp_i(w) = sum_{f in F(i)} f(y^*_i, w)
//   + log sum_{y_i in Y_i} exp( -sum_{f in F(i)} f(y_i, y^*_{V \ {i}}, w) )
//
// The gradient \nabla_w nlp_i(w) is
// \nabla_w nlp_i(w) = sum_{f in F(i)} \nabla_w f(y^*, w)
//   - \expect_{y ~ p(y_i | y^*_{V \ {i}}, w)}[
//        sum_{f in F(i)} \nabla_w f(y_i, y^*_{V \ {i}}, w)].
double Pseudolikelihood::ComputeNegLogPseudolikelihood(const FactorGraph* fg,
	const FactorGraphUtility* fgu,
	const std::vector<unsigned int>& observed_state,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const
{
	assert(observed_state.size() == fg->Cardinalities().size());

	// Prepare variable conditional marginal distributions
	const std::vector<unsigned int>& var_card = fg->Cardinalities();
	std::vector<std::vector<double> > cond_site_marginals(var_card.size());
	std::vector<double> cond_log_z_sum(var_card.size());

	// TODO: avoid copy
	std::vector<unsigned int> state_cur(observed_state);
	for (unsigned int vi = 0; vi < var_card.size(); ++vi) {
		cond_site_marginals[vi].resize(var_card[vi], 0.0);
		// Compute site marginals
		cond_log_z_sum[vi] = fgu->ComputeConditionalSiteDistribution(
			state_cur, vi, cond_site_marginals[vi]);

		// Normalize
		for (unsigned int vsi = 0; vsi < var_card[vi]; ++vsi)
			cond_site_marginals[vi][vsi] /= cond_log_z_sum[vi];

		// Local log partition function
		cond_log_z_sum[vi] = std::log(cond_log_z_sum[vi]);
	}

	// Setup temporary marginal distributions (key is size)
	std::tr1::unordered_map<size_t, std::vector<double> >
		temp_marginals;
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

	double nll = 0.0;

	// PART 1: Compute energy gradient of observations
	// For all sites,
	double scale = 1.0 / static_cast<double>(var_card.size());
	for (unsigned int vi = 0; vi < var_card.size(); ++vi) {
		const std::set<unsigned int>& adj_factors =
			fgu->AdjacentFactors(vi);

		// For all adjacent factors,
		for (std::set<unsigned int>::const_iterator afi = adj_factors.begin();
			afi != adj_factors.end(); ++afi) {
			// Get factor
			const Factor* factor = factors[*afi];
			const std::string& ftype_name = factor->Type()->Name();
			std::vector<double>& temp_f_m =
				temp_marginals[factor->Type()->ProdCardinalities()];

			// Compute -log p(y^*_F|w)
			unsigned int ei = factor->ComputeAbsoluteIndex(observed_state);
			temp_f_m[ei] = 1.0;
			factor->BackwardMap(temp_f_m, parameter_gradient[ftype_name], scale);
			temp_f_m[ei] = 0.0;

			// + f(y^*_i, w)
			nll += scale * factor->Energies()[ei];

			// Compute
			// -E_{y~p(y_i|y^*_{V\{i}},w)}[\nabla_w f(y,y^*_{V\{i}},w)]
			factor->ExpandVariableMarginalToFactorMarginal(observed_state,
				vi, cond_site_marginals[vi], temp_f_m);
			factor->BackwardMap(temp_f_m, parameter_gradient[ftype_name],
				-scale);
			std::fill(temp_f_m.begin(), temp_f_m.end(), 0.0);
		}
		nll += scale*cond_log_z_sum[vi];
	}

	return (nll);
}

double Pseudolikelihood::ComputeNegLogPseudolikelihood(const FactorGraph* fg,
	const FactorGraphUtility* fgu,
	const std::vector<std::vector<double> >& observed_expectations,
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const
{
	// TODO
	assert(0);	// not implemented yet
	return (0.0);
}

}

