
#include <iostream>
#include <cassert>

#include "GibbsSampler.h"
#include "Factor.h"
#include "FactorType.h"
#include "ContrastiveDivergence.h"

namespace Grante {

ContrastiveDivergence::ContrastiveDivergence(FactorGraphModel* fg_model,
	unsigned int cd_k)
	: model(fg_model), cd_k(cd_k) {
	assert(cd_k > 0);

	// Setup temporary marginal distributions (key is size)
	const std::vector<FactorType*>& ftypes = model->FactorTypes();
	for (std::vector<FactorType*>::const_iterator fti = ftypes.begin();
		fti != ftypes.end(); ++fti) {
		const FactorType* ftype = *fti;
		size_t pcard = ftype->ProdCardinalities();
		if (temp_marginals.find(pcard) != temp_marginals.end())
			continue;
		temp_marginals[pcard] = std::vector<double>(pcard, 0.0);
	}
}

void ContrastiveDivergence::ComputeGradientFullyObserved(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, const FactorGraph* fg,
	const FactorGraphObservation* obs) const {
	// we can only work with discrete observations for now
	assert(obs->Type() == FactorGraphObservation::DiscreteLabelingType);

	// Ground truth state
	const std::vector<unsigned int>& y_obs = obs->State();

	// Obtain a sample from the model distribution by starting a Gibbs sampler
	// at the ground truth labeling
	GibbsSampler model_sampler(fg);
	model_sampler.SetState(y_obs);	// initialize with truth
	model_sampler.Sweep(cd_k);	// walk away from truth
	const std::vector<unsigned int>& y_model = model_sampler.State();

	// Compute: \nabla_w E(y_obs,x,w) - \nabla_w E(y_model,x,w)
	AddBackwardMap(parameter_gradient, fg, y_obs, 1.0);
	AddBackwardMap(parameter_gradient, fg, y_model, -1.0);
}

void ContrastiveDivergence::ComputeGradientPartiallyObserved(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, const FactorGraph* fg,
	const FactorGraphPartialObservation* pobs) const {
	// restricted to discrete observations for now
	assert(pobs->Type() == FactorGraphObservation::DiscreteLabelingType);

	// Get partial observations
	const std::vector<unsigned int>& pobs_vars = pobs->ObservedVariableSet();
	const std::vector<unsigned int>& pobs_states =
		pobs->ObservedVariableState();

	// Obtain y_pobs: fix all observed variables, sample the hidden variables.
	GibbsSampler model_sampler(fg);
	model_sampler.SetStateUniformRandom();	// initialize all variables
	for (unsigned int voi = 0; voi < pobs_vars.size(); ++voi)
		model_sampler.SetState(pobs_vars[voi], pobs_states[voi]);
	model_sampler.SetFixedVariableIndices(pobs_vars);	// fix

	model_sampler.Sweep(cd_k);	// walk away on hiddens, but fix observations
	const std::vector<unsigned int>& y_pobs = model_sampler.State();
	AddBackwardMap(parameter_gradient, fg, y_pobs, 1.0);

	// Obtain y_model: freely sample all variables
	model_sampler.SetFixedVariableIndices();
	model_sampler.Sweep(cd_k);	// walk away on all variables
	const std::vector<unsigned int>& y_model = model_sampler.State();
	AddBackwardMap(parameter_gradient, fg, y_model, -1.0);
}

void ContrastiveDivergence::AddBackwardMap(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, const FactorGraph* fg,
		const std::vector<unsigned int>& y, double scale) const {
	// Go over all factors
	const std::vector<Factor*>& factors = fg->Factors();
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		const Factor* factor = *fi;
		std::vector<double>& temp_f_m =
			temp_marginals[factor->Type()->ProdCardinalities()];

		// Compute "marginal" distribution from a single sample
		// TODO: this can be made much more efficient if we would have a
		// BackwardMap supporting a discrete label (with fallback)
		unsigned int ei = factor->ComputeAbsoluteIndex(y);
		assert(ei < temp_f_m.size());
		temp_f_m[ei] = 1.0;
		factor->BackwardMap(temp_f_m,
			parameter_gradient[factor->Type()->Name()], scale);
		temp_f_m[ei] = 0.0;
	}
}

}

