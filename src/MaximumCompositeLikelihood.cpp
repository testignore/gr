
#include <tr1/unordered_set>
#include <numeric>
#include <iostream>
#include <cassert>

#include <boost/timer.hpp>

#include "Conditioning.h"
#include "ConditionedFactorType.h"
#include "FactorGraphObservation.h"
#include "FactorGraphStructurizer.h"
#include "FactorGraphPartialObservation.h"
#include "VAcyclicDecomposition.h"
#include "RandomSource.h"
#include "MaximumCompositeLikelihood.h"

namespace Grante {

MaximumCompositeLikelihood::MaximumCompositeLikelihood(
	FactorGraphModel* fg_model, int decomp)
	: ParameterEstimationMethod(fg_model), decomp(decomp), mle(fg_model) {
	assert(decomp >= -1);
}

MaximumCompositeLikelihood::~MaximumCompositeLikelihood() {
	assert(comp_training_data.size() == comp_inference_methods.size());
	for (unsigned int cn = 0; cn < comp_training_data.size(); ++cn) {
		delete (comp_training_data[cn].first);
		delete (comp_training_data[cn].second);
		delete (comp_inference_methods[cn]);
	}
}

void MaximumCompositeLikelihood::SetOptimizationMethod(
	MaximumLikelihood::MLEOptimizationMethod opt_method) {
	mle.SetOptimizationMethod(opt_method);
}

MaximumLikelihood::MLEProblem* MaximumCompositeLikelihood::GetLearnProblem() {
	return (mle.GetLearnProblem());
}

void MaximumCompositeLikelihood::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*> inference_methods) {
	assert(comp_training_data.size() == 0);
	assert(comp_inference_methods.size() == 0);
	assert(inference_methods.size() == training_data.size());

	// Number of times each component will be covered
	unsigned int cover_count = 1;
	assert(decomp >= -1);
	if (decomp == DecomposePseudolikelihood) {
		cover_count = 1;
	} else if (decomp > 0) {
		cover_count = decomp;
	}

	// Produce composite factor graphs
	boost::timer decomp_timer;
	int training_data_size = static_cast<int>(training_data.size());
	fg_cc_var_label.resize(cover_count * training_data_size);
	fg_cc_count.resize(cover_count * training_data_size);
	fg_orig_index.resize(cover_count * training_data_size);
	std::fill(fg_cc_count.begin(), fg_cc_count.end(), 0);
	unsigned int cn = 0;
	for (int n = 0; n < training_data_size; ++n) {
		FactorGraph* fg = training_data[n].first;
		size_t var_count = fg->Cardinalities().size();

		// Get observation
		const FactorGraphObservation* obs = training_data[n].second;

		// Obtain one or more decomposition(s)
		for (unsigned int cover_iter = 0; cover_iter < cover_count;
			++cover_iter) {
			VAcyclicDecomposition vac(fg);
			std::vector<bool> factor_is_removed;

			if (decomp == DecomposePseudolikelihood) {
				factor_is_removed.resize(fg->Factors().size());
				std::fill(factor_is_removed.begin(),
					factor_is_removed.end(), true);
			} else {
				std::vector<double> factor_weight(fg->Factors().size(), 0.0);
				if (decomp == DecomposeUniform) {
					// Use constant weights
					std::fill(factor_weight.begin(), factor_weight.end(), 1.0);
				} else {
					// Use uniform random weights
					boost::uniform_real<double> uniform_dist(0.0, 1.0);
					boost::variate_generator<boost::mt19937&,
						boost::uniform_real<double> >
						rgen(RandomSource::GlobalRandomSampler(), uniform_dist);

					for (unsigned int fi = 0; fi < factor_weight.size(); ++fi)
						factor_weight[fi] = rgen();
				}
				vac.ComputeDecompositionSP(factor_weight, factor_is_removed);
			}

			// Shatter factor graph into trees
			fg_cc_count[cn] += FactorGraphStructurizer::ConnectedComponents(
				fg, factor_is_removed, fg_cc_var_label[cn]);
#if 0
			std::cout << "MCL, instance " << n << " decomposed into " << cc_count
				<< " components" << std::endl;
#endif

			// Add each component as separate factor graph
			for (unsigned int ci = 0; ci < fg_cc_count[cn]; ++ci) {
				std::vector<unsigned int> cond_var_set;
				cond_var_set.reserve(var_count);

				// Add all variables not in this component to the conditioning set
				for (size_t vi = 0; vi < var_count; ++vi) {
					if (fg_cc_var_label[cn][vi] != ci)
						cond_var_set.push_back(static_cast<unsigned int>(vi));
				}
				AddTrainingComponentCond(fg, obs, inference_methods[n],
					cond_var_set);
			}
			fg_orig_index[cn] = n;
			cn += 1;
		}
	}
	std::cout << "MCL, decomposed " << training_data.size() << " instances "
		<< "into " << comp_training_data.size() << " instances "
		<< (decomp == DecomposeUniform ? "(uniform)" : "(randomized)")
		<< " in " << decomp_timer.elapsed() << "s." << std::endl;

	// Initialize MLE training data from created components
	SetupMLETrainingData();
}

void MaximumCompositeLikelihood::SetupMLETrainingData() {
	mle.SetupTrainingData(comp_training_data, comp_inference_methods);
}

void MaximumCompositeLikelihood::AddTrainingComponentUncond(
	const FactorGraph* fg, const FactorGraphObservation* obs,
	InferenceMethod* inference_method,
	const std::vector<unsigned int>& uncond_var_set) {
	// Produce conditioning variable set A = V\B, where B is uncond_var_set.
	std::vector<unsigned int> cond_var_set;
	size_t var_count = fg->Cardinalities().size();
	cond_var_set.reserve(var_count);

	// Add all variables not in this component to the conditioning set
	unsigned int uncond_var_count = 0;
	std::tr1::unordered_set<unsigned int> uncond_var_set_h(
		uncond_var_set.begin(), uncond_var_set.end());
	for (size_t vi = 0; vi < var_count; ++vi) {
		if (uncond_var_set_h.count(static_cast<unsigned int>(vi)) > 0) {
			uncond_var_count += 1;
			continue;
		}
		cond_var_set.push_back(static_cast<unsigned int>(vi));
	}

	// Add component based on conditioning variable set
	AddTrainingComponentCond(fg, obs, inference_method, cond_var_set);
}

void MaximumCompositeLikelihood::AddTrainingComponentCond(
	const FactorGraph* fg, const FactorGraphObservation* obs,
	InferenceMethod* inference_method,
	const std::vector<unsigned int>& cond_var_set) {
	// Create partial observation from full observation
	FactorGraphPartialObservation* pobs =
		CreatePartialObservationCond(fg, obs, cond_var_set);

	// Condition
	std::vector<unsigned int> var_new_to_orig;
	std::vector<unsigned int> fac_new_to_orig;
	FactorGraph* fg_cond = Conditioning::ConditionFactorGraph(
		&ftab, fg, pobs, var_new_to_orig, fac_new_to_orig);
	delete (pobs);

	FactorGraphObservation* new_obs =
		CreatePartialObservationUncond(fg, fg_cond, obs,
			var_new_to_orig, fac_new_to_orig);

	// Build derived composite training set
	comp_training_data.push_back(labeled_instance_type(fg_cond, new_obs));
	assert(inference_method != 0);
	comp_inference_methods.push_back(inference_method->Produce(fg_cond));
}

void MaximumCompositeLikelihood::UpdateTrainingComponentCond(
	const FactorGraph* fg, const FactorGraphObservation* obs,
	const std::vector<unsigned int>& cond_var_set,
	unsigned int cti) {
	assert(cti < comp_training_data.size());

	// Create partial observation from full observation
	FactorGraphPartialObservation* pobs =
		CreatePartialObservationCond(fg, obs, cond_var_set);

	// Obtain variable and factor maps
	std::vector<unsigned int> var_new_to_orig;
	std::vector<unsigned int> fac_new_to_orig;
	FactorGraph* fg_cond = Conditioning::ConditionFactorGraph(
		&ftab, fg, pobs, var_new_to_orig, fac_new_to_orig);
	delete (pobs);

	FactorGraphObservation* new_obs =
		CreatePartialObservationUncond(fg, fg_cond, obs,
			var_new_to_orig, fac_new_to_orig);
	delete (fg_cond);

	// Update observation
	assert(comp_training_data[cti].second->Type() == new_obs->Type());
	if (new_obs->Type() == FactorGraphObservation::DiscreteLabelingType) {
		assert(comp_training_data[cti].second->State().size() ==
			new_obs->State().size());
	} else {
		assert(comp_training_data[cti].second->Expectation().size() ==
			new_obs->Expectation().size());
	}
	delete (comp_training_data[cti].second);
	comp_training_data[cti].second = new_obs;
}

// This method's responsibility is to create the information the subgraph is
// conditioned on.  For discrete observations this is simply the conditioning
// variable state adjacent to cross-factors.  For expectation observations
// this is the factor expectation with the unconditioned variables
// marginalized out.
FactorGraphPartialObservation*
MaximumCompositeLikelihood::CreatePartialObservationCond(
	const FactorGraph* fg, const FactorGraphObservation* obs,
	const std::vector<unsigned int>& cond_var_set) const {
	if (obs->Type() == FactorGraphObservation::DiscreteLabelingType) {
		size_t cond_var_count = cond_var_set.size();
		std::vector<unsigned int> cond_var_state;
		cond_var_state.reserve(cond_var_count);

		// Add all variables not in this component to the conditioning set
		const std::vector<unsigned int>& obs_state = obs->State();
		for (size_t cvi = 0; cvi < cond_var_count; ++cvi) {
			// Condition
			unsigned int vi = cond_var_set[cvi];
			cond_var_state.push_back(obs_state[vi]);
		}

		// Create conditioning observation
		return (new FactorGraphPartialObservation(cond_var_set,
			cond_var_state));
	} else {
		assert(obs->Type() == FactorGraphObservation::ExpectationType);
		std::vector<unsigned int> fac_subset;
		std::vector<std::vector<double> > obs_e;
		const std::vector<std::vector<double> >& expect = obs->Expectation();
		const std::vector<Factor*>& factors = fg->Factors();
		std::tr1::unordered_set<unsigned int> cond_var_set_u(
			cond_var_set.begin(), cond_var_set.end());

		// Sum out the unconditioned variable elements.
		//
		// For discrete observations we can identify the conditioning
		// distribution uniquely.  For expectation observations we can not.
		// But as the expectation is assumed to be realizable by the model
		// (ideally a product distribution), we sum out the unconditioned
		// variables to obtain the per-site conditioning expectations.
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			// Check whether this factor is a cross factor
			const std::vector<unsigned int>& fac_vars =
				factors[fi]->Variables();
			const std::vector<unsigned int>& fac_card =
				factors[fi]->Cardinalities();

			bool has_cond = false;
			bool has_uncond = false;
			std::vector<unsigned int> cond_fvar;
			unsigned int cond_card = 1;
			for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
				if (cond_var_set_u.count(fac_vars[fvi]) > 0) {
					has_cond = true;
					cond_fvar.push_back(fvi);
					cond_card *= fac_card[fvi];
				} else {
					has_uncond = true;
				}
			}
			if (has_cond == false || has_uncond == false)
				continue;

			// This factor is a cross-factor
			fac_subset.push_back(fi);
			const std::vector<double>& m_e = expect[fi];
			std::vector<double> obs_e_fi(cond_card, 0.0);
			for (unsigned int oei = 0; oei < m_e.size(); ++oei) {
				// oei: index in original factor expectations,
				// cei: index into conditioning expectations.
				unsigned int cei =
					FactorConditioningTable::IndexMapConditioned(
						factors[fi]->Type(), cond_fvar, oei);
				assert(cei < obs_e_fi.size());

				// Marginalize out unconditioned variables
				obs_e_fi[cei] += m_e[oei];
			}
			assert(std::fabs(std::accumulate(obs_e_fi.begin(),
				obs_e_fi.end(), 0.0) - 1.0) <= 1.0e-8);
			obs_e.push_back(obs_e_fi);
		}
		return (new FactorGraphPartialObservation(cond_var_set,
			fac_subset, obs_e));
	}
	assert(0);
	return (0);
}

// This method's responsibility is to create the new target ground truth for
// the subgraphs.  For discrete observations this is the variable subset
// covered by the graph, whereas for expectation observations this are the
// factor observations with the conditioned-on variables marginalized out.
FactorGraphObservation*
MaximumCompositeLikelihood::CreatePartialObservationUncond(
	const FactorGraph* fg, const FactorGraph* fg_cond,
	const FactorGraphObservation* obs,
	const std::vector<unsigned int>& var_new_to_orig,
	const std::vector<unsigned int>& fac_new_to_orig) const {
	if (obs->Type() == FactorGraphObservation::DiscreteLabelingType) {
		// Create matching observation
		size_t uncond_var_count = var_new_to_orig.size();
		const std::vector<unsigned int>& obs_state = obs->State();
		std::vector<unsigned int> new_obs_state(uncond_var_count);
		for (size_t nvi = 0; nvi < uncond_var_count; ++nvi) {
			assert(nvi < var_new_to_orig.size());
			assert(var_new_to_orig[nvi] < obs_state.size());
			new_obs_state[nvi] = obs_state[var_new_to_orig[nvi]];
		}
		return (new FactorGraphObservation(new_obs_state));
	} else {
		assert(obs->Type() == FactorGraphObservation::ExpectationType);
		size_t new_fac_count = fac_new_to_orig.size();

		// For each conditioned factor
		const std::vector<std::vector<double> >& expect = obs->Expectation();
		const std::vector<Factor*>& factors = fg->Factors();
		const std::vector<Factor*>& cond_factors = fg_cond->Factors();
		std::vector<std::vector<double> > obs_e;
		for (size_t nfi = 0; nfi < new_fac_count; ++nfi) {
			const ConditionedFactorType* cft =
				dynamic_cast<const ConditionedFactorType*>(
					cond_factors[nfi]->Type());
			assert(cft != 0);
			const std::vector<unsigned int>& cond_fvar =
				cft->ConditionedVariableIndices();

			// Create unconditioned expectations
			std::vector<double> obs_e_fi(cft->ProdCardinalities(), 0.0);
			unsigned int ofi = fac_new_to_orig[nfi];
			const std::vector<double>& m_e = expect[ofi];
			if (cond_fvar.empty()) {
				// Fully unconditioned factor: copy ground truth expectations
				assert(m_e.size() == obs_e_fi.size());
				std::copy(m_e.begin(), m_e.end(), obs_e_fi.begin());
			} else {
				// Conditioned factor
				for (unsigned int oei = 0; oei < m_e.size(); ++oei) {
					// oei: index in original factor expectations,
					// nei: index in remaining unconditioned variable
					//    'ground-truth' expectations.
					unsigned int nei =
						FactorConditioningTable::IndexMapUnconditioned(
							factors[ofi]->Type(), cond_fvar, oei);
					assert(nei < obs_e_fi.size());

					// Marginalize out conditioned variables
					obs_e_fi[nei] += m_e[oei];
				}
			}
			assert(std::fabs(std::accumulate(obs_e_fi.begin(),
				obs_e_fi.end(), 0.0) - 1.0) <= 1.0e-8);
			obs_e.push_back(obs_e_fi);
		}
		return (new FactorGraphObservation(obs_e));
	}
	assert(0);
	return (0);
}

void MaximumCompositeLikelihood::AddPrior(const std::string& factor_type,
	Prior* prior) {
	mle.AddPrior(factor_type, prior);
}

double MaximumCompositeLikelihood::Train(double conv_tol, unsigned int max_iter)
{
	return (mle.Train(conv_tol, max_iter));
}

void MaximumCompositeLikelihood::UpdateTrainingLabeling(
	const std::vector<labeled_instance_type>& training_update) {
	assert(fg_orig_index.size() == comp_training_data.size());

	// For all decomposed components
	for (unsigned int cn = 0; cn < fg_orig_index.size(); ++cn) {
		// Original factor graph index
		unsigned int n = fg_orig_index[cn];
		assert(n < training_update.size());

		FactorGraph* fg = training_update[n].first;
		const FactorGraphObservation* obs = training_update[n].second;
		size_t var_count = fg->Cardinalities().size();

		// Update each component of the current decomposition
		for (unsigned int ci = 0; ci < fg_cc_count[cn]; ++ci) {
			std::vector<unsigned int> cond_var_set;
			cond_var_set.reserve(var_count);

			// Add all variables not in this component to the conditioning set
			for (size_t vi = 0; vi < var_count; ++vi) {
				if (fg_cc_var_label[cn][vi] != ci)
					cond_var_set.push_back(static_cast<unsigned int>(vi));
			}
			UpdateTrainingComponentCond(fg, obs, cond_var_set, cn);
		}
	}

	// Update fully observed components
	mle.UpdateTrainingLabeling(comp_training_data);
}

}

