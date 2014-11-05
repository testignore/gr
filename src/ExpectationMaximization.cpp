
#include <algorithm>
#include <limits>
#include <iostream>
#include <tr1/unordered_set>
#include <cmath>
#include <cassert>

#include "Conditioning.h"
#include "FactorGraphObservation.h"
#include "ExpectationMaximization.h"

namespace Grante {

ExpectationMaximization::ExpectationMaximization(FactorGraphModel* model,
	ParameterEstimationMethod* parest_method)
	: fg_model(model), parest_method(parest_method) {
}

ExpectationMaximization::~ExpectationMaximization() {
	delete (parest_method);
}

void ExpectationMaximization::AddPrior(const std::string& factor_type,
	Prior* prior) {
	parest_method->AddPrior(factor_type, prior);
	priors.insert(std::multimap<std::string, Prior*>::value_type(
		factor_type, prior));
}

void ExpectationMaximization::SetupTrainingData(
	const std::vector<partially_labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*>& hidden_inference_methods,
	const std::vector<InferenceMethod*>& observed_inference_methods) {
	// Simpler use-case: observed inference method is equal to parest
	// inference method
	SetupTrainingData(training_data, hidden_inference_methods,
		observed_inference_methods, observed_inference_methods);
}

void ExpectationMaximization::SetupTrainingData(
	const std::vector<partially_labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*>& hidden_inference_methods,
	const std::vector<InferenceMethod*>& observed_inference_methods,
	const std::vector<InferenceMethod*>& parest_inference_methods) {
	this->training_data = training_data;
	this->hidden_inference_methods = hidden_inference_methods;
	this->observed_inference_methods = observed_inference_methods;
	this->parest_inference_methods = parest_inference_methods;
	fg_m_training_data.clear();

	// Decompose each training instance into E- and M-subgraphs necessary for
	// E/M steps.
	size_t sample_count = training_data.size();
	fi_to_efi.clear();
	fi_to_efi.resize(sample_count);
	hidden_efi.clear();
	hidden_efi.resize(sample_count);
	for (size_t si = 0; si < sample_count; ++si) {
		// TODO: right now only discrete observations are supported
		assert(training_data[si].second->Type() ==
			FactorGraphObservation::DiscreteLabelingType);

		// 1. Partition factors
		std::vector<unsigned int> obsfactors;
		std::vector<unsigned int> crossfactors;
		std::vector<unsigned int> hiddenfactors;
		PartitionFactors(training_data[si], obsfactors, crossfactors,
			hiddenfactors);

		// 2. Build E-subgraph: the conditioned subgraph that contains only
		// the unobserved variables.  It is composed of the hiddenfactors and
		// the conditioned crossfactors (conditioned by ground truth
		// observations).
		// We will not need to update the conditioning information in fg_e but
		// only the expectations of all cross factors.
		std::vector<unsigned int> fg_e_var_new_to_orig;
		std::vector<unsigned int> fg_e_fac_new_to_orig;
		FactorGraph* fg_e = Conditioning::ConditionFactorGraph(&ftab,
			training_data[si].first, training_data[si].second,
			fg_e_var_new_to_orig, fg_e_fac_new_to_orig);
		E_fg.push_back(fg_e);
		// Instantiate inference method on hidden-induced subgraph
		E_inf.push_back(hidden_inference_methods[si]->Produce(fg_e));

		// 3. Build M-graph: the full factor graph.
		const std::vector<Factor*>& factors =
			training_data[si].first->Factors();
		// Setup expectations
		std::vector<std::vector<double> > obs_expect(factors.size());
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			obs_expect[fi].resize(factors[fi]->Type()->ProdCardinalities());

			// Initialize with uniform expectations for the fully-observed
			// training method.  The obs_expect[fi] will be updated throughout
			// the EM iterations.
			std::fill(obs_expect[fi].begin(), obs_expect[fi].end(),
				1.0 / static_cast<double>(obs_expect[fi].size()));
		}
		// Fix expectations of all factors containing only observed variables
		std::vector<unsigned int> full_state;
		ComputeFullState(training_data[si], full_state);
		for (std::vector<unsigned int>::const_iterator
			ofi = obsfactors.begin(); ofi != obsfactors.end(); ++ofi) {
			unsigned int fi = *ofi;
			std::fill(obs_expect[fi].begin(), obs_expect[fi].end(), 0.0);
			obs_expect[fi][factors[fi]->ComputeAbsoluteIndex(full_state)]= 1.0;
		}

		// Add training instance
		FactorGraphObservation* fg_m_obs =
			new FactorGraphObservation(obs_expect);
		fg_m_training_data.push_back(
			std::pair<FactorGraph*, const FactorGraphObservation*>(
				training_data[si].first, fg_m_obs));

		// Build mapping between factor indices and factor index in E-graph.
		// This is needed for efficiently updating expectations in the M-graph
		// during EM learning.
		for (unsigned int efi = 0; efi < fg_e_fac_new_to_orig.size(); ++efi) {
			// Cross- and hidden factors
			fi_to_efi[si].insert(std::pair<unsigned int, unsigned int>(
				fg_e_fac_new_to_orig[efi], efi));
		}
		hidden_efi[si].insert(hiddenfactors.begin(), hiddenfactors.end());
	}

	// Setup M-step parameter estimation method
	parest_method->SetupTrainingData(fg_m_training_data,
		parest_inference_methods);
}

void ExpectationMaximization::Train(double conv_tol, unsigned int max_iter,
	double parest_conv_tol, unsigned int parest_max_iter) {
	unsigned int em_iter = 1;
	double em_conv_measure = std::numeric_limits<double>::infinity();
	double em_obj = std::numeric_limits<double>::infinity();
	double em_obj_prev = std::numeric_limits<double>::infinity();
	double nll_t = std::numeric_limits<double>::infinity();
	for ( ; em_iter <= max_iter && em_conv_measure >= conv_tol; ++em_iter) {
		std::cout << std::endl;
		std::cout << "EM iter " << em_iter << ", obj " << em_obj
			<< ", tol " << em_conv_measure << std::endl;

		// 1. Compute hidden variable expectations
		double Hy = ComputeHiddenVariableExpectations();

		// 2. Update conditioning in fully observed subgraph by hidden
		// expectations
		double expected_M_energy = UpdateMExpectationTargets();

		// EM objective being minimized is (x_n observed, h hidden)
		// nll_n(y_n,x_n,w)
		//    = - log[\sum_{h \in H_n} exp(-E(y_n,h;x_n,w)-log Z_n(x_n,w))]
		//                                   // (incomplete log-likelihood)
		//    = - log[\sum_{h \in H_n} (p(h;y_n,x_n,w)/p(h;y_n,x_n,w))
		//            exp(-E(y_n,h;x_n,w)) ] + log Z_n(x_n,w)
		//   <= - \sum_{h \in H_n}[ p(h;y_n,x_n,w) log[
		//          (1/p(h;y_n,x_n,w)) exp(-E(y_n,h;x_n,w))]] + log Z_n(x_n,w)
		//    = - \sum_{h \in H_n}[ -p(h;y_n,x_n,w) log p(h;y_n,x_n,w)
		//            + p(h;y_n,x_n,w) (-E(y_n,h;x_n,w))] + log Z_n(x_n,w)
		//    = - H(p(h;y_n,x_n,w))       // (entropy of distribution on h)
		//      + \expects_{h~p(h;y_n,x_n,w)}[E(y_n,h;w)]  // (expected energy)
		//      + log Z_n(x_n,w)		  // (joint partition function)
		// with
		//   log Z_n(x_n,w) =
		//       log \sum_{y_n \in Y_n} \sum_{h \in H_n} exp(-E(y_n,h;x_n,w)).
		//
		// The full objective minimized with prior is the negative
		// log-likelihood bound:
		//    nll(X,w) = 1/N \sum_{n=1}^{N} nll_n(y_n,x_n,w) - 1/N log p(w).
		em_obj_prev = em_obj;
		double logZ = ComputeLogZ();
		double logpw = ComputeLogP();
		double Hmu = Hy - expected_M_energy;	// entropy H(\mu)
		double Hmu_real = ComputeHiddenVariableEntropies();
		assert(Hmu >= 0.0);	// non-negativity of the entropy
		std::cout << "OBJ, Hmu " << Hmu << ", Hmu_real " << Hmu_real
			<< ", expected_M_energy " << expected_M_energy
			<< ", logZ " << logZ << std::endl;
		double tobj = -Hmu + expected_M_energy + logZ - logpw;
		double tobj2 = -Hmu_real + expected_M_energy + logZ - logpw;
		std::cout << "OBJ, tobj " << tobj << ", tobj2 " << tobj2 << std::endl;
//		em_obj = (1.0/static_cast<double>(training_data.size()))*(-Hy+logZ-logpw);
#if 0
		em_obj = (1.0/static_cast<double>(training_data.size()))*
			(-Hy + 2.0*expected_M_energy + logZ - logpw);
		em_obj = (1.0/static_cast<double>(training_data.size()))*tobj;
#endif
		em_obj = (1.0/static_cast<double>(training_data.size()))*tobj2;
		std::cout << "   EM obj: (1/N)*(Hy=" << Hy << ", logZ=" << logZ
			<< ", logpw=" << logpw << ") = " << em_obj << std::endl;

		// Check for a decrease in the EM objective
		if (em_iter > 1) {
			if (em_obj - em_obj_prev > 1.0e-8) {
				std::cout << "Warning: EM objective increased, this "
					<< "should not happen!" << std::endl;
				std::cout << "(This can happen if you use approximate "
					<< "inference for the E/M-steps.)" << std::endl;

				std::cout << "em_obj: " << em_obj << std::endl;
				std::cout << "em_obj_prev: " << em_obj_prev << std::endl;
				std::cout << "em_obj_prev-em_obj: " << (em_obj_prev-em_obj) << std::endl;
			}
//			assert(em_obj - em_obj_prev <= 1.0e-8);
		}
		parest_method->UpdateTrainingLabeling(fg_m_training_data);

		// 3. Estimate parameters
		double cur_obj = EstimateParameters(parest_conv_tol, parest_max_iter);
		nll_t = cur_obj;

		em_conv_measure = em_obj_prev - em_obj;
		if (em_iter == 1)
			em_conv_measure = std::numeric_limits<double>::infinity();
		// FIXME: this is an adhoc-fix for approximate inference problems, but
		// we should replace the EM objective with a surrogate objective
		// evaluated on the likelihood approximation we use
		if (em_conv_measure <= -1.0e-3)
			em_conv_measure = std::numeric_limits<double>::infinity();
	}
	std::cout << "EM converged after " << em_iter << " iterations, "
		<< "obj " << em_obj << ", tol " << em_conv_measure << "." << std::endl;
}

double ExpectationMaximization::ComputeHiddenVariableExpectations() {
	double Hy = 0.0;
	assert(E_fg.size() == E_inf.size());
	int E_fg_size = static_cast<int>(E_fg.size());
	#pragma omp parallel for schedule(dynamic)
	for (int si = 0; si < E_fg_size; ++si) {
		// Parameters in the model have changed, update energies
		E_fg[si]->ForwardMap();
		// Compute marginals (expectations)
		E_inf[si]->PerformInference();
		#pragma omp critical
		{
			// TODO: do we need to perform a correction here?  That is, when
			// we condition the factor graph we throw away all the fully
			// observed factors which incur a constant energy, but the
			// log-sum-exp equality says we should add it here, right?
			// Yes, this is the case!
			// TODO: in the conditioning functions, return the constant, then
			// add it here, (see LogSumExp as to how)
			// The problem when not doing this is that the EM objective, which
			// is a lower bound on the marginal likelihood can have the wrong
			// sign.

			// (3.45) in Wainwright:
			// log Z = sup_\mu [<\theta,\mu> + H(\mu)]
			//       = sup_\mu [-E(\mu) + H(\mu)]
			Hy += E_inf[si]->LogPartitionFunction();
		}
	}
	return (Hy);
}

double ExpectationMaximization::ComputeHiddenVariableEntropies() {
	double Hmu = 0.0;
	assert(E_fg.size() == E_inf.size());
	int E_fg_size = static_cast<int>(E_fg.size());
	for (int si = 0; si < E_fg_size; ++si) {
		// Parameters in the model have changed, update energies
		E_fg[si]->ForwardMap();
		// Compute marginals (expectations)
		E_inf[si]->PerformInference();
		Hmu += E_inf[si]->Entropy();
	}
	return (Hmu);
}

double ExpectationMaximization::UpdateMExpectationTargets() {
	double expected_M_energy = 0.0;

	// For each observed factor graph
	for (unsigned int n = 0; n < fg_m_training_data.size(); ++n) {
		// Update expectations of all cross- and hidden-factors
		std::vector<std::vector<double> >& fg_m_expects =
			const_cast<FactorGraphObservation*>(fg_m_training_data[n].second)->Expectation();
		const std::vector<Factor*>& e_factors = E_fg[n]->Factors();
		for (fi_to_efi_t::const_iterator fi_i = fi_to_efi[n].begin();
			fi_i != fi_to_efi[n].end(); ++fi_i) {
			unsigned int fi = fi_i->first;
			unsigned int efi = fi_i->second;
			const std::vector<double>& ef_marg = E_inf[n]->Marginal(efi);

			// Two cases:
			// 1. Hidden factor: copy the entire marginal
			if (hidden_efi[n].count(fi) > 0) {
				// Copy
				assert(fg_m_expects[fi].size() == ef_marg.size());
				std::copy(ef_marg.begin(), ef_marg.end(),
					fg_m_expects[fi].begin());
				continue;
			}

			// 2. Cross factor: extend the conditional marginals to the full
			// marginals using the observations.
			assert(fg_m_expects[fi].size() > ef_marg.size());
			assert(fg_m_expects[fi].size() ==
				fg_m_training_data[n].first->Factors()[fi]->Type()->ProdCardinalities());
			ftab.ExtendMarginals(e_factors[efi], ef_marg,
				fg_m_expects[fi], false);
		}
		expected_M_energy +=
			fg_m_training_data[n].first->EvaluateEnergy(fg_m_expects);
	}

	return (expected_M_energy);
}

double ExpectationMaximization::EstimateParameters(double conv_tol,
	unsigned int max_iter) {
	double obj = parest_method->Train(conv_tol, max_iter);

	return (obj);
}

// Compute full model logZ
double ExpectationMaximization::ComputeLogZ() {
	double logZ = 0.0;
	int training_data_size = static_cast<int>(training_data.size());
	#pragma omp parallel for schedule(dynamic)
	for (int si = 0; si < training_data_size; ++si) {
		training_data[si].first->ForwardMap();
		observed_inference_methods[si]->PerformInference();
		#pragma omp critical
		{
			logZ += observed_inference_methods[si]->LogPartitionFunction();
		}
	}
	return (logZ);
}

double ExpectationMaximization::ComputeLogP() {
	double logp = 0.0;
	std::vector<double> dummy;
	for (std::multimap<std::string, Prior*>::const_iterator
		prior = priors.begin(); prior != priors.end(); ++prior)
	{
		FactorType* ft = fg_model->FindFactorType(prior->first);
		logp -= prior->second->EvaluateNegLogP(ft->Weights(), dummy);
	}
	return (logp);
}

void ExpectationMaximization::PartitionFactors(
	const partially_labeled_instance_type& instance,
	std::vector<unsigned int>& fg_obsfactors,
	std::vector<unsigned int>& fg_crossfactors,
	std::vector<unsigned int>& fg_hiddenfactors) const {
	fg_obsfactors.clear();
	fg_crossfactors.clear();
	fg_hiddenfactors.clear();

	// Obtain subset of observed variables and factors
	const std::vector<unsigned int>& var_subset_v =
		instance.second->ObservedVariableSet();
	for (unsigned int vgi = 0; vgi < var_subset_v.size(); ++vgi) {
		assert(var_subset_v[vgi] < instance.first->Cardinalities().size());
	}

	std::tr1::unordered_set<unsigned int> var_subset(var_subset_v.begin(),
		var_subset_v.end());
	const std::vector<Factor*>& factors = instance.first->Factors();

	// For each factor
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// Obtain variables the factor operates on
		const std::vector<unsigned int>& fac_vars =
			factors[fi]->Variables();

		bool has_observed = false;
		bool has_unobserved = false;
		for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
			if (var_subset.count(fac_vars[fvi]) > 0) {
				has_observed = true;
			} else {
				has_unobserved = true;
			}
		}
		assert(has_observed || has_unobserved);
		if (has_observed && has_unobserved) {
			// Cross-factor
			fg_crossfactors.push_back(fi);
		} else if (has_observed == false && has_unobserved) {
			// Factor is between hiddens
			fg_hiddenfactors.push_back(fi);
		} else {
			// Factor between observed variables
			assert(has_observed && has_unobserved == false);
			fg_obsfactors.push_back(fi);
		}
	}
}

void ExpectationMaximization::ComputeFullState(
	const partially_labeled_instance_type& instance,
	std::vector<unsigned int>& full_state) const {
	// Resize and fill full state vector
	size_t var_count = instance.first->Cardinalities().size();
	full_state.resize(var_count);
	std::fill(full_state.begin(), full_state.end(),
		std::numeric_limits<unsigned int>::max());

	// Copy observed variables
	assert(instance.second->Type() ==
		FactorGraphObservation::DiscreteLabelingType);
	const std::vector<unsigned int>& obs_varset =
		instance.second->ObservedVariableSet();
	const std::vector<unsigned int>& obs_varstate =
		instance.second->ObservedVariableState();
	for (unsigned int oi = 0; oi < obs_varset.size(); ++oi) {
		assert(obs_varset[oi] < full_state.size());
		full_state[obs_varset[oi]] = obs_varstate[oi];
	}
}

}

