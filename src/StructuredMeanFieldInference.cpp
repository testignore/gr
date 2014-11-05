
#include <algorithm>
#include <numeric>
#include <limits>
#include <tr1/unordered_set>

#include <boost/lambda/lambda.hpp>

#include "Conditioning.h"
#include "LogSumExp.h"
#include "FactorGraphPartialObservation.h"
#include "VAcyclicDecomposition.h"
#include "FactorGraphStructurizer.h"
#include "StructuredMeanFieldInference.h"

using namespace boost::lambda;

namespace Grante {

StructuredMeanFieldInference::StructuredMeanFieldInference(
	const FactorGraph* fg, FactorConditioningTable* fcond_tab,
	DecompositionType decomp_type)
	: InferenceMethod(fg), fcond_tab(fcond_tab),
		log_z(std::numeric_limits<double>::signaling_NaN()),
		verbose(true), conv_tol(1.0e-6), max_iter(50) {
	// 1. Compute a v-acyclic decomposition of the factor graph
	VAcyclicDecomposition vac(fg);
	std::vector<double> factor_weight(fg->Factors().size(), 1.0);

	// Total correlation based weights
	if (decomp_type == TotalCorrelationWeights) {
		const std::vector<Factor*>& factors = fg->Factors();
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			double tcorr = factors[fi]->TotalCorrelation();
			assert(tcorr >= -1.0e-10);
			// Add an epsilon so it is always better to add factors than to
			// leave them out.
			factor_weight[fi] = std::min(1.0e-3, tcorr);
		}
	}
	std::vector<bool> factor_is_removed;
	vac.ComputeDecompositionSP(factor_weight, factor_is_removed);

	InitializeVAC(factor_is_removed);
}

StructuredMeanFieldInference::StructuredMeanFieldInference(
	const FactorGraph* fg, FactorConditioningTable* fcond_tab,
	const std::vector<bool>& factor_is_removed)
	: InferenceMethod(fg), fcond_tab(fcond_tab),
		log_z(std::numeric_limits<double>::signaling_NaN()),
		verbose(true), conv_tol(1.0e-6), max_iter(50) {
	// TODO: assert this yields a v-acyclic decomposition
	assert(factor_is_removed.size() == fg->Factors().size());
	InitializeVAC(factor_is_removed);
}

StructuredMeanFieldInference::~StructuredMeanFieldInference() {
	// Delete component factor graphs (created in InitializeVAC)
	assert(mf_comp.size() == mf_comp_inf.size());
	for (unsigned int ci = 0; ci < mf_comp.size(); ++ci) {
		delete (mf_comp[ci]);
		delete (mf_comp_inf[ci]);
	}
}

void StructuredMeanFieldInference::InitializeVAC(
	const std::vector<bool>& factor_is_removed) {
	// Create simple factor to fi lookup table
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi)
		fac_to_fi.insert(std::pair<Factor*, unsigned int>(factors[fi], fi));

	// Decompose factor graph into components
	std::vector<unsigned int> cc_var_label;
	unsigned int cc_count = FactorGraphStructurizer::ConnectedComponents(
		fg, factor_is_removed, cc_var_label);

	// Instantiate a conditioned factor graph for each connected component
	mf_comp.resize(cc_count);
	mf_comp_inf.resize(cc_count);
	for (unsigned int ci = 0; ci < cc_count; ++ci) {
		// Collect all variables in this component
		std::vector<unsigned int> cond_var_set;
		std::tr1::unordered_set<unsigned int> cond_var;
		for (unsigned int vi = 0; vi < cc_var_label.size(); ++vi) {
			// Variable in this component -> do not condition
			if (cc_var_label[vi] == ci)
				continue;

			// Condition on all variables not in this component
			cond_var_set.push_back(vi);
			cond_var.insert(vi);
		}

		// Collect all factors in which both conditioned and unconditioned
		// variables appear and initialize the expectation to uniform.
		std::vector<unsigned int> cond_fac_subset;
		std::vector<std::vector<double> > cond_expect;
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			bool has_cond = false;	// Factor has conditioned-on variables
			bool has_uncond = false;	// Factor has unconditioned variables
			const std::vector<unsigned int>& fac_vars =
				factors[fi]->Variables();
			const std::vector<unsigned int>& fac_card =
				factors[fi]->Cardinalities();
			unsigned int cond_prod_card = 1;
			for (unsigned int fvi = 0; fvi < fac_vars.size(); ++fvi) {
				if (cond_var.count(fac_vars[fvi]) == 0) {
					has_uncond = true;
				} else {
					has_cond = true;
					cond_prod_card *= fac_card[fvi];
				}
			}
			// Fully conditioned and fully unconditioned factors -> skip
			if (has_cond == false || has_uncond == false)
				continue;

			// We have a mixed factor, produce a uniform expectation
			// observation
			std::vector<double> cur_cond_marg(cond_prod_card,
				1.0/static_cast<double>(cond_prod_card));

			// Add this factor as to be conditioned
			cond_fac_subset.push_back(fi);
			cond_expect.push_back(cur_cond_marg);
		}

		// Build a partial observation from all the conditioned factors
		FactorGraphPartialObservation pobs(cond_var_set, cond_fac_subset,
			cond_expect);

		// Condition and add
		std::vector<unsigned int> var_new_to_orig;
		FactorGraph* fg_cond = Conditioning::ConditionFactorGraph(fcond_tab,
			fg, &pobs, var_new_to_orig);

		// Store conditioned factor graph and create an inference object.
		// The conditioned factor graph is always tree structured.
		mf_comp[ci] = fg_cond;
		mf_comp_inf[ci] = new TreeInference(fg_cond);

		// Collect all mean-field factors
		std::tr1::unordered_set<unsigned int>
			cond_fac_subset_hs(cond_fac_subset.begin(), cond_fac_subset.end());
		const std::vector<Factor*>& comp_factors = fg_cond->Factors();
		for (unsigned int comp_fi = 0; comp_fi < comp_factors.size();
			++comp_fi) {
			Factor* comp_fac = comp_factors[comp_fi];
			// Is conditioned?  No -> skip
			if (cond_fac_subset_hs.count(
				fac_to_fi[fcond_tab->OriginalFactor(comp_fac)]) == 0)
				continue;

			meanfield_factor_set.insert(comp_fac);
			cfac_to_mfi[comp_fac] = ci;
			cfac_to_cfi[comp_fac] = comp_fi;
		}
	}

	// Create lookup tables needed during inference
	for (unsigned int mfi = 0; mfi < mf_comp.size(); ++mfi) {
		const std::vector<Factor*>& mfi_factors = mf_comp[mfi]->Factors();
		for (unsigned int comp_fi = 0; comp_fi < mfi_factors.size();
			++comp_fi) {
			Factor* new_factor = mfi_factors[comp_fi];
			bool is_conditioned = meanfield_factor_set.count(new_factor) != 0;
			if (is_conditioned == false)
				continue;

			unsigned int orig_fi =
				fac_to_fi[fcond_tab->OriginalFactor(new_factor)];
			fi_to_condfac[orig_fi].insert(new_factor);
		}
	}
}

InferenceMethod* StructuredMeanFieldInference::Produce(
	const FactorGraph* fg) const {
	StructuredMeanFieldInference* smf =
		new StructuredMeanFieldInference(fg, fcond_tab);
	smf->SetParameters(verbose, conv_tol, max_iter);

	return (smf);
}

void StructuredMeanFieldInference::SetParameters(bool verbose,
	double conv_tol, unsigned int max_iter) {
	this->verbose = verbose;
	assert(conv_tol >= 0.0);
	this->conv_tol = conv_tol;
	this->max_iter = max_iter;
}

void StructuredMeanFieldInference::PerformInference() {
	// 1. Compute initial log partition function
	log_z = 0.0;
	for (unsigned int mfi = 0; mfi < mf_comp.size(); ++mfi) {
		mf_comp[mfi]->ForwardMap();
		mf_comp_inf[mfi]->PerformInference();
	}

	ProduceMarginals();
	log_z = ComputeLogPartitionFunction();

	// 2. Iterate mean field on components
	double conv_measure = std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; (max_iter == 0 || iter <= max_iter) &&
		conv_measure >= conv_tol; ++iter)
	{
		if (verbose) {
			std::cout << "iter " << iter << ", logZ " << log_z
				<< ", conv " << conv_measure << std::endl;
		}

		// For each component,
		for (unsigned int mfi = 0; mfi < mf_comp.size(); ++mfi) {
			// i) update all conditioning expectation information
			UpdateComponentEnergies(mfi);
			mf_comp[mfi]->ForwardMap();

			// ii) Perform inference
			mf_comp_inf[mfi]->PerformInference();
		}
		// Produce all marginals, either as copy or product
		ProduceMarginals();

		// Compute bound to the log-partition function
		double log_z_prev = log_z;
		log_z = ComputeLogPartitionFunction();
		assert(log_z >= (log_z_prev - 1.0e-11));
		conv_measure = std::max(0.0, log_z - log_z_prev);
	}
	if (verbose)
		std::cout << "Converged with tol " << conv_measure << "." << std::endl;

	// 3. Delete component inference results
	for (unsigned int mfi = 0; mfi < mf_comp.size(); ++mfi)
		mf_comp_inf[mfi]->ClearInferenceResult();
}

// This method has the responsibility to update all information related to
// component 'mfi'.  In particular, the factors that are part of mfi but
// conditioned on other component inference results need to be processed.  The
// result of this processing are new expectations applied to the conditioning
// factors, producing new energies.
void StructuredMeanFieldInference::UpdateComponentEnergies(unsigned int mfi) {
	FactorGraph* comp = mf_comp[mfi];
	const std::vector<Factor*>& mfi_factors = comp->Factors();

	// For each factor in mfi
	for (unsigned int comp_fi = 0; comp_fi < mfi_factors.size();
		++comp_fi) {
		Factor* new_factor = mfi_factors[comp_fi];
		bool is_conditioned = meanfield_factor_set.count(new_factor) != 0;
		if (is_conditioned == false)
			continue;	// Skip unconditioned factors

		// It is a conditioned factor
		Factor* orig_factor = fcond_tab->OriginalFactor(new_factor);
		assert(orig_factor != 0);
		assert(fac_to_fi.count(orig_factor) > 0);
		unsigned int orig_fi = fac_to_fi[orig_factor];
		assert(fi_to_condfac.count(orig_fi) > 0);
		// Set of all factors derived by conditioning this original factor
		const std::tr1::unordered_set<Factor*>& cond_facs =
			fi_to_condfac[orig_fi];

		// Schematic
		//
		//       (0) ---[F]--- (1)     // original factor
		//       (0) ---[G]            // conditioned factor 1
		//              [H]--- (1)     // conditioned factor 2
		//
		// new_factor is G, orig_factor is F, cond_facs is {G,H}.
		//
		// We now build new expectations used to update factor G.  We build
		// these from the marginals of cond_facs \ {new_factor}, hence from H.
		std::vector<double> ext_marginals(
			orig_factor->Type()->ProdCardinalities(), 1.0);
		size_t cond_expect_prodcard = 1;
		for (std::tr1::unordered_set<Factor*>::const_iterator
			cfi = cond_facs.begin(); cfi != cond_facs.end(); ++cfi) {
			// Ignore own marginals
			if (*cfi == new_factor)
				continue;

			// Obtain the conditional marginals
			assert(cfac_to_mfi.find(new_factor) != cfac_to_mfi.end());
			assert(cfac_to_cfi.find(new_factor) != cfac_to_cfi.end());
			const std::vector<double>& cfi_marg =
				mf_comp_inf[cfac_to_mfi[*cfi]]->Marginal(cfac_to_cfi[*cfi]);

			// Extend and multiply
			std::vector<double> cfi_ext_marginals(
				orig_factor->Type()->ProdCardinalities(), 0.0);
			fcond_tab->ExtendMarginals(*cfi, cfi_marg,
				cfi_ext_marginals, true);
			std::transform(cfi_ext_marginals.begin(), cfi_ext_marginals.end(),
				ext_marginals.begin(), ext_marginals.begin(), _1 * _2);
			cond_expect_prodcard *= (*cfi)->Type()->ProdCardinalities();
		}

		// Project to conditioned-on expectations
		assert(cond_expect_prodcard ==
			(orig_factor->Type()->ProdCardinalities() /
			new_factor->Type()->ProdCardinalities()));
		std::vector<double> cond_var_expect(cond_expect_prodcard);
		fcond_tab->ProjectExtendedMarginalsCond(new_factor, ext_marginals,
			cond_var_expect);

		// Update conditioned factor with new expectations
		fcond_tab->UpdateConditioningInformation(new_factor, cond_var_expect);
	}
}

void StructuredMeanFieldInference::ProduceMarginals() {
	// Initialize all marginals to 1.0 because unconditioned factors will
	// overwrite the marginals and conditioned factors multiply the values
	// (factorial meanfield assumption).
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	std::tr1::unordered_map<Factor*, unsigned int> fac_to_fi;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 1.0);
		fac_to_fi.insert(std::pair<Factor*, unsigned int>(factors[fi], fi));
	}
	for (unsigned int mfi = 0; mfi < mf_comp.size(); ++mfi) {
		const std::vector<Factor*>& mfi_factors = mf_comp[mfi]->Factors();
		for (unsigned int comp_fi = 0; comp_fi < mfi_factors.size();
			++comp_fi) {
			Factor* new_factor = mfi_factors[comp_fi];
			// Original factor index
			unsigned int orig_fi =
				fac_to_fi[fcond_tab->OriginalFactor(new_factor)];
			// Marginal from the component (possibly conditioned)
			const std::vector<double>& comp_marg =
				mf_comp_inf[mfi]->Marginal(comp_fi);

			bool is_conditioned = meanfield_factor_set.count(new_factor) != 0;
			if (is_conditioned == false) {
				// Unconditioned factor: copy marginals
				assert(comp_marg.size() == marginals[orig_fi].size());
				std::copy(comp_marg.begin(), comp_marg.end(),
					marginals[orig_fi].begin());
				continue;
			}

			// Conditioned factor: extend to full marginals and multiply
			std::vector<double> ext_marginals(marginals[orig_fi].size(), 0.0);
			fcond_tab->ExtendMarginals(new_factor, comp_marg,
				ext_marginals, true);
			std::transform(ext_marginals.begin(), ext_marginals.end(),
				marginals[orig_fi].begin(), marginals[orig_fi].begin(),
				_1 * _2);
		}
	}
}

double StructuredMeanFieldInference::ComputeLogPartitionFunction() {
	const std::vector<Factor*>& factors = fg->Factors();
	double mu_theta = 0.0;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// <\theta,\mu>, where \theta=-E in our notation.
		mu_theta -= std::inner_product(marginals[fi].begin(),
			marginals[fi].end(), factors[fi]->Energies().begin(), 0.0);
	}
	// + \sum_m H_m(\mu)
	double H_sum = 0.0;
	for (unsigned int mfi = 0; mfi < mf_comp.size(); ++mfi)
		H_sum += mf_comp_inf[mfi]->Entropy();
#if 0
	std::cout << "mu_theta = " << mu_theta << std::endl;
	std::cout << "H_sum = " << H_sum << std::endl;
	std::cout << "log Z >= " << (mu_theta + H_sum) << std::endl;
#endif

	return (mu_theta + H_sum);
}

void StructuredMeanFieldInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& StructuredMeanFieldInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >&
StructuredMeanFieldInference::Marginals() const {
	return (marginals);
}

double StructuredMeanFieldInference::LogPartitionFunction() const {
	return (log_z);
}

void StructuredMeanFieldInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	// NOT IMPLEMENTED
	assert(0);
}

double StructuredMeanFieldInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	// NOT IMPLEMENTED
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}

