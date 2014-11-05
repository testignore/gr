
#include <iostream>
#include <fstream>
#include <limits>
#include <cassert>

#include "GibbsSampler.h"
#include "NaiveMeanFieldInference.h"
#include "SwendsenWangInference.h"

namespace Grante {

SwendsenWangInference::SwendsenWangInference(const FactorGraph* fg,
	const std::vector<std::vector<double> >& marg)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::signaling_NaN()),
		verbose(false), sw(0), burnin_sweeps(50), spacing_sweeps(0),
		sample_count(100), use_single_swsteps(false) {
	// Marginal same-state probabilities
	size_t factor_count = fg->Factors().size();
	const std::vector<Factor*>& factors = fg->Factors();
	std::vector<double> qf(factor_count, 0.0);
	assert(marg.size() == factor_count);

	// Obtain realizable approximate marginals using mean field
	for (size_t fi = 0; fi < factor_count; ++fi) {
		const Factor* fac = factors[fi];
		const std::vector<unsigned int>& fvars = fac->Variables();
		if (fvars.size() <= 1)
			continue;
		assert(fvars.size() == 2);

		unsigned int state_count = fac->Cardinalities()[0];
		const std::vector<double>& margc = marg[fi];
		double p_equal = 0.0;
		for (unsigned int vsi = 0; vsi < state_count; ++vsi)
			p_equal += margc[vsi*state_count + vsi];

		qf[fi] = p_equal;
	}

	// Setup SW by computing factor appearance probabilities
#if 0
	std::vector<double> qf(fg->Factors().size(), 0.0);
	SwendsenWangSampler::ComputeFactorProb(fg, qf, logistic_temp);
#endif

	EdgeAppearanceFromCoclusterProb(qf);
}

void SwendsenWangInference::EdgeAppearanceFromCoclusterProb(
	const std::vector<double>& qf) {
	edgeprob_out.resize(qf.size(), 0.0);
	std::vector<double> qf_actual_cc;
	SwendsenWangSampler::AdjustFactorProbStochastic(fg,
		qf, edgeprob_out, qf_actual_cc, 2000);

#if 0
	// TODO: write to file to get numbers for Matlab
	std::ofstream ofs("sw_weights.txt");
	for (unsigned int fi = 0; fi < edgeprob_out.size(); ++fi) {
		ofs << "qfe fi " << fi << " " << edgeprob_out[fi]
			<< ", actual cc " << qf_actual_cc[fi]
			<< ", (desired cc " << qf[fi] << ")"
			<< std::endl;
	}
	ofs.close();
#endif
	sw = new SwendsenWangSampler(fg, edgeprob_out);
}

SwendsenWangInference::~SwendsenWangInference() {
	if (sw != 0)
		delete sw;
}

// Directly use user-provided factor appearance probabilities
SwendsenWangInference::SwendsenWangInference(const FactorGraph* fg,
	const std::vector<double>& qf, bool cocluster_prob)
	: InferenceMethod(fg), log_z(std::numeric_limits<double>::signaling_NaN()),
		verbose(false), sw(0), burnin_sweeps(50), spacing_sweeps(0),
		sample_count(100), use_single_swsteps(false) {
	assert(qf.size() == fg->Factors().size());

	if (cocluster_prob) {
		EdgeAppearanceFromCoclusterProb(qf);
	} else {
		// Direct edge appearance probabilities
		sw = new SwendsenWangSampler(fg, qf);
	}
}

InferenceMethod* SwendsenWangInference::Produce(const FactorGraph* fg) const {
	assert(fg->Factors().size() == this->fg->Factors().size());

	SwendsenWangInference* sw_new =
		new SwendsenWangInference(fg, edgeprob_out, false);
	sw_new->SetSamplingParameters(verbose, burnin_sweeps, spacing_sweeps,
		sample_count);

	return (sw_new);
}

void SwendsenWangInference::SetSamplingParameters(bool verbose,
	unsigned int burnin_sweeps, unsigned int spacing_sweeps,
	unsigned int sample_count, bool use_single_swsteps) {
	this->verbose = verbose;
	this->burnin_sweeps = burnin_sweeps;
	this->spacing_sweeps = spacing_sweeps;
	assert(sample_count > 0);
	this->sample_count = sample_count;
	this->use_single_swsteps = use_single_swsteps;
}

void SwendsenWangInference::PerformInference() {
	// 1. Setup marginals
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 0.0);
	}

	PerformBurninPhase();

	// 2. Produce approximate samples
	double sample_contribution = 1.0 / static_cast<double>(sample_count);
	double mean_part_size = 0.0;
	for (unsigned int si = 0; si < sample_count; ++si) {
		if (use_single_swsteps) {
			double step_mps = 0.0;
			for (unsigned int swi = 0; swi <= spacing_sweeps; ++swi)
				step_mps += sw->SingleStep();
			mean_part_size += step_mps / static_cast<double>(1+spacing_sweeps);
		} else {
			mean_part_size += sw->Sweep(1 + spacing_sweeps);
		}
		const std::vector<unsigned int>& sample = sw->State();

		// Add to marginals
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			marginals[fi][factors[fi]->ComputeAbsoluteIndex(sample)] +=
				sample_contribution;
		}
	}
	mean_part_size /= static_cast<double>(sample_count);
	if (verbose) {
		std::cout << "Swendsen-Wang average cluster size was "
			<< mean_part_size << "." << std::endl;
	}
}

void SwendsenWangInference::PerformBurninPhase() {
	// Annealed burnin, just as in the Gibbs inference class, but this time
	// with SW sweeps.
	GibbsSampler gibbs(fg);
	gibbs.SetStateUniformRandom();	// independent random initialization
	sw->SetState(gibbs.State());

	if (burnin_sweeps > 0) {
		unsigned int burnin_anneal = burnin_sweeps / 2;
		if (burnin_anneal > 1) {
			// The first non-zero beta,
			// 0 = beta_0 < beta_1 < beta_2 < ... < beta_K
			// beta_k = gamma^(k-1)*beta_1
			double beta_1 = 1.0e-4;
			double gamma = std::pow(1.0 / beta_1,
				1.0 / static_cast<double>(burnin_anneal-1));

			// Compute a single annealed sample
			double cur_beta = 0.0;
			for (unsigned int k = 0; k < burnin_anneal; ++k) {
				sw->SetInverseTemperature(cur_beta);
				if (use_single_swsteps) {
					sw->SingleStep();
				} else {
					sw->Sweep(1);
				}

				cur_beta = std::pow(gamma, static_cast<double>(k)) * beta_1;
			}
			sw->SetInverseTemperature(1.0);
			sw->Sweep(1);
		}
		// Normal SW sweeps (temperature 1)
		if (use_single_swsteps) {
			for (unsigned int swi = 0; swi < (burnin_sweeps - burnin_anneal);
				++swi) {
				sw->SingleStep();
			}
		} else {
			sw->Sweep(burnin_sweeps - burnin_anneal);
		}
	}
}

void SwendsenWangInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& SwendsenWangInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >&
SwendsenWangInference::Marginals() const {
	return (marginals);
}

double SwendsenWangInference::LogPartitionFunction() const {
	return (std::numeric_limits<double>::signaling_NaN());
}

// Produce approximate samples from the distribution
void SwendsenWangInference::Sample(
	std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	states.resize(sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		if (use_single_swsteps) {
			for (unsigned int swi = 0; swi <= spacing_sweeps; ++swi)
				sw->SingleStep();
		} else {
			sw->Sweep(1 + spacing_sweeps);
		}
		states[si] = sw->State();
	}
}

double SwendsenWangInference::MinimizeEnergy(
	std::vector<unsigned int>& state) {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

SwendsenWangSampler* SwendsenWangInference::Sampler(void) {
	return (sw);
}

const std::vector<double>&
SwendsenWangInference::EdgeAppearanceProbabilities(void) const {
	return (edgeprob_out);
}

}

