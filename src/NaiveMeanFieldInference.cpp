
#include <limits>
#include <cmath>
#include <cassert>

#include "LogSumExp.h"
#include "NaiveMeanFieldInference.h"

namespace Grante {

NaiveMeanFieldInference::NaiveMeanFieldInference(const FactorGraph* fg)
	: InferenceMethod(fg), fgu(fg),
	log_z(std::numeric_limits<double>::signaling_NaN()),
	verbose(true), conv_tol(1.0e-6), max_iter(50) {
}

NaiveMeanFieldInference::~NaiveMeanFieldInference() {
}

InferenceMethod* NaiveMeanFieldInference::Produce(const FactorGraph* fg) const
{
	NaiveMeanFieldInference* nmf = new NaiveMeanFieldInference(fg);
	nmf->SetParameters(verbose, conv_tol, max_iter);

	return (nmf);
}

void NaiveMeanFieldInference::SetParameters(bool verbose, double conv_tol,
	unsigned int max_iter) {
	assert(conv_tol > 0.0);
	this->verbose = verbose;
	this->conv_tol = conv_tol;
	this->max_iter = max_iter;
}

void NaiveMeanFieldInference::PerformInference() {
	// Setup variable distributions (we will construct the full factor
	// marginals after everything else)
	const std::vector<unsigned int>& card = fg->Cardinalities();
	std::vector<std::vector<double> > vmarg(card.size());
	for (unsigned int vi = 0; vi < card.size(); ++vi) {
		vmarg[vi].resize(card[vi]);
		std::fill(vmarg[vi].begin(), vmarg[vi].end(),
			1.0 / static_cast<double>(card[vi]));
	}

	// Iterate naive mean field on variables
	double conv_measure = std::numeric_limits<double>::infinity();
	log_z = -std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; (max_iter == 0 || iter <= max_iter) &&
		conv_measure >= conv_tol; ++iter)
	{
		// Update all site distributions
		for (unsigned int vi = 0; vi < card.size(); ++vi)
			UpdateSite(vmarg, vi);

		double log_z_prev = log_z;
		log_z = ComputeLogPartitionFunction(vmarg);
		assert(log_z >= log_z_prev);	// Monotonic ascent
		conv_measure = log_z - log_z_prev;
	}

	// Produce final approximate marginals
	ProduceMarginals(vmarg);
}

// Update a site distribution analytically, (3.39) in [Nowozin2011].
double NaiveMeanFieldInference::UpdateSite(
	std::vector<std::vector<double> >& vmarg, unsigned int vi) const {
	std::vector<double> E_vi(vmarg[vi].size(), 1.0);

	// Walk all adjacent factors
	const std::set<unsigned int>& adj = fgu.AdjacentFactors(vi);
	const std::vector<Factor*>& factors = fg->Factors();
	// F in \mathcal{F}
	for (std::set<unsigned int>::const_iterator afi = adj.begin();
		afi != adj.end(); ++afi) {
		// Get information required from this factor
		const Factor* fac = factors[*afi];
		const FactorType* ftype = fac->Type();
		const std::vector<unsigned int>& fvars = fac->Variables();
		const std::vector<double>& E = fac->Energies();

		// y_F in \mathcal{Y}_F
		for (unsigned int ei = 0; ei < E.size(); ++ei) {
			double P_vi = 1.0;	// \prod_{j in N(F) \ {i}} q_j(y_j)
			unsigned int vi_state = std::numeric_limits<unsigned int>::max();
			for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
				// j in N(F) \ {i}
				if (fvars[fvi] == vi) {
					vi_state = ftype->LinearIndexToVariableState(ei, fvi);
					continue;
				}

				unsigned int fvi_state =
					ftype->LinearIndexToVariableState(ei, fvi);
				P_vi *= vmarg[fvars[fvi]][fvi_state];	// q_j(y_j)
			}
			E_vi[vi_state] -= P_vi * E[ei];
		}
	}
	double lambda = -LogSumExp::Compute(E_vi);	// (3.40)

	// Update distribution
	double max_diff = -std::numeric_limits<double>::infinity();
	for (unsigned int vsi = 0; vsi < E_vi.size(); ++vsi) {
		double new_val = std::exp(E_vi[vsi] + lambda);
		max_diff = std::max(std::fabs(vmarg[vi][vsi] - new_val), max_diff);
		vmarg[vi][vsi] = new_val;
	}

	return (max_diff);
}

// Compute logZ bound from Gibbs inequality (3.30)
double NaiveMeanFieldInference::ComputeLogPartitionFunction(
	const std::vector<std::vector<double> >& vmarg) const {
	// Compute the entropy, (3.32)
	double lz = 0.0;
	for (unsigned int vi = 0; vi < vmarg.size(); ++vi) {
		for (unsigned int vsi = 0; vsi < vmarg[vi].size(); ++vsi) {
			lz += vmarg[vi][vsi] * std::log(vmarg[vi][vsi]);
		}
	}

	// Add the average energy as by the meanfield approximation to the factor
	// marginals
	const std::vector<Factor*>& factors = fg->Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const Factor* fac = factors[fi];
		const FactorType* ftype = fac->Type();
		const std::vector<unsigned int>& fvars = fac->Variables();
		const std::vector<double>& E = fac->Energies();

		for (unsigned int ei = 0; ei < E.size(); ++ei) {
			double P_vi = 1.0;	// q_F := \prod_{j in N(F)} q_j(y_j)
			for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi)
				P_vi *= vmarg[fvars[fvi]][ftype->LinearIndexToVariableState(ei, fvi)];

			lz += P_vi * E[ei];
		}
	}
	return (-lz);
}

void NaiveMeanFieldInference::ProduceMarginals(
	const std::vector<std::vector<double> >& vmarg) {
	// Produce factor marginals by naive mean field approximation
	const std::vector<Factor*>& factors = fg->Factors();
	marginals.resize(factors.size());
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		marginals[fi].resize(factors[fi]->Type()->ProdCardinalities());
		std::fill(marginals[fi].begin(), marginals[fi].end(), 1.0);

		const Factor* fac = factors[fi];
		const FactorType* ftype = fac->Type();
		const std::vector<unsigned int>& fvars = fac->Variables();
		for (unsigned int ei = 0; ei < marginals[fi].size(); ++ei) {
			for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
				marginals[fi][ei] *= vmarg[fvars[fvi]][
					ftype->LinearIndexToVariableState(ei, fvi)];
			}
		}
	}
}

void NaiveMeanFieldInference::ClearInferenceResult() {
	marginals.clear();
}

const std::vector<double>& NaiveMeanFieldInference::Marginal(
	unsigned int factor_id) const {
	assert(factor_id < marginals.size());
	return (marginals[factor_id]);
}

const std::vector<std::vector<double> >&
NaiveMeanFieldInference::Marginals() const {
	return (marginals);
}

double NaiveMeanFieldInference::LogPartitionFunction() const {
	return (log_z);
}

// NOT IMPLEMENTED
void NaiveMeanFieldInference::Sample(std::vector<std::vector<unsigned int> >& states,
	unsigned int sample_count) {
	assert(0);
}

// NOT IMPLEMENTED
double NaiveMeanFieldInference::MinimizeEnergy(std::vector<unsigned int>& state) {
	assert(0);
	return (std::numeric_limits<double>::signaling_NaN());
}

}

