
#include <algorithm>
#include <iostream>
#include <queue>
#include <cmath>
#include <cassert>

#include "LogSumExp.h"
#include "RandomSource.h"
#include "DisjointSet.h"
#include "SwendsenWangSampler.h"

namespace Grante {

SwendsenWangSampler::SwendsenWangSampler(const FactorGraph* fg,
	const std::vector<double>& qf)
	: fg(fg), fgu(fg), qf(qf), var_active(fg->Cardinalities().size()),
		label_count(0), inv_temperature(1.0),
		rgen(RandomSource::GetGlobalRandomSeed()), randu(rgen, rdestu),
		rgen_var(RandomSource::GetGlobalRandomSeed()),
		rdest_var(0, static_cast<int>(fg->Cardinalities().size()-1)),
		randu_var(rgen_var, rdest_var) {
	// Check dimension
	assert(qf.size() == fg->Factors().size());

	// Check that all variables have the same cardinality
	const std::vector<unsigned int>& card = fg->Cardinalities();
	assert(card.empty() == false);
	label_count = card[0];
	for (unsigned int vi = 1; vi < card.size(); ++vi) {
		assert(card[vi] == label_count);
	}

	// Check that all non-unary factors are pairwise
	const std::vector<FactorType*>& factortypes =
		fg->Model()->FactorTypes();
	for (unsigned int fti = 0; fti < factortypes.size(); ++fti) {
		const std::vector<unsigned int>& fcard =
			factortypes[fti]->Cardinalities();
		assert(fcard.size() == 1 || fcard.size() == 2);
	}

	// Initialize state
	state.resize(fg->Cardinalities().size());
	std::fill(state.begin(), state.end(), 0);
}

size_t SwendsenWangSampler::SampleSite(unsigned int var_index) {
	assert(var_index < state.size());
	// Basic data structures: active set, grow queue
	var_active.clear();
	var_active.insert(var_index);
	std::queue<unsigned int> var_todo;
	var_todo.push(var_index);
	unsigned int grow_label = state[var_index];

	// Grow partition
	const std::vector<Factor*>& factors = fg->Factors();
	while (var_todo.empty() == false) {
		// Get currently hot variable
		unsigned int vi = var_todo.front();
		var_todo.pop();

		// Get adjacent factors
		const std::set<unsigned int>& adj_facs = fgu.AdjacentFactors(vi);
		for (std::set<unsigned int>::const_iterator afi = adj_facs.begin();
			afi != adj_facs.end(); ++afi) {
			const Factor* fac = factors[*afi];
			const std::vector<unsigned int>& fvars = fac->Variables();
			if (fvars.size() <= 1)
				continue;	// nothing to merge
			assert(fvars.size() == 2);	// only pairwise for now

			// A) check all labels agree
			bool all_agree = true;
			for (unsigned int fvi = 0; all_agree && fvi < fvars.size();
				++fvi) {
				if (state[fvars[fvi]] != grow_label)
					all_agree = false;
			}
			if (all_agree == false)
				continue;	// remove factor

			// B) turn off with prob 1-qf
			if (randu() > qf[*afi])
				continue;

			// C) merge all variables, if not already in the same partition
			for (unsigned int fvi = 0; fvi < fvars.size(); ++fvi) {
				if (var_active.count(fvars[fvi]) == 0)
					var_todo.push(fvars[fvi]);	// enqueue

				var_active.insert(fvars[fvi]);
			}
		}
	}

	// Now we have a partition, we need to compute the Swendsen Wang weights
	std::vector<double> log_sw_weights(label_count, 0.0);
	std::tr1::unordered_set<unsigned int> part_factors(fg->Factors().size());
	for (std::tr1::unordered_set<unsigned int>::const_iterator
		pvi = var_active.begin(); pvi != var_active.end(); ++pvi) {
		unsigned int vi = *pvi;
		const std::set<unsigned int>& adj_facs = fgu.AdjacentFactors(vi);
		for (std::set<unsigned int>::const_iterator afi = adj_facs.begin();
			afi != adj_facs.end(); ++afi) {
			part_factors.insert(*afi);	// all factors involved
			const Factor* fac = factors[*afi];
			const std::vector<unsigned int>& fvars = fac->Variables();
			if (fvars.size() <= 1)
				continue;	// unary -> skip

			// It is a pairwise factor.  Check that it crosses the partition
			bool all_in_part = true;
			unsigned int cut_label = 0;
			for (unsigned int fvi = 0; all_in_part && fvi < fvars.size();
				++fvi) {
				if (var_active.count(fvars[fvi]) != 0)
					continue;

				all_in_part = false;
				cut_label = state[fvars[fvi]];
			}
			if (all_in_part)
				continue;	// no SW cut edge

			// We have a SW cut edge, update weight
			log_sw_weights[cut_label] += std::log(1.0 - qf[*afi]);
		}
	}

	// Block-Gibbs energies for active variables with SW correction
	std::vector<double> part_energy(label_count, 0.0);
	for (unsigned int li = 0; li < label_count; ++li) {
		// 0. Compute E_i - log \omega_i
		part_energy[li] = -log_sw_weights[li];

		// 1. Set all variables in partition to label
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			pvi = var_active.begin(); pvi != var_active.end(); ++pvi) {
			state[*pvi] = li;
		}

		// 2. Compute energy of all factors involved
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			pfi = part_factors.begin(); pfi != part_factors.end(); ++pfi) {
			part_energy[li] += factors[*pfi]->EvaluateEnergy(state);
		}

		// 3. Temper distribution
		part_energy[li] *= inv_temperature;
	}
#if 0
	for (unsigned int li = 0; li < label_count; ++li) {
		std::cout << "   state " << li << ", energy " << part_energy[li]
			<< std::endl;
	}
#endif
	// Compute CDF
	double lse = LogSumExp::ComputeNeg(part_energy);
	double cumsum = 0.0;
#if 0
	std::cout << "lse " << lse << std::endl;
#endif
	for (unsigned int li = 0; li < label_count; ++li) {
		part_energy[li] = std::exp(-part_energy[li] - lse);
#if 0
		std::cout << "   state " << li << ", prob " << part_energy[li]
			<< std::endl;
#endif
		cumsum += part_energy[li];
	}
	assert(std::fabs(cumsum - 1.0) <= 1.0e-5);

	// Sample
	double rv = randu() * cumsum;
	double runsum = 0.0;
	for (unsigned int li = 0; li < label_count; ++li) {
		runsum += part_energy[li];
		if (rv > runsum)
			continue;

		// Set state for partition
		for (std::tr1::unordered_set<unsigned int>::const_iterator
			pvi = var_active.begin(); pvi != var_active.end(); ++pvi) {
#if 0
			std::cout << "   setting var " << *pvi << " to " << li
				<< std::endl;
#endif
			state[*pvi] = li;
		}
		break;
	}

	// Cleanup
	size_t part_size = var_active.size();
	var_active.clear();

	return (part_size);
}

size_t SwendsenWangSampler::SingleStep(void) {
	return (SampleSite(randu_var()));
}

double SwendsenWangSampler::Sweep(unsigned int sweep_count) {
	if (sweep_count == 0)
		return (0.0);

	// Order variables randomly
	size_t var_count = fg->Cardinalities().size();
	std::vector<unsigned int> vec(var_count);
	for (size_t vi = 0; vi < var_count; ++vi)
		vec[vi] = static_cast<unsigned int>(vi);

	double part_size_sum = 0.0;
	unsigned int parts_sampled = 0;
	for (unsigned int sweep = 0; sweep < sweep_count; ++sweep) {
		RandomSource::ShuffleRandom(vec);
		size_t total_vars_resampled = 0;
		for (unsigned int cvi = 0; cvi < var_count; ++cvi) {
			unsigned int vi = vec[cvi];
			assert(vi < var_count);
			size_t part_size = SampleSite(vi);
			total_vars_resampled += part_size;

			part_size_sum += part_size;
			parts_sampled += 1;

			// SW sweep: count of resampled variables larger than variables
			if (total_vars_resampled >= var_count) {
				break;
			}
		}
	}
	return (part_size_sum / static_cast<double>(parts_sampled));
}

const std::vector<unsigned int>& SwendsenWangSampler::State() const {
	return (state);
}

void SwendsenWangSampler::SetInverseTemperature(double inv_temperature) {
	this->inv_temperature = inv_temperature;
}

void SwendsenWangSampler::SetState(
	const std::vector<unsigned int>& new_state) {
	assert(new_state.size() == state.size());
	this->state = new_state;
}

double SwendsenWangSampler::ComputeFactorProb(const FactorGraph* fg,
	std::vector<double>& qf_out, double logistic_temp) {
	assert(logistic_temp > 0.0);
	FactorGraphUtility fgu(fg);
	const std::vector<Factor*>& factors = fg->Factors();

	// Initialize probabilities
	qf_out.resize(factors.size());
	std::fill(qf_out.begin(), qf_out.end(), 0.0);

	// Compute a probability for each factor
	unsigned int factors_done = 0;
	double mean_prob = 0.0;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		const Factor* fac = factors[fi];
		const std::vector<unsigned int>& fcard = fac->Cardinalities();
		if (fcard.size() == 1)
			continue;
		assert(fcard.size() == 2);
		assert(fcard[0] == fcard[1]);

#if 1
		// Compute MeanCompat function from [Kim2011], "Variable Grouping for
		// Energy Minimization", CVPR, 2011.
		unsigned int label_count = fcard[0];
		double b_meancompat = 0.0;
		double agree_scale = 1.0 / static_cast<double>(label_count);
		double disagree_scale =
			1.0 / static_cast<double>(label_count*label_count - label_count);

		// Pairwise contribution: E[equal] - E[disagree]
		const std::vector<double>& E = fac->Energies();
		assert(E.size() == label_count*label_count);
		unsigned int agree_index = 0;
		for (unsigned int ei = 0; ei < E.size(); ++ei) {
			if (ei == agree_index) {
				b_meancompat += agree_scale * E[ei];
				agree_index += label_count + 1;
			} else {
				b_meancompat -= disagree_scale * E[ei];
			}
		}
		// Large values of b_meancompat are repulsive interactions,
		// low/negative values are associative interactions.  We want factors
		// that are associative to appear more often in the SW grouping.

		// Squash using a logistic function
		qf_out[fi] = 1.0 / (1.0 + std::exp(b_meancompat / logistic_temp));
#endif
#if 0
		double fac_max_tc = 0.0;
		double fac_tc = fac->TotalCorrelation(fac_max_tc);
		qf_out[fi] = fac_tc / fac_max_tc;
#endif
#if 0
		std::cout << "tc " << fac_tc << ", max " << fac_max_tc << std::endl;
#endif
		assert(qf_out[fi] >= 0.0 && qf_out[fi] <= 1.0);
		mean_prob += qf_out[fi];
		factors_done += 1;
	}
	mean_prob /= static_cast<double>(factors_done);

	return (mean_prob);
}

void SwendsenWangSampler::ComputeNetworkReliability(const FactorGraph* fg,
	std::vector<double>& qf, std::vector<double>& qf_out, unsigned int mc_runs) {
	FactorGraphUtility fgu(fg);
	const std::vector<Factor*>& factors = fg->Factors();

	// Initialize probabilities
	qf_out.resize(factors.size());
	std::fill(qf_out.begin(), qf_out.end(), 0.0);

	// Random number generator
	boost::mt19937 rgen(RandomSource::GetGlobalRandomSeed());
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Monte Carlo runs
	size_t var_count = fg->Cardinalities().size();
	for (unsigned int mc = 0; mc < mc_runs; ++mc) {
		DisjointSet dset(var_count);

		for (size_t fi = 0; fi < factors.size(); ++fi) {
			const Factor* fac = factors[fi];
			const std::vector<unsigned int>& fcard = fac->Cardinalities();
			if (fcard.size() == 1)
				continue;

			assert(fcard.size() == 2);
			assert(fcard[0] == fcard[1]);

			if (randu() > qf[fi])
				continue;	// factor does not appear

			const std::vector<unsigned int>& fvars = fac->Variables();
			assert(fvars.size() == 2);

			unsigned int d0 = dset.FindSet(fvars[0]);
			unsigned int d1 = dset.FindSet(fvars[1]);
			if (d0 != d1)
				dset.Link(d0, d1);	// Merge
		}

		// Collect result runs
		for (size_t fi = 0; fi < factors.size(); ++fi) {
			const Factor* fac = factors[fi];
			const std::vector<unsigned int>& fcard = fac->Cardinalities();
			if (fcard.size() == 1)
				continue;

			const std::vector<unsigned int>& fvars = fac->Variables();
			unsigned int d0 = dset.FindSet(fvars[0]);
			unsigned int d1 = dset.FindSet(fvars[1]);
			if (d0 != d1)
				continue;

			qf_out[fi] += 1.0;	// Variables are linked
		}
	}

	// Monte Carlo estimate
	for (unsigned int fi = 0; fi < factors.size(); ++fi)
		qf_out[fi] /= static_cast<double>(mc_runs);
}

#if 0
// FIXME: obsoleted code, use AdjustFactorProbStochastic instead
double SwendsenWangSampler::AdjustFactorProb(const FactorGraph* fg,
	const std::vector<double>& q_cc, std::vector<double>& qf_out,
	unsigned int correction_iter, unsigned int mc_runs) {
	std::vector<double> qf_actual_cc;
	return (AdjustFactorProb(fg, q_cc, qf_out, qf_actual_cc,
		correction_iter, mc_runs));
}

double SwendsenWangSampler::AdjustFactorProb(const FactorGraph* fg,
	const std::vector<double>& q_cc,
	std::vector<double>& qf_out, std::vector<double>& qf_actual_cc,
	unsigned int correction_iter, unsigned int mc_runs) {
	std::vector<double> qf(q_cc);
	qf_out.resize(q_cc.size());
	std::fill(qf_out.begin(), qf_out.end(), 0.0);

	// Control loop
	for (unsigned int ci = 0; ci < correction_iter; ++ci) {
		ComputeNetworkReliability(fg, qf, qf_out, mc_runs);
		for (unsigned int fi = 0; fi < qf_out.size(); ++fi) {
			qf[fi] *= q_cc[fi] / qf_out[fi];	// (desired) / (actual)
			// FIXME: is this is a bug or instability?
			qf[fi] = std::min(1.0 - 1.0e-3, qf[fi]);
			qf[fi] = std::max(0.0, qf[fi]);
		}
	}

	// Actual estimated network reliabilities
	qf_actual_cc = qf_out;

	// Compute maximum difference
	double max_diff = 0.0;
	for (unsigned int fi = 0; fi < qf_out.size(); ++fi)
		max_diff = std::max(max_diff, std::fabs(qf_out[fi] - q_cc[fi]));

	qf_out = qf;

	return (max_diff);
}
#endif

double SwendsenWangSampler::AdjustFactorProbStochastic(const FactorGraph* fg,
	const std::vector<double>& qf_desired_cc, std::vector<double>& edgeprob_out,
	std::vector<double>& qf_actual_cc, unsigned int max_iter) {
	// Initialize output edge appearance probabilities to be equal to desired
	// co-cluster probabilities.  This is a perfect guess for tree-structured
	// graphs, and reasonable for sparsely connected graphs with relatively
	// uniform degrees.  For cyclic graphs it is typically an overestimate.
	edgeprob_out.resize(qf_desired_cc.size());
	std::copy(qf_desired_cc.begin(), qf_desired_cc.end(), edgeprob_out.begin());

	qf_actual_cc.resize(qf_desired_cc.size());
	std::fill(qf_actual_cc.begin(), qf_actual_cc.end(), 0.0);

	// Random number generator
	boost::mt19937 rgen(RandomSource::GetGlobalRandomSeed());
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	size_t var_count = fg->Cardinalities().size();
	const std::vector<Factor*>& factors = fg->Factors();
	std::vector<bool> qf_chosen(factors.size(), false);

	// Step size control
	double alpha_m = 1000.0;

	// Stochastic gradient descent
	double obj = std::numeric_limits<double>::signaling_NaN();
	unsigned int mc_runs = 2500;
	for (unsigned int iter = 0; iter < max_iter; ++iter) {
		// 1. Obtain Monte Carlo estimates of actual co-cluster probabilities
		if (iter == (max_iter - 1))
			mc_runs = 1000;
		ComputeNetworkReliability(fg, edgeprob_out, qf_actual_cc, mc_runs);

		// 2. Compute objective
		obj = 0.0;
		for (unsigned int fi = 0; fi < qf_actual_cc.size(); ++fi) {
			if (factors[fi]->Cardinalities().size() != 2)
				continue;

			obj += std::pow(qf_desired_cc[fi] - qf_actual_cc[fi], 2.0);
		}
		obj *= 0.5;
#if 1
		std::cout << "[afps] iter " << iter << "  obj " << obj << std::endl;
#endif

		// 3. Compute approximate gradient using Monte Carlo
		std::vector<double> grad_apx(qf_desired_cc.size(), 0.0);
		for (unsigned int mc = 0; mc < mc_runs; ++mc) {
			DisjointSet dset(var_count);
			std::fill(qf_chosen.begin(), qf_chosen.end(), false);
			// Roll-out a subgraph
			for (unsigned int fi = 0; fi < factors.size(); ++fi) {
				const Factor* fac = factors[fi];
				const std::vector<unsigned int>& fcard = fac->Cardinalities();
				if (fcard.size() == 1)
					continue;

				assert(fcard.size() == 2);
				assert(fcard[0] == fcard[1]);

				if (randu() > edgeprob_out[fi])
					continue;	// factor does not appear

				qf_chosen[fi] = true;	// mark as selected
				const std::vector<unsigned int>& fvars = fac->Variables();
				assert(fvars.size() == 2);

				unsigned int d0 = dset.FindSet(fvars[0]);
				unsigned int d1 = dset.FindSet(fvars[1]);
				if (d0 != d1)
					dset.Link(d0, d1);	// Merge
			}

			// Now we have one roll out, add gradient contributions
			for (size_t fi_ei = 0; fi_ei < factors.size(); ++fi_ei) {
				const std::vector<unsigned int>& ei_fvars =
					factors[fi_ei]->Variables();
				if (ei_fvars.size() == 1)
					continue;

				// No (i,j) connection -> zero contribution
				if (dset.FindSet(ei_fvars[0]) != dset.FindSet(ei_fvars[1]))
					continue;

				double scaling_ei = qf_desired_cc[fi_ei] - qf_actual_cc[fi_ei];

				// Otherwise contributions to all edges
				for (unsigned int fi_st = 0; fi_st < factors.size(); ++fi_st) {
					if (factors[fi_st]->Cardinalities().size() != 2)
						continue;

					grad_apx[fi_st] += scaling_ei * (qf_chosen[fi_st] ?
						(-1.0 / edgeprob_out[fi_st]) :
						(1.0 / (1.0 - edgeprob_out[fi_st])));
				}
			}
		}
		// Normalize to obtain sample average estimate
		for (unsigned int fi = 0; fi < grad_apx.size(); ++fi)
			grad_apx[fi] /= static_cast<double>(mc_runs);

		// Step size
		double alpha = 1.0 / (static_cast<double>(iter) + alpha_m);

		// Collect result runs
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			edgeprob_out[fi] -= alpha * grad_apx[fi];
			edgeprob_out[fi] = std::min(1.0 - 1.0e-3, edgeprob_out[fi]);
			edgeprob_out[fi] = std::max(0.0, edgeprob_out[fi]);
		}
	}
	return (obj);
}

}

