
#ifndef GRANTE_SWENDSENWANGSAMPLER_H
#define GRANTE_SWENDSENWANGSAMPLER_H

#include <vector>
#include <tr1/unordered_set>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorGraphUtility.h"

namespace Grante {

/* The Swendsen-Wang sampler is described in
 *
 * [Barbu2005], Adrian Barbu and Song-Chun Zhu, "Generalizing Swendsen-Wang to
 *    Sampling Arbitrary Posterior Probabilities", PAMI, 2005.
 *
 * In particular we implement the SWC-3 generalized Gibbs sampler.  The edge
 * appearance probabilities ($q_e$ in [Barbu2005]) are user-defined.
 *
 * Right now, this sampler has two main assumptions:
 *    1. All factor types are unary or pairwise,
 *    2. All variables have the same cardinality.
 */
class SwendsenWangSampler {
public:
	// qf: array of factor sampling probabilities, one for each factor of the
	//    factor graph.  qf[fi] > 0.0, qf[fi] <= 1.0.  A large probability
	//    means this factor will be preserved more often.  These weights are
	//    application-dependent.
	explicit SwendsenWangSampler(const FactorGraph* fg,
		const std::vector<double>& qf);

	// Grow a partition around the given variable and resample the partition.
	// Return the size of the partition.
	size_t SampleSite(unsigned int var_index);

	// Perform a SW sweep.  Here we define a single "SW sweep" as repeatedly
	// resampling a partition around a random site until the cummulative
	// number of variables resampled is greater-equal to the number of model
	// variables.
	//
	// Return the mean partition size or zero if no sampling was done.
	double Sweep(unsigned int sweep_count);

	// Perform a single SW step, growing a cluster from a uniformly at random
	// chosen variable.
	//
	// Return the partition size of the cluster grown.
	size_t SingleStep(void);

	const std::vector<unsigned int>& State() const;

	// Set temperature: 1.0 is the original distribution, 0.0 the uniform
	// distribution.
	void SetInverseTemperature(double inv_temperature);

	void SetState(const std::vector<unsigned int>& new_state);

	// Heuristic to set the factor appearance probabilities.
	// A large temperature (> 1.0) will make the appearance probabilities
	// close to 0.5.
	// A small temperature (< 1.0, > 0.0) will make the appearance
	// probabilities more extreme towards zero and one, depending on the
	// mean-compat score.
	static double ComputeFactorProb(const FactorGraph* fg,
		std::vector<double>& qf_out, double logistic_temp = 1.0);

	// Naive Monte-Carlo computation of the pairwise network reliability.
	// After the run, the vector qf_out contains network reliabilities between
	// all (s,t) node pairs for which there are factors.  The edge appearance
	// probabilities qf are used to simulate the stochastic network.
	//
	//    qf_out[fi] >= 0, <= 1.0, the probability that s(fi), t(fi) are in
	//       one component.
	//
	// The estimate is unbiased, and if the true reliability is b, the
	// variance is (b (1-b)) / mc_runs.
	static void ComputeNetworkReliability(const FactorGraph* fg,
		std::vector<double>& qf, std::vector<double>& qf_out,
		unsigned int mc_runs = 1000);

	// Adjust desired "variable-pair co-cluster probabilities" qf_desired_cc
	// into edge appearance probabilities by taking into account the graph
	// structure.  This is done by estimating the network reliability of each
	// edge and globally adjusting the edge appearance probabilities so as to
	// realize the co-cluster probabilities.
	//
	// Return the edge appearance probabilities in edgeprob_out (these can be
	// used for creating the SW sampler).
	//
	// The return value is the estimated objective function
	//   0.5 * sum_{(i,j) in E(G)} (qf_desired_cc[i,j] - qf_actual_cc[i,j])^2.
	//
	// The estimated actual co-cluster probabilities are returned in
	// qf_actual_cc.
	static double AdjustFactorProbStochastic(const FactorGraph* fg,
		const std::vector<double>& qf_desired_cc,
		std::vector<double>& edgeprob_out,
		std::vector<double>& qf_actual_cc, unsigned int max_iter);

#if 0
// FIXME: obsoleted code
	static double AdjustFactorProb(const FactorGraph* fg,
		const std::vector<double>& q_cc, std::vector<double>& qf_out,
		unsigned int correction_iter = 5, unsigned int mc_runs = 1000);

	static double AdjustFactorProb(const FactorGraph* fg,
		const std::vector<double>& q_cc,
		std::vector<double>& qf_out, std::vector<double>& qf_actual_cc,
		unsigned int correction_iter = 5, unsigned int mc_runs = 1000);
#endif

private:
	const FactorGraph* fg;
	mutable std::vector<unsigned int> state;
	FactorGraphUtility fgu;
	std::vector<double> qf;	// factor appearance probabilities

	std::tr1::unordered_set<unsigned int> var_active;
	unsigned int label_count;

	double inv_temperature;

	// Random number generation
	boost::mt19937 rgen;
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu;

	boost::mt19937 rgen_var;
	boost::uniform_int<int> rdest_var;	// range [0,var_count-1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<int> > randu_var;
};

}

#endif

