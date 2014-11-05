
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <ctime>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "GibbsSampler.h"
#include "GibbsInference.h"
#include "MaximumLikelihood.h"
#include "MaximumCompositeLikelihood.h"
#include "MaximumPseudolikelihood.h"
#include "MaximumCrissCrossLikelihood.h"
#include "NaivePiecewiseTraining.h"
#include "FactorGraphObservation.h"
#include "FactorGraphStructurizer.h"
#include "NormalPrior.h"

#define BOOST_TEST_MODULE(CompositeLikelihoodTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

/* Recover the pseudolikelihood objective from the composite likelihood
 * method.  This is merely for testing, and not the fastest way to compute the
 * pseudolikelihood.
 */
BOOST_AUTO_TEST_CASE(MCLEPseudolikelihood)
{
	// PRNG
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4);
	// Random uniform pairwise energies
	for (unsigned int di = 0; di < w.size(); ++di)
		w[di] = randu();

	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a simple (0)---#---(1) model
	std::vector<unsigned int> vc(2, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac);

	// Sample a population from the true model
	fg.ForwardMap();
	Grante::GibbsSampler gibbs(&fg);
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 100;
	states.reserve(sample_count);
	gibbs.Sweep(1000);
	for (unsigned int si = 0; si < sample_count; ++si) {
		gibbs.Sweep(50);
		states.push_back(gibbs.State());
	}

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by CMLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		// Empty tree inference object, will be instantiated on subgraphs
		inference_methods.push_back(new Grante::TreeInference(0));
	}

	// MCLE-Pseudolikelihood
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);
	Grante::MaximumCompositeLikelihood mcle_mple(&model,
		Grante::MaximumCompositeLikelihood::DecomposePseudolikelihood);
	mcle_mple.SetupTrainingData(training_data, inference_methods);
	double mcle_mple_obj = mcle_mple.Train(1.0e-8);
	std::vector<double> w_mcle_mple(pt->Weights());

	// MPLE
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);
	Grante::MaximumPseudolikelihood mple(&model);
	mple.SetupTrainingData(training_data, inference_methods);
	double mple_obj = mple.Train(1.0e-8);
	std::vector<double> w_mple(pt->Weights());

	BOOST_CHECK_CLOSE_ABS(mcle_mple_obj, mple_obj, 1.0e-5);

	// Delete
	for (unsigned int n = 0; n < training_data.size(); ++n)
		delete (training_data[n].second);
}

BOOST_AUTO_TEST_CASE(MCLESimple)
{
	// PRNG
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4);
	// Random uniform pairwise energies
	for (unsigned int di = 0; di < w.size(); ++di)
		w[di] = randu();

	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N-by-N grid-structured model
	unsigned int N = 8;

	// Create a factor graph from the model
	std::vector<unsigned int> vc(N*N, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 1; x < N; ++x) {
			// Horizontal edge
			var_index[0] = y*N + x - 1;
			var_index[1] = y*N + x;

			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}
	for (unsigned int y = 1; y < N; ++y) {
		for (unsigned int x = 0; x < N; ++x) {
			// Vertical edge
			var_index[0] = (y-1)*N + x;
			var_index[1] = y*N + x;

			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}

	fg.ForwardMap();
	Grante::GibbsSampler gibbs(&fg);

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 500;
	states.reserve(sample_count);
	gibbs.Sweep(1000);
	for (unsigned int si = 0; si < sample_count; ++si) {
		gibbs.Sweep(50);
		states.push_back(gibbs.State());
	}

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by CMLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		// Empty tree inference object, will be instantiated on subgraphs
		inference_methods.push_back(new Grante::TreeInference(0));
	}
	Grante::MaximumCompositeLikelihood mcle(&model);
	mcle.SetupTrainingData(training_data, inference_methods);
	mcle.Train(1e-6);

	double w_true_mean = std::accumulate(w_truth.begin(), w_truth.end(), 0.0)
		/ static_cast<double>(w_truth.size());
	double w_pred_mean = std::accumulate(pt->Weights().begin(),
		pt->Weights().end(), 0.0) / static_cast<double>(pt->Weights().size());
	std::cout << "Maximum composite likelihood estimator" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		double w_adj = pt->Weights()[wi] - w_pred_mean + w_true_mean;
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi]
			<< ", adjusted " << w_adj << std::endl;
		BOOST_CHECK(std::fabs(w_truth[wi] - w_adj) <= 0.1);
	}

	// Criss-cross likelihood
	Grante::MaximumCrissCrossLikelihood mxxle(&model);
	mxxle.SetupTrainingData(training_data, inference_methods);
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);
	mxxle.Train(1e-6);

	w_pred_mean = std::accumulate(pt->Weights().begin(),
		pt->Weights().end(), 0.0) / static_cast<double>(pt->Weights().size());
	std::cout << "Criss-cross likelihood estimator" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		double w_adj = pt->Weights()[wi] - w_pred_mean + w_true_mean;
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi]
			<< ", adjusted " << w_adj << std::endl;
		BOOST_CHECK(std::fabs(w_truth[wi] - w_adj) <= 0.1);
	}

	// 4-cover MCLE
	Grante::MaximumCompositeLikelihood mcle4(&model, 4);
	mcle4.SetupTrainingData(training_data, inference_methods);
	mcle4.Train(1e-6);

	w_pred_mean = std::accumulate(pt->Weights().begin(),
		pt->Weights().end(), 0.0) / static_cast<double>(pt->Weights().size());
	std::cout << "Maximum 4-fold composite likelihood estimator" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		double w_adj = pt->Weights()[wi] - w_pred_mean + w_true_mean;
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi]
			<< ", adjusted " << w_adj << std::endl;
		BOOST_CHECK(std::fabs(w_truth[wi] - w_adj) <= 0.1);
	}

	// Naive piecewise training
	Grante::NaivePiecewiseTraining pwmle(&model);
	pwmle.SetupTrainingData(training_data, inference_methods);
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);
	pwmle.Train(1e-6);

	w_pred_mean = std::accumulate(pt->Weights().begin(),
		pt->Weights().end(), 0.0) / static_cast<double>(pt->Weights().size());
	std::cout << "Piecewise training estimator" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		double w_adj = pt->Weights()[wi] - w_pred_mean + w_true_mean;
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi]
			<< ", adjusted " << w_adj << std::endl;
		//BOOST_CHECK_CLOSE_ABS(w_truth[wi], w_adj, 0.025);
	}

	// Check MCLE-MPLE and MPLE yield the same objective
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);
	Grante::MaximumCompositeLikelihood mcle_mple(&model,
		Grante::MaximumCompositeLikelihood::DecomposePseudolikelihood);
	mcle_mple.SetupTrainingData(training_data, inference_methods);
	double mcle_mple_obj = mcle_mple.Train(1e-7);

	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);
	Grante::MaximumPseudolikelihood mple(&model);
	mple.SetupTrainingData(training_data, inference_methods);
	double mple_obj = mple.Train(1e-7);
	BOOST_CHECK_CLOSE_ABS(mple_obj, mcle_mple_obj, 1.0e-4);

	// Delete
	for (unsigned int n = 0; n < training_data.size(); ++n)
		delete (training_data[n].second);
}

BOOST_AUTO_TEST_CASE(MCLEExpectationTarget)
{
	// PRNG
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4);
	// Random uniform pairwise energies
	for (unsigned int di = 0; di < w.size(); ++di)
		w[di] = randu();

	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N1-by-N2 grid-structured model
	unsigned int N1 = 4;
	unsigned int N2 = 7;

	// Create a factor graph from the model
	std::vector<unsigned int> vc(N1*N2, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	for (unsigned int y = 0; y < N1; ++y) {
		for (unsigned int x = 1; x < N2; ++x) {
			// Horizontal edge
			var_index[0] = y*N2 + x - 1;
			var_index[1] = y*N2 + x;

			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}
	for (unsigned int y = 1; y < N1; ++y) {
		for (unsigned int x = 0; x < N2; ++x) {
			// Vertical edge
			var_index[0] = (y-1)*N2 + x;
			var_index[1] = y*N2 + x;

			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}

	fg.ForwardMap();
	// TODO
	std::vector<std::vector<unsigned int> > var_rows;
	std::vector<std::vector<unsigned int> > var_cols;
	bool is_grid =
		Grante::FactorGraphStructurizer::IsOrderedPairwiseGridStructured(
			&fg, var_rows, var_cols);
	BOOST_CHECK(is_grid);
	for (unsigned int ri = 0; ri < var_rows.size(); ++ri) {
		std::cout << "row:";
		for (unsigned int ci = 0; ci < var_rows[ri].size(); ++ci)
			std::cout << " " << var_rows[ri][ci];
		std::cout << std::endl;
	}
	for (unsigned int ci = 0; ci < var_cols.size(); ++ci) {
		std::cout << "col:";
		for (unsigned int ri = 0; ri < var_cols[ci].size(); ++ri)
			std::cout << " " << var_cols[ci][ri];
		std::cout << std::endl;
	}

	// Obtain approximate marginals by Gibbs sampling
	Grante::GibbsInference ginf(&fg);
	ginf.SetSamplingParameters(1000, 1, 50000);
	ginf.PerformInference();
	std::vector<std::vector<double> > marg = ginf.Marginals();

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population marginals by CMLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	training_data.push_back(
		Grante::ParameterEstimationMethod::labeled_instance_type(
			&fg, new Grante::FactorGraphObservation(marg)));
	inference_methods.push_back(new Grante::TreeInference(0));

	// Composite likelihood
	Grante::MaximumCompositeLikelihood mcle(&model);
	mcle.SetupTrainingData(training_data, inference_methods);
	mcle.Train(1e-6);

	double w_true_mean = std::accumulate(w_truth.begin(), w_truth.end(), 0.0)
		/ static_cast<double>(w_truth.size());
	double w_pred_mean = std::accumulate(pt->Weights().begin(),
		pt->Weights().end(), 0.0) / static_cast<double>(pt->Weights().size());
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		double w_adj = pt->Weights()[wi] - w_pred_mean + w_true_mean;
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi]
			<< ", adjusted " << w_adj << std::endl;
		BOOST_CHECK_CLOSE_ABS(w_truth[wi], w_adj, 0.25);
	}
	for (unsigned int n = 0; n < training_data.size(); ++n)
		delete (training_data[n].second);
}

// Test case for training on expectations
BOOST_AUTO_TEST_CASE(MLECRFSimple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type:
	// Each energy is the inner product of a two-vector data with a
	// state-specific weight vector.
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.3);	// for (0,0)
	w.push_back(0.5);	// for (0,0)
	w.push_back(1.0);	// for (1,0)
	w.push_back(0.2);	// for (1,0)
	w.push_back(0.05);	// for (0,1)
	w.push_back(0.6);	// for (0,1)
	w.push_back(-0.2);	// for (1,1)
	w.push_back(0.75);	// for (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);

	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> var_index(2);

	// Randomly set the data observations
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Create a set of factor graphs realizing this model
	unsigned int instance_count = 1000;
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data_bad;
	std::vector<Grante::InferenceMethod*> inference_methods;
	training_data.reserve(instance_count);
	training_data_bad.reserve(instance_count);
	inference_methods.reserve(instance_count);
	for (unsigned int n = 0; n < instance_count; ++n) {
		Grante::FactorGraph* fg = new Grante::FactorGraph(&model, vc);
		// Add factors
		data[0] = 2.0*randu() - 1.0;
		data[1] = 2.0*randu() - 1.0;
		var_index[0] = 0;
		var_index[1] = 1;
		Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
		fg->AddFactor(fac1);

		data[0] = 2.0*randu() - 1.0;
		data[1] = 2.0*randu() - 1.0;
		var_index[0] = 1;
		var_index[1] = 2;
		Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
		fg->AddFactor(fac2);

		// Compute the forward map
		fg->ForwardMap();

		// Perform inference
		Grante::TreeInference* tinf = new Grante::TreeInference(fg);
		inference_methods.push_back(tinf);
		tinf->PerformInference();
		std::vector<std::vector<double> > marg_true = tinf->Marginals();
		std::vector<std::vector<double> > marg_bad(marg_true);
		for (unsigned int fi = 0; fi < marg_bad.size(); ++fi) {
			std::fill(marg_bad[fi].begin(), marg_bad[fi].end(),
				1.0/static_cast<double>(marg_bad[fi].size()));
		}

		// Use exact marginals as target distribution
		// (multiple instances should ensure identifiable parameters)
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				fg, new Grante::FactorGraphObservation(marg_true)));
		training_data_bad.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				fg, new Grante::FactorGraphObservation(marg_bad)));
	}

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	Grante::MaximumCompositeLikelihood mle(&model);
	mle.SetupTrainingData(training_data_bad, inference_methods);
	mle.AddPrior("pairwise", new Grante::NormalPrior(10.0, w.size()));
	mle.Train(1e-5);
	mle.UpdateTrainingLabeling(training_data);
	mle.Train(1e-5);

	// Compare marginals produced by learned weights with truth
	for (unsigned int n = 0; n < instance_count; ++n) {
		training_data[n].first->ForwardMap();
		inference_methods[n]->PerformInference();
		std::vector<std::vector<double> > marg_learned =
			inference_methods[n]->Marginals();
		const std::vector<std::vector<double> >& marg_truth =
			training_data[n].second->Expectation();

		assert(marg_learned.size() == marg_truth.size());
		for (unsigned int fi = 0; fi < marg_learned.size(); ++fi) {
			assert(marg_learned[fi].size() == marg_truth[fi].size());
			for (unsigned int ei = 0; ei < marg_learned[fi].size(); ++ei) {
#if 0
				std::cout << "instance " << n << ", factor " << fi
					<< ", element " << ei << ": learned " << marg_learned[fi][ei]
					<< ", truth " << marg_truth[fi][ei] << std::endl;
#endif
				BOOST_CHECK_CLOSE_ABS(marg_truth[fi][ei], marg_learned[fi][ei], 0.025);
			}
		}
		delete (inference_methods[n]);
		delete (training_data[n].first);
		delete (training_data[n].second);
	}
}

