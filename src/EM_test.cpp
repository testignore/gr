
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "Conditioning.h"
#include "TreeInference.h"
#include "DiffusionInference.h"
#include "StructuredMeanFieldInference.h"
#include "AISInference.h"
#include "BeliefPropagation.h"
#include "BruteForceExactInference.h"
#include "ExpectationMaximization.h"
#include "MaximumLikelihood.h"
#include "MaximumCompositeLikelihood.h"
#include "MaximumPseudolikelihood.h"
#include "NormalPrior.h"
#include "FactorConditioningTable.h"

#define BOOST_TEST_MODULE(EMTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

#if 1
BOOST_AUTO_TEST_CASE(Simple)
{
	Grante::FactorGraphModel model;

	// Create one data-independent pairwise factor type that encourages taking
	// the identical value
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(4);
	std::vector<double> w12;
	w12.push_back(0.0);	// (0,0)
	w12.push_back(0.0);	// (1,0)
	w12.push_back(0.0);	// (0,1)
	w12.push_back(0.0);	// (1,1)
	w12.push_back(1.0);	// (0,2)
	w12.push_back(1.0);	// (1,2)
	w12.push_back(1.0);	// (0,3)
	w12.push_back(1.0);	// (1,3)
	Grante::FactorType* factortype12 =
		new Grante::FactorType("pairwise12", card, w12);
	model.AddFactorType(factortype12);

	// Add a factor type that encourages taking the opposite value
	std::vector<double> w23;
	w23.push_back(1.0);	// (0,0)
	w23.push_back(0.0);	// (1,0)
	w23.push_back(1.0);	// (0,1)
	w23.push_back(1.0);	// (1,1)
	w23.push_back(0.0);	// (0,2)
	w23.push_back(0.0);	// (1,2)
	w23.push_back(1.0);	// (0,3)
	w23.push_back(1.0);	// (1,3)
	Grante::FactorType* factortype23 =
		new Grante::FactorType("pairwise23", card, w23);
	model.AddFactorType(factortype23);

	// Create a factor graph from the model: 2 binary variables, one 4-state
	// variable:
	//
	//    (0) ---[12]--- (1) ---[23]--- (2),
	//
	// Variable (1) is unobserved.
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(4);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt12 = model.FindFactorType("pairwise12");
	BOOST_REQUIRE(pt12 != 0);
	Grante::FactorType* pt23 = model.FindFactorType("pairwise23");
	BOOST_REQUIRE(pt23 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt12, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 2;
	var_index[1] = 1;
	Grante::Factor* fac2 = new Grante::Factor(pt23, var_index, data);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Get marginals
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_true = tinf.Marginals();

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 1500;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth12(pt12->Weights());
	std::vector<double> w_truth23(pt23->Weights());
	std::fill(pt12->Weights().begin(), pt12->Weights().end(), 0.0);
	std::fill(pt23->Weights().begin(), pt23->Weights().end(), 0.0);

#if 0
	// Seems to work fine with SMF inference
	Grante::FactorConditioningTable fcond_tab;
	Grante::StructuredMeanFieldInference smfinf(&fg, &fcond_tab);
	smfinf.SetParameters(false, 1.0e-7, 30);
#endif

	// TRAINING RUN 1: Fully observed data
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		o_training_data;
	std::vector<Grante::InferenceMethod*> o_inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		o_training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		// Push the same inference object again (graph is of fixed-structure)
		o_inference_methods.push_back(&tinf);
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(o_training_data, o_inference_methods);
	double mle_obj = mle.Train(1e-5);
	std::cout << "MLE obj: " << mle_obj << std::endl;

	// Reset model parameters
	std::fill(pt12->Weights().begin(), pt12->Weights().end(), 0.0);
	std::fill(pt23->Weights().begin(), pt23->Weights().end(), 0.0);
	boost::mt19937 rgen2(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen2, rdestu);
	// Random initialization
	for (unsigned int wi = 0; wi < pt12->Weights().size(); ++wi)
		pt12->Weights()[wi] = randu() - 0.5;
	for (unsigned int wi = 0; wi < pt23->Weights().size(); ++wi)
		pt23->Weights()[wi] = randu() - 0.5;

	// Reconstruct model weights from population by EM-MLE
	std::vector<Grante::ExpectationMaximization::partially_labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	std::vector<unsigned int> var_subset;
	var_subset.push_back(0);
	var_subset.push_back(2);
	for (unsigned int si = 0; si < states.size(); ++si) {
		std::vector<unsigned int> cur_state(2);
		// Do not observe variable 1
		cur_state[0] = states[si][0];
		cur_state[1] = states[si][2];
		training_data.push_back(
			Grante::ExpectationMaximization::partially_labeled_instance_type(
				&fg, new Grante::FactorGraphPartialObservation(
					var_subset, cur_state)));

		// Push the same inference object again (graph is of fixed-structure)
		inference_methods.push_back(&tinf);
#if 0
		inference_methods.push_back(&smfinf);
#endif
	}
	Grante::MaximumLikelihood* parest = new Grante::MaximumLikelihood(&model);
	Grante::ExpectationMaximization em(&model, parest);
	std::cout << "EM SetupTrainingData" << std::endl;
	em.SetupTrainingData(training_data, inference_methods, inference_methods);
	em.AddPrior("pairwise12", new Grante::NormalPrior(10.0, w_truth12.size()));
	em.AddPrior("pairwise23", new Grante::NormalPrior(10.0, w_truth23.size()));
	std::cout << "EM Train" << std::endl;
	em.Train(1.0e-5, 10, 1.0e-6, 100);

	std::cout << "Factor 1-2" << std::endl;
	for (unsigned int wi = 0; wi < w_truth12.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth12[wi]
			<< ", learned " << pt12->Weights()[wi] << std::endl;
	}
	std::cout << "Factor 2-3" << std::endl;
	for (unsigned int wi = 0; wi < w_truth23.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth23[wi]
			<< ", learned " << pt23->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			//BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n)
		delete (training_data[n].second);

	// Compare (0)/(2) marginals
	std::vector<std::vector<unsigned int> > learned_states;
	fg.ForwardMap();
	tinf.Sample(learned_states, sample_count);
	double ta = 1.0 / static_cast<double>(sample_count);
	std::vector<double> mtrue_02(4, 0.0);
	std::vector<double> mlearned_02(4, 0.0);
	for (unsigned int si = 0; si < sample_count; ++si) {
		mtrue_02[states[si][0] + 2*states[si][2]] += ta;
		mlearned_02[learned_states[si][0] + 2*learned_states[si][2]] += ta;
	}
	for (unsigned int mi = 0; mi < mtrue_02.size(); ++mi) {
		std::cout << "(0,2), 0, true: " << mtrue_02[mi]
			<< ", learned: " << mlearned_02[mi] << std::endl;
		BOOST_CHECK_CLOSE_ABS(mtrue_02[mi], mlearned_02[mi], 0.05);
	}
}

BOOST_AUTO_TEST_CASE(SimpleData)
{
	Grante::FactorGraphModel model;

	// Create one data-independent pairwise factor type that encourages taking
	// the identical value
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(4);
	std::vector<double> w12;
	w12.push_back(-10.0);	// (0,0)
	w12.push_back(10.0);	// (1,0)
	w12.push_back(-10.0);	// (0,1)
	w12.push_back(10.0);	// (1,1)
	w12.push_back(10.0);	// (0,2)
	w12.push_back(-10.0);	// (1,2)
	w12.push_back(10.0);	// (0,3)
	w12.push_back(-10.0);	// (1,3)
	Grante::FactorType* factortype12 =
		new Grante::FactorType("pairwise12", card, w12);
	model.AddFactorType(factortype12);

	// Add a factor type that encourages taking the opposite value
	std::vector<double> w23;
	w23.push_back(-10.0);	// (0,0)
	w23.push_back(10.0);	// (1,0)
	w23.push_back(10.0);	// (0,1)
	w23.push_back(-10.0);	// (1,1)
	w23.push_back(-10.0);	// (0,2)
	w23.push_back(10.0);	// (1,2)
	w23.push_back(10.0);	// (0,3)
	w23.push_back(-10.0);	// (1,3)
	Grante::FactorType* factortype23 =
		new Grante::FactorType("pairwise23", card, w23);
	model.AddFactorType(factortype23);

	std::vector<unsigned int> card1;
	card1.push_back(4);
	std::vector<double> w1;
	w1.push_back(-10.0);	// (0)
	w1.push_back(+10.0);	// (0)
	w1.push_back(+10.0);	// (0)
	w1.push_back(+10.0);	// (0)
	w1.push_back(+10.0);	// (1)
	w1.push_back(-10.0);	// (1)
	w1.push_back(+10.0);	// (1)
	w1.push_back(+10.0);	// (1)
	w1.push_back(+10.0);	// (2)
	w1.push_back(+10.0);	// (2)
	w1.push_back(-10.0);	// (2)
	w1.push_back(+10.0);	// (2)
	w1.push_back(+10.0);	// (3)
	w1.push_back(+10.0);	// (3)
	w1.push_back(+10.0);	// (3)
	w1.push_back(-10.0);	// (3)
	Grante::FactorType* factortype1 =
		new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1);

	// Create a factor graph from the model: 2 binary variables (0), (2), one
	// 4-state variable (1):
	//
	//    (0) ---[12]--- (1) ---[23]--- (2),
	//                    |
	//                   [1]
	//
	// Variable (1) is unobserved but data coming from factor [1] provides a
	// perfect cue at reconstructing the state of (1).
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(4);
	vc.push_back(2);

	unsigned int sample_count = 500;
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+2);
	boost::uniform_int<unsigned int> rdestd(0, 3);
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand(rgen, rdestd);

	// EM training data
	std::vector<Grante::ExpectationMaximization::partially_labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	std::vector<unsigned int> var_subset;
	var_subset.push_back(0);
	var_subset.push_back(2);
	Grante::FactorType* pt12 = model.FindFactorType("pairwise12");
	BOOST_REQUIRE(pt12 != 0);
	Grante::FactorType* pt23 = model.FindFactorType("pairwise23");
	BOOST_REQUIRE(pt23 != 0);
	Grante::FactorType* ptH = model.FindFactorType("unary1");
	BOOST_REQUIRE(ptH != 0);
	std::vector<unsigned int> L;
	for (unsigned int si = 0; si < sample_count; ++si) {
		Grante::FactorGraph* fg = new Grante::FactorGraph(&model, vc);

		// Add factors
		std::vector<double> data;
		std::vector<unsigned int> var_index(2);
		var_index[0] = 0;
		var_index[1] = 1;
		Grante::Factor* fac1 = new Grante::Factor(pt12, var_index, data);
		fg->AddFactor(fac1);
		var_index[0] = 2;
		var_index[1] = 1;
		Grante::Factor* fac2 = new Grante::Factor(pt23, var_index, data);
		fg->AddFactor(fac2);

		// Add factor to hidden
		std::vector<double> data1(4);
		unsigned int latent_state = rand();
		switch (latent_state) {
		case (0):
			data1[0] = 1.0;
			data1[1] = -1.0;
			data1[2] = -1.0;
			data1[3] = -1.0;
			break;
		case (1):
			data1[0] = -1.0;
			data1[1] = 1.0;
			data1[2] = -1.0;
			data1[3] = -1.0;
			break;
		case (2):
			data1[0] = -1.0;
			data1[1] = -1.0;
			data1[2] = 1.0;
			data1[3] = -1.0;
			break;
		case (3):
			data1[0] = -1.0;
			data1[1] = -1.0;
			data1[2] = -1.0;
			data1[3] = 1.0;
			break;
		default:
			assert(0);
			break;
		}
		std::vector<unsigned int> var_index1(1);
		var_index1[0] = 1;
		Grante::Factor* facH = new Grante::Factor(ptH, var_index1, data1);
		fg->AddFactor(facH);

		// Compute the forward map
		fg->ForwardMap();

		// Get marginals
		Grante::TreeInference* tinf = new Grante::TreeInference(fg);
		tinf->PerformInference();
		std::vector<std::vector<double> > marg_true = tinf->Marginals();

		// Sample a population from the true model
		std::vector<std::vector<unsigned int> > states;
		tinf->Sample(states, 1);

		std::vector<unsigned int> cur_state(2);
		// Do not observe variable 1
		cur_state[0] = states[0][0];
		cur_state[1] = states[0][2];
		std::cout << "latent_state " << latent_state
			<< ", v0 " << cur_state[0] << ", v2 " << cur_state[1] << std::endl;
		L.push_back(latent_state);
		training_data.push_back(
			Grante::ExpectationMaximization::partially_labeled_instance_type(
				fg, new Grante::FactorGraphPartialObservation(
					var_subset, cur_state)));
		inference_methods.push_back(tinf);
	}

	// Change model parameters
	std::vector<double> w_truth12(pt12->Weights());
	std::vector<double> w_truth23(pt23->Weights());
	std::vector<double> w_truth1(ptH->Weights());
#if 1
	std::fill(pt12->Weights().begin(), pt12->Weights().end(), 0.0);
	std::fill(pt23->Weights().begin(), pt23->Weights().end(), 0.0);
	std::fill(ptH->Weights().begin(), ptH->Weights().end(), 0.0);
#endif
	boost::mt19937 rgen2(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen2, rdestu);
	// Random initialization
	for (unsigned int wi = 0; wi < pt12->Weights().size(); ++wi)
		pt12->Weights()[wi] = randu() - 0.5;
	for (unsigned int wi = 0; wi < pt23->Weights().size(); ++wi)
		pt23->Weights()[wi] = randu() - 0.5;
	for (unsigned int wi = 0; wi < ptH->Weights().size(); ++wi)
		ptH->Weights()[wi] = randu() - 0.5;

	// Reconstruct model weights from population by EM-MLE
	Grante::MaximumLikelihood* parest = new Grante::MaximumLikelihood(&model);
	Grante::ExpectationMaximization em(&model, parest);
	std::cout << "EM SetupTrainingData" << std::endl;
	em.SetupTrainingData(training_data, inference_methods, inference_methods);
	em.AddPrior("pairwise12", new Grante::NormalPrior(10.0, w_truth12.size()));
	em.AddPrior("pairwise23", new Grante::NormalPrior(10.0, w_truth23.size()));
	em.AddPrior("unary1", new Grante::NormalPrior(10.0, w_truth1.size()));
	std::cout << "EM Train" << std::endl;
	em.Train(1.0e-5, 10, 1.0e-6, 100);

	std::cout << "Factor 12" << std::endl;
	for (unsigned int wi = 0; wi < w_truth12.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth12[wi]
			<< ", learned " << pt12->Weights()[wi] << std::endl;
	}
	std::cout << "Factor 23" << std::endl;
	for (unsigned int wi = 0; wi < w_truth23.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth23[wi]
			<< ", learned " << pt23->Weights()[wi] << std::endl;
	}
	std::cout << "Factor H" << std::endl;
	for (unsigned int wi = 0; wi < w_truth1.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth1[wi]
			<< ", learned " << ptH->Weights()[wi] << std::endl;
	}

	// Test learned model
	unsigned int error_count = 0;
	for (unsigned int ti = 0; ti < sample_count; ++ti) {
		training_data[ti].first->ForwardMap();
#if 0
		if (ti < 20) {
			std::cout << "latent_state " << L[ti] << std::endl;
			const std::vector<double>& Henergies =
				training_data[ti].first->Factors()[2]->Energies();
			for (unsigned int hi = 0; hi < Henergies.size(); ++hi)
				std::cout << "hi " << hi << ", E " << Henergies[hi]
					<< std::endl;
//			assert(0);
		}
#endif

		std::vector<std::vector<unsigned int> > states;
		inference_methods[ti]->PerformInference();
		inference_methods[ti]->Sample(states, 1);
		if (ti < 20) {
			std::cout << "v0 " << states[0][0] << ", v1 " << states[0][1]
				<< "(L=" << L[ti] << "), v2 " << states[0][2] << std::endl;
		}
		if ((2*states[0][0]+states[0][2])==L[ti])
			continue;
		error_count += 1;
		//BOOST_CHECK((2*states[0][0]+states[0][2])==states[0][1]);
	}
	std::cout << error_count << " out of " << sample_count << " errors."
		<< std::endl;
	BOOST_CHECK_LT(error_count, sample_count/3);
}
#endif

#if 1
BOOST_AUTO_TEST_CASE(SimpleCyclic)
{
	Grante::FactorGraphModel model;

	// Setup a model as follows:
	//
	//  (0) --[1]-- (2) --[4]-- (4)
	//   |           |           |
	//  [0]         [3]         [6]
	//   |           |           |
	//  (1) --[2]-- (3) --[5]-- (5)
	//
	// Where (0),(1),(4),(5) is observed but (2),(3) are unobserved.
	// At test time, we observe only (0) and (1) and want to infer (4) and (5).
	// Each factor has its own type.
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w2(4, 0.0);
	Grante::FactorType* factortype01 =
		new Grante::FactorType("pairwise01", card, w2);
	model.AddFactorType(factortype01);
	Grante::FactorType* factortype02 =
		new Grante::FactorType("pairwise02", card, w2);
	model.AddFactorType(factortype02);
	Grante::FactorType* factortype13 =
		new Grante::FactorType("pairwise13", card, w2);
	model.AddFactorType(factortype13);
	Grante::FactorType* factortype23 =
		new Grante::FactorType("pairwise23", card, w2);
	model.AddFactorType(factortype23);
	Grante::FactorType* factortype24 =
		new Grante::FactorType("pairwise24", card, w2);
	model.AddFactorType(factortype24);
	Grante::FactorType* factortype35 =
		new Grante::FactorType("pairwise35", card, w2);
	model.AddFactorType(factortype35);
	Grante::FactorType* factortype45 =
		new Grante::FactorType("pairwise45", card, w2);
	model.AddFactorType(factortype45);

	// Create the factor graph
	std::vector<unsigned int> vc(6, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	fg.AddFactor(new Grante::Factor(factortype01, var_index, data));

	var_index[0] = 0;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(factortype02, var_index, data));

	var_index[0] = 1;
	var_index[1] = 3;
	fg.AddFactor(new Grante::Factor(factortype13, var_index, data));

	var_index[0] = 2;
	var_index[1] = 3;
	fg.AddFactor(new Grante::Factor(factortype23, var_index, data));

	var_index[0] = 2;
	var_index[1] = 4;
	fg.AddFactor(new Grante::Factor(factortype24, var_index, data));

	var_index[0] = 3;
	var_index[1] = 5;
	fg.AddFactor(new Grante::Factor(factortype35, var_index, data));

	var_index[0] = 4;
	var_index[1] = 5;
	fg.AddFactor(new Grante::Factor(factortype45, var_index, data));

	// Use structured mean field for hidden inference
#if 1
	Grante::FactorConditioningTable smf_fcond_tab;
	Grante::StructuredMeanFieldInference smfinf(&fg, &smf_fcond_tab);
	smfinf.SetParameters(false, 1.0e-7, 50);
#endif
#if 0
	Grante::AISInference aisinf(&fg);
	aisinf.SetSamplingParameters(20, 1, 100);
#endif
#if 0
	Grante::DiffusionInference diffinf(&fg);
#endif
#if 0
	Grante::BeliefPropagation lbpinf(&fg);
	lbpinf.SetParameters(false, 30, 1.0e-5);
#endif
	Grante::BruteForceExactInference bfinf(&fg);

#if 0
	// fg=0 here because it will be instantiated in the composite likelihood
	Grante::TreeInference tinf(0);
#endif

	// Learn model weights from population by EM-MCLE
	std::vector<Grante::ExpectationMaximization::partially_labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> hidden_inference_methods;
	std::vector<Grante::InferenceMethod*> observed_inference_methods;
	std::vector<Grante::InferenceMethod*> parest_inference_methods;
	std::vector<unsigned int> var_subset;
	var_subset.push_back(0);
	var_subset.push_back(1);
	var_subset.push_back(4);
	var_subset.push_back(5);

	// Produce 70%/30% samples of A/B types,
	unsigned int sample_count = 100;
	for (unsigned int si = 0; si < sample_count; ++si) {
		std::vector<unsigned int> cur_state(4);
		if (si <= sample_count/2) {
			// A: (0)=1,(1)=0,(4)=0,(5)=1,
			cur_state[0] = 1;
			cur_state[1] = 0;
			cur_state[2] = 0;
			cur_state[3] = 1;
		} else {
			// B: (0)=0,(1)=1,(4)=1,(5)=0.
			cur_state[0] = 0;
			cur_state[1] = 1;
			cur_state[2] = 1;
			cur_state[3] = 0;
		}
		training_data.push_back(
			Grante::ExpectationMaximization::partially_labeled_instance_type(
				&fg, new Grante::FactorGraphPartialObservation(
					var_subset, cur_state)));

		// Push the same inference object again (graph is of fixed-structure)
//		hidden_inference_methods.push_back(&aisinf);
		hidden_inference_methods.push_back(&smfinf);
//		hidden_inference_methods.push_back(&tinf);
//		hidden_inference_methods.push_back(&bfinf);
//		observed_inference_methods.push_back(&aisinf);
		observed_inference_methods.push_back(&smfinf);
//		observed_inference_methods.push_back(&bfinf);
		//parest_inference_methods.push_back(&tinf);
		//parest_inference_methods.push_back(&lbpinf);
		parest_inference_methods.push_back(&bfinf);
		//parest_inference_methods.push_back(&smfinf);
		//parest_inference_methods.push_back(&aisinf);
		//parest_inference_methods.push_back(&diffinf);
	}

	// Random initialization
	boost::mt19937 rgen2(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen2, rdestu);
	// Random initialization
	for (unsigned int wi = 0; wi < factortype01->Weights().size(); ++wi) {
		factortype01->Weights()[wi] = randu() - 0.5;
		factortype02->Weights()[wi] = randu() - 0.5;
		factortype13->Weights()[wi] = randu() - 0.5;
		factortype23->Weights()[wi] = randu() - 0.5;
		factortype24->Weights()[wi] = randu() - 0.5;
		factortype35->Weights()[wi] = randu() - 0.5;
		factortype45->Weights()[wi] = randu() - 0.5;
	}
	std::cout << "SETUP done" << std::endl;

#if 1
	Grante::MaximumLikelihood* parest =
		new Grante::MaximumLikelihood(&model);
#endif
#if 0
	// TODO: so far, MCLE for the M-step does not work well
	Grante::MaximumCompositeLikelihood* parest =
		new Grante::MaximumCompositeLikelihood(&model);
#endif
	Grante::ExpectationMaximization em(&model, parest);
	std::cout << "EM SetupTrainingData" << std::endl;
	em.SetupTrainingData(training_data, hidden_inference_methods,
		observed_inference_methods, parest_inference_methods);
#if 0
	em.AddPrior("pairwise01", new Grante::NormalPrior(1.0, 4));
	em.AddPrior("pairwise02", new Grante::NormalPrior(1.0, 4));
	em.AddPrior("pairwise13", new Grante::NormalPrior(1.0, 4));
	em.AddPrior("pairwise23", new Grante::NormalPrior(1.0, 4));
	em.AddPrior("pairwise24", new Grante::NormalPrior(1.0, 4));
	em.AddPrior("pairwise35", new Grante::NormalPrior(1.0, 4));
	em.AddPrior("pairwise45", new Grante::NormalPrior(1.0, 4));
#endif

	std::cout << "EM Train" << std::endl;
	em.Train(1.0e-5, 50, 1.0e-8, 100);

	const std::vector<double>& f01w = factortype01->Weights();
	for (unsigned int wi = 0; wi < f01w.size(); ++wi) {
		std::cout << "ft01, weights[" << wi << "]: " << f01w[wi]
			<< std::endl;
	}
	const std::vector<double>& f02w = factortype02->Weights();
	for (unsigned int wi = 0; wi < f02w.size(); ++wi) {
		std::cout << "ft02, weights[" << wi << "]: " << f02w[wi]
			<< std::endl;
	}

	// Test
	// Condition the factor graph on (0)=1, (1)=0
	Grante::FactorConditioningTable ftab;
	std::vector<unsigned int> cond_var_set;
	std::vector<unsigned int> cond_var_state;
	cond_var_set.push_back(0);
	cond_var_set.push_back(1);
	cond_var_state.push_back(1);
	cond_var_state.push_back(0);
	std::vector<unsigned int> var_new_to_orig;

	// Test conditioned energies
	Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_var_state);
	Grante::FactorGraph* fg_condA = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobs, var_new_to_orig);
	fg_condA->ForwardMap();
	Grante::AISInference ainf_condA(fg_condA);
	ainf_condA.PerformInference();
	std::cout << "marg5 is between variables "
		<< var_new_to_orig[fg_condA->Factors()[5]->Variables()[0]]
		<< " and "
		<< var_new_to_orig[fg_condA->Factors()[5]->Variables()[1]]
		<< std::endl;
	const std::vector<double>& marg5 = ainf_condA.Marginal(5);
	BOOST_CHECK(marg5.size() == 4);
	std::cout << "marg5: " << marg5[0] << " " << marg5[1]
		<< " " << marg5[2] << " " << marg5[3] << std::endl;
	// Should be: 0 0 1 0
	BOOST_CHECK(marg5[2] > marg5[0]);
	BOOST_CHECK(marg5[2] > marg5[1]);
	BOOST_CHECK(marg5[2] > marg5[3]);

	// Condition the factor graph on (0)=0, (1)=1
	cond_var_state[0] = 0;
	cond_var_state[1] = 1;
	var_new_to_orig.clear();

	// Test conditioned energies
	Grante::FactorGraphPartialObservation pobsB(cond_var_set, cond_var_state);
	Grante::FactorGraph* fg_condB = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobsB, var_new_to_orig);
	fg_condB->ForwardMap();
	Grante::AISInference ainf_condB(fg_condB);
	ainf_condB.PerformInference();
	std::cout << "marg5 is between variables "
		<< var_new_to_orig[fg_condB->Factors()[5]->Variables()[0]]
		<< " and "
		<< var_new_to_orig[fg_condB->Factors()[5]->Variables()[1]]
		<< std::endl;
	const std::vector<double>& marg5B = ainf_condB.Marginal(5);
	BOOST_CHECK(marg5B.size() == 4);
	std::cout << "marg5: " << marg5B[0] << " " << marg5B[1]
		<< " " << marg5B[2] << " " << marg5B[3] << std::endl;
	// Should be: 0 1 0 0
	BOOST_CHECK(marg5B[1] > marg5B[0]);
	BOOST_CHECK(marg5B[1] > marg5B[2]);
	BOOST_CHECK(marg5B[1] > marg5B[3]);
}
#endif

