
#include <iostream>
#include <ctime>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "FactorGraphObservation.h"
#include "TreeInference.h"
#include "ContrastiveDivergenceTraining.h"
#include "NormalPrior.h"

#include "Conditioning.h"
#include "AISInference.h"
#include "FactorConditioningTable.h"

#define BOOST_TEST_MODULE(ContrastiveDivergenceTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(CDSimpleDataIndependent)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	w.push_back(-0.2);
	w.push_back(1.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Get marginals
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_true = tinf.Marginals();

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 10000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by CD-10
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));
	}
	Grante::ContrastiveDivergenceTraining cd_train(&model,
		1 /*CD-k*/, 20 /*mini-batch size*/);
	cd_train.SetupTrainingData(training_data, inference_methods);
	cd_train.Train(1e-5, 20);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
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
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n)
		delete (training_data[n].second);
}

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

	// Learn model weights from population by CD
	std::vector<Grante::ContrastiveDivergenceTraining::partially_labeled_instance_type>
		training_data;
	std::vector<unsigned int> var_subset;
	var_subset.push_back(0);
	var_subset.push_back(1);
	var_subset.push_back(4);
	var_subset.push_back(5);

	// Produce 50%/50% samples of A/B types,
	unsigned int sample_count = 1000;
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
			Grante::ContrastiveDivergenceTraining::partially_labeled_instance_type(
				&fg, new Grante::FactorGraphPartialObservation(
					var_subset, cur_state)));
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

	Grante::ContrastiveDivergenceTraining cd_train(&model,
		2 /*CD-k*/, 10 /*mini-batch size*/);
	cd_train.SetupPartiallyObservedTrainingData(training_data);
	cd_train.Train(1e-5, 200);

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

