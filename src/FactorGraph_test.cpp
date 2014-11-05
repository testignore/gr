
#include <vector>
#include <iostream>
#include <tr1/unordered_set>

#include "FactorGraph.h"
#include "FactorType.h"
#include "Factor.h"
#include "FactorGraphModel.h"
#include "FactorGraphStructurizer.h"
#include "RandomFactorGraphGenerator.h"

#define BOOST_TEST_MODULE(FactorGraphTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(Simple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
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
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
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
	var_index[0] = 0;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(pt, var_index, data));

	// Compute the forward map
	fg.ForwardMap();

	// Compute the backward map for some marginal vector
	std::vector<double> marg(4);
	marg[0] = 0.25;
	marg[1] = 0.4;
	marg[2] = 0.1;
	marg[3] = 0.25;
	std::vector<double> pargrad(4, 0.0);
	fac1->BackwardMap(marg, pargrad);
	for (unsigned int pi = 0; pi < pargrad.size(); ++pi) {
		BOOST_CHECK_CLOSE_ABS(marg[pi], pargrad[pi], 1.0e-7);
	}
}

BOOST_AUTO_TEST_CASE(Serialization)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
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

	model.Save("testdata.out/model1.fgm");

	Grante::FactorGraphModel* model_l =
		Grante::FactorGraphModel::Load("testdata.out/model1.fgm");
	BOOST_CHECK(model_l->FactorTypes().size() == 1);
	BOOST_CHECK(model_l->FindFactorType("pairwise") != 0);
	BOOST_CHECK(model_l->FindFactorType("pairwise")->WeightDimension() == 4);
	BOOST_CHECK_CLOSE_ABS(model_l->FindFactorType("pairwise")->Weights()[0], w[0], 1e-7);
	BOOST_CHECK_CLOSE_ABS(model_l->FindFactorType("pairwise")->Weights()[1], w[1], 1e-7);
	BOOST_CHECK_CLOSE_ABS(model_l->FindFactorType("pairwise")->Weights()[2], w[2], 1e-7);
	BOOST_CHECK_CLOSE_ABS(model_l->FindFactorType("pairwise")->Weights()[3], w[3], 1e-7);
	delete (model_l);

	// Create a factor graph from the model
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
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
	var_index[0] = 0;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(pt, var_index, data));

	// Compute the forward map
	fg.ForwardMap();

	fg.Save("testdata.out/graph1.fg");
	Grante::FactorGraph* fg_l = Grante::FactorGraph::Load("testdata.out/graph1.fg");
	fg_l->ForwardMap();

	// Delete the model that was deserialized
	delete (const_cast<Grante::FactorGraphModel*>(fg_l->Model()));
	delete (fg_l);
}

BOOST_AUTO_TEST_CASE(SimpleEnergy)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);	// (0,0)
	w.push_back(0.3);	// (1,0)
	w.push_back(0.2);	// (0,1)
	w.push_back(0.0);	// (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.3;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
	model.AddFactorType(factortype1b);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);
	var_index1[0] = 1;
	Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
	fg.AddFactor(fac1b);

	// Compute the forward map
	fg.ForwardMap();

	std::vector<unsigned int> state(2);
	state[0] = 0;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(0.4, fg.EvaluateEnergy(state), 1.0e-7);
	state[0] = 0;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(0.9, fg.EvaluateEnergy(state), 1.0e-7);
	state[0] = 1;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1.0e-7);
	state[0] = 1;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1.0e-7);
}

BOOST_AUTO_TEST_CASE(SimpleW0)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	data.push_back(1.0);
	data.push_back(0.2);
	data.push_back(-0.2);
	data.push_back(1.0);
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);
	var_index[0] = 0;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(pt, var_index, data));

	// Compute the forward map
	fg.ForwardMap();

	// Compute energy of a given configuration
	std::vector<unsigned int> state(3);
	for (unsigned int s1 = 0; s1 <= 1; ++s1) {
		state[0] = s1;
		for (unsigned int s2 = 0; s2 <= 1; ++s2) {
			state[1] = s2;
			for (unsigned int s3 = 0; s3 <= 1; ++s3) {
				state[2] = s3;
				double energy = fg.EvaluateEnergy(state);
				BOOST_CHECK(energy > 0.0);
#if 0
				std::cout << "[" << s1 << "," << s2 << "," << s3 << "]: "
					<< energy << std::endl;
#endif
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(SimpleDataDependent)
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
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> var_index(2);
	data[0] = 0.1;
	data[1] = 0.2;
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	data[0] = 0.3;
	data[1] = 0.4;
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	data[0] = 0.5;
	data[1] = 0.6;
	var_index[0] = 0;
	var_index[1] = 2;
	Grante::Factor* fac3 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac3);

	// Compute the forward map
	fg.ForwardMap();

	// Compute the backward map for some marginal vector
	std::vector<double> marg(4);
	marg[0] = 0.25;
	marg[1] = 0.4;
	marg[2] = 0.1;
	marg[3] = 0.25;
	std::vector<double> pargrad(8, 0.0);
	const std::vector<Grante::Factor*>& factors = fg.Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// There is only one factor type, "pairwise"
		factors[fi]->BackwardMap(marg, pargrad);
	}

	BOOST_CHECK_CLOSE_ABS(0.225, pargrad[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.3, pargrad[1], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.36, pargrad[2], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.48, pargrad[3], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.09, pargrad[4], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.12, pargrad[5], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.225, pargrad[6], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.3, pargrad[7], 1.0e-6);
}

BOOST_AUTO_TEST_CASE(SimpleDataDependentSparse)
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
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> data_idx(2);
	std::vector<unsigned int> var_index(2);
	data[0] = 0.1;
	data[1] = 0.2;
	data_idx[0] = 0;
	data_idx[1] = 1;
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data, data_idx);
	fg.AddFactor(fac1);

	data[0] = 0.3;
	data[1] = 0.4;
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data, data_idx);
	fg.AddFactor(fac2);

	data[0] = 0.5;
	data[1] = 0.6;
	var_index[0] = 0;
	var_index[1] = 2;
	Grante::Factor* fac3 = new Grante::Factor(pt, var_index, data, data_idx);
	fg.AddFactor(fac3);

	// Compute the forward map
	fg.ForwardMap();

	// Compute the backward map for some marginal vector
	std::vector<double> marg(4);
	marg[0] = 0.25;
	marg[1] = 0.4;
	marg[2] = 0.1;
	marg[3] = 0.25;
	std::vector<double> pargrad(8, 0.0);
	const std::vector<Grante::Factor*>& factors = fg.Factors();
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		// There is only one factor type, "pairwise"
		factors[fi]->BackwardMap(marg, pargrad);
	}

	BOOST_CHECK_CLOSE_ABS(0.225, pargrad[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.3, pargrad[1], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.36, pargrad[2], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.48, pargrad[3], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.09, pargrad[4], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.12, pargrad[5], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.225, pargrad[6], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.3, pargrad[7], 1.0e-6);
}

BOOST_AUTO_TEST_CASE(SimpleDataSource)
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
	Grante::FactorGraph fg(&model, vc);

	// Add data source
	std::vector<double> data(2);
	data[0] = 0.1;	// (0,0): 0.13, (1,0): 0.14, (0,1): 0.125, (1,1): 0.13
	data[1] = 0.2;
	Grante::FactorDataSource* ds = new Grante::FactorDataSource(data);
	fg.AddDataSource(ds);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, ds);
	fg.AddFactor(fac1);

	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, ds);
	fg.AddFactor(fac2);

	data[0] = 0.5;	// (0,0): 0.45, (1,0): 0.22, (0,1): 0.385, (1,1): 0.35
	data[1] = 0.6;
	var_index[0] = 0;
	var_index[1] = 2;
	Grante::Factor* fac3 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac3);

	// Compute the forward map
	fg.ForwardMap();

	std::vector<unsigned int> state(3);

	// (0,0,0): 0.13+0.13+0.45
	state[0] = 0;
	state[1] = 0;
	state[2] = 0;
	BOOST_CHECK_CLOSE_ABS(0.13+0.13+0.45, fg.EvaluateEnergy(state), 1.0e-7);

	// (0,0,1): 0.13+0.125+0.385
	state[0] = 0;
	state[1] = 0;
	state[2] = 1;
	BOOST_CHECK_CLOSE_ABS(0.13+0.125+0.385, fg.EvaluateEnergy(state), 1.0e-7);

	// (0,1,0): 0.125+0.14+0.45
	state[0] = 0;
	state[1] = 1;
	state[2] = 0;
	BOOST_CHECK_CLOSE_ABS(0.125+0.14+0.45, fg.EvaluateEnergy(state), 1.0e-7);
}

BOOST_AUTO_TEST_CASE(CycleDetection)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.0);
	w.push_back(0.0);
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
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 1;
	var_index[1] = 2;

	// 0-1-2
	BOOST_CHECK(Grante::FactorGraphStructurizer::IsForestStructured(&fg) == true);

	// 0-1-2-0
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);
	var_index[0] = 0;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(pt, var_index, data));
	BOOST_CHECK(Grante::FactorGraphStructurizer::IsForestStructured(&fg) == false);
}

BOOST_AUTO_TEST_CASE(TreeOrder)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
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
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
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
	var_index[0] = 3;
	var_index[1] = 1;
	Grante::Factor* fac3 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac3);

	// Compute tree order
	std::vector<Grante::FactorGraphStructurizer::OrderStep> order;
	std::tr1::unordered_set<unsigned int> tree_roots;
	Grante::FactorGraphStructurizer::ComputeTreeOrder(&fg, order, tree_roots);
	BOOST_CHECK(order.size() == 6);

	std::vector<Grante::FactorGraphStructurizer::OrderStep> order_eulerian;
	Grante::FactorGraphStructurizer::ComputeEulerianMessageTrail(&fg,
		order_eulerian);
	//Grante::FactorGraphStructurizer::PrintOrder(order_eulerian);
	BOOST_CHECK(order_eulerian.size() == 12);
}

BOOST_AUTO_TEST_CASE(TreeOrderHO)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
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

	// Add higher order factor type
	card.push_back(2);
	w.push_back(1.0);
	w.push_back(2.0);
	w.push_back(0.4);
	w.push_back(0.7);
	Grante::FactorType* factortype3 = new Grante::FactorType("tripple", card, w);
	model.AddFactorType(factortype3);

	// Create a factor graph from the model: 5 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	const Grante::FactorType* pt3 = model.FindFactorType("tripple");
	BOOST_REQUIRE(pt != 0);
	BOOST_REQUIRE(pt3 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 2;
	var_index[1] = 4;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	var_index[0] = 1;
	var_index[1] = 2;
	var_index.push_back(3);
	Grante::Factor* fac3 = new Grante::Factor(pt3, var_index, data);
	fg.AddFactor(fac3);

	// Compute tree order
	std::vector<Grante::FactorGraphStructurizer::OrderStep> order;
	std::tr1::unordered_set<unsigned int> tree_roots;
	Grante::FactorGraphStructurizer::ComputeTreeOrder(&fg, order, tree_roots);
	BOOST_CHECK(order.size() == 7);
}

BOOST_AUTO_TEST_CASE(DisconnectedUnary)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	Grante::FactorType* factortype = new Grante::FactorType("unary", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("unary");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(1);
	var_index[0] = 0;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 1;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);
	var_index[0] = 2;
	Grante::Factor* fac3 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac3);

	// Compute trail
	std::vector<Grante::FactorGraphStructurizer::OrderStep> order_eulerian;
	Grante::FactorGraphStructurizer::ComputeEulerianMessageTrail(&fg,
		order_eulerian);
	BOOST_CHECK(order_eulerian.size() == 6);
}

BOOST_AUTO_TEST_CASE(RandomTreeSimple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
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

	// Add higher order factor type
	card.push_back(3);
	w.push_back(1.0);
	w.push_back(2.0);
	w.push_back(0.4);
	w.push_back(0.7);
	w.push_back(1.1);
	w.push_back(-0.5);
	w.push_back(0.3);
	w.push_back(0.75);
	Grante::FactorType* factortype3 = new Grante::FactorType("tripple", card, w);
	model.AddFactorType(factortype3);

	Grante::RandomFactorGraphGenerator rfg_gen(&model);
	std::vector<double> ft_dist(2, 0.5);
	Grante::FactorGraph* fg = rfg_gen.GenerateTreeStructured(ft_dist, 10);

	delete (fg);
}

