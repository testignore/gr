
#include <vector>
#include <iostream>

#include "RegressionTreeFactorType.h"
#include "RegressionTree.h"
#include "BeliefPropagation.h"
#include "Factor.h"

#define BOOST_TEST_MODULE(RegressionTreeFactorTypeTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(MarginalComputation)
{
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	Grante::RegressionTreeFactorType rtftype("F1", card, 0);

	// Build regression tree
	Grante::RegressionTree* tree = new Grante::RegressionTree();
	//         0(d[0] == 0)
	//          |       |
	//  1(d[1] == 0)    4
	//    /      |
	//   2       3
	Grante::RegressionTree::RegressionTreeNode node0(0u, 0u, 0u, 1u, 4u);
	tree->AddTreeNode(node0, 0.0);
	Grante::RegressionTree::RegressionTreeNode node1(1u, 1u, 0u, 2u, 3u);
	tree->AddTreeNode(node1, 0.0);
	Grante::RegressionTree::RegressionTreeNode node2(2u);
	tree->AddTreeNode(node2, 0.0);
	Grante::RegressionTree::RegressionTreeNode node3(3u);
	tree->AddTreeNode(node3, 0.0);
	Grante::RegressionTree::RegressionTreeNode node4(4u);
	tree->AddTreeNode(node4, 0.0);

	rtftype.SetRegressionTree(tree);
	std::vector<double>& w = rtftype.Weights();
	BOOST_CHECK(w.size() == 3);
	w[0] = 0.25;	// (0,0)
	w[1] = 0.3;	// (0,1)
	w[2] = 0.5;	// (1,0) and (1,1)

	std::vector<unsigned int> var_index;
	var_index.push_back(0);
	var_index.push_back(1);
	std::vector<double> data;
	Grante::Factor fac(&rtftype, var_index, data);
	fac.ForwardMap();

	std::vector<unsigned int> var_state(2);
	var_state[0] = 0;
	var_state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(w[0], fac.EvaluateEnergy(var_state), 1.0e-5);
	var_state[0] = 0;
	var_state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(w[1], fac.EvaluateEnergy(var_state), 1.0e-5);
	var_state[0] = 1;
	var_state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(w[2], fac.EvaluateEnergy(var_state), 1.0e-5);
	var_state[0] = 1;
	var_state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(w[2], fac.EvaluateEnergy(var_state), 1.0e-5);

	// Compute 'marginal'
	std::vector<double> marginal(3, 0.0);
	std::vector<unsigned int> msglist_for_factor_cur;
	msglist_for_factor_cur.push_back(0);
	msglist_for_factor_cur.push_back(1);
	std::vector<std::vector<double> > msg_for_factor;
	std::vector<double> empty_message(2, 0.0);
	msg_for_factor.push_back(empty_message);
	msg_for_factor.push_back(empty_message);
	double max_change = rtftype.ComputeBPMarginal(&fac,
		msglist_for_factor_cur, msg_for_factor, marginal, false);

	BOOST_CHECK_CLOSE_ABS(0.28500, marginal[0], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.27110, marginal[1], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.44391, marginal[2], 1.0e-5);
	BOOST_CHECK_GT(max_change, 0.44);

	// Compute BP message
	std::vector<unsigned int> msg_for_factor_srcvar;
	msg_for_factor_srcvar.push_back(0);
	msg_for_factor_srcvar.push_back(1);
	msg_for_factor[0][0] = 0.1;
	msg_for_factor[0][1] = 0.2;
	msg_for_factor[1][0] = 0.3;
	msg_for_factor[1][1] = 0.4;

	// r_{F->Y0}
	std::vector<double> msg0(2);
	rtftype.ComputeBPMessage(&fac, /*vi*/ 0, /*fvi_to*/ 0,
		msglist_for_factor_cur, msg_for_factor, msg_for_factor_srcvar,
		msg0, false);
	BOOST_CHECK_CLOSE_ABS(0.76846, msg0[0], 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(0.54440, msg0[1], 1.0e-4);

	// r_{F->Y1}
	std::vector<double> msg1(2);
	rtftype.ComputeBPMessage(&fac, /*vi*/ 1, /*fvi_to*/ 1,
		msglist_for_factor_cur, msg_for_factor, msg_for_factor_srcvar,
		msg1, false);
	BOOST_CHECK_CLOSE_ABS(0.47096, msg1[0], 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(0.44440, msg1[1], 1.0e-4);
}

BOOST_AUTO_TEST_CASE(BPTest)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	Grante::RegressionTreeFactorType* rtftype =
		new Grante::RegressionTreeFactorType("F1", card, 0);
	model.AddFactorType(rtftype);

	// Build regression tree
	Grante::RegressionTree* tree = new Grante::RegressionTree();
	//         0(d[0] == 0)
	//          |       |
	//  1(d[1] == 0)    4
	//    /      |
	//   2       3
	Grante::RegressionTree::RegressionTreeNode node0(0u, 0u, 0u, 1u, 4u);
	tree->AddTreeNode(node0, 0.0);
	Grante::RegressionTree::RegressionTreeNode node1(1u, 1u, 0u, 2u, 3u);
	tree->AddTreeNode(node1, 0.0);
	Grante::RegressionTree::RegressionTreeNode node2(2u);
	tree->AddTreeNode(node2, 0.0);
	Grante::RegressionTree::RegressionTreeNode node3(3u);
	tree->AddTreeNode(node3, 0.0);
	Grante::RegressionTree::RegressionTreeNode node4(4u);
	tree->AddTreeNode(node4, 0.0);

	rtftype->SetRegressionTree(tree);
	std::vector<double>& w = rtftype->Weights();
	BOOST_CHECK(w.size() == 3);
	w[0] = 0.25;	// (0,0)
	w[1] = 0.3;	// (0,1)
	w[2] = 0.5;	// (1,0) and (1,1)

	// Unary zero-energy factor
	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.0);
	w1.push_back(0.0);
	Grante::FactorType* factortype1a = new Grante::FactorType("unaryzero", card1, w1);
	model.AddFactorType(factortype1a);


	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	std::vector<unsigned int> var_index;
	var_index.push_back(0);
	var_index.push_back(1);
	std::vector<double> data;
	Grante::Factor* fac = new Grante::Factor(rtftype, var_index, data);
	fg.AddFactor(fac);

	var_index.clear();
	var_index.push_back(0);
	fac = new Grante::Factor(factortype1a, var_index, data);
	fg.AddFactor(fac);
	var_index[0] = 1;
	fac = new Grante::Factor(factortype1a, var_index, data);
	fg.AddFactor(fac);

	// Compute the forward map
	fg.ForwardMap();

	// Test inference results
	Grante::BeliefPropagation bpinf(&fg);
	bpinf.PerformInference();

	double log_z = bpinf.LogPartitionFunction();
	std::cout << "log_z " << log_z << std::endl;

	const std::vector<double>& marginal = bpinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.28500, marginal[0], 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(0.27110, marginal[1], 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(0.44391, marginal[2], 1.0e-4);
}

