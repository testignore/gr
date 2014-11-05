
#include <vector>
#include <ctime>

#include <boost/random.hpp>
#include <boost/timer.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "FactorGraphStructurizer.h"
#include "VAcyclicDecomposition.h"

#define BOOST_TEST_MODULE(VAcyclicDecompositionTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(SimpleCycle)
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

	// Create a factor graph from the model: 4 binary variables on a cycle
	std::vector<unsigned int> vc(4, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);

	// 0-1
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	// 1-2
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// 2-3
	var_index[0] = 2;
	var_index[1] = 3;
	Grante::Factor* fac3 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac3);

	// 3-0
	var_index[0] = 3;
	var_index[1] = 0;
	Grante::Factor* fac4 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac4);

	// Test v-acyclic decomposition
	Grante::VAcyclicDecomposition vac(&fg);
	std::vector<bool> factor_is_removed;
	std::vector<double> factor_weight(4);
	factor_weight[0] = 1.0;
	factor_weight[1] = 0.5;
	factor_weight[2] = 1.0;
	factor_weight[3] = 0.3;
#if 0
	double obj_sa = vac.ComputeDecompositionSA(factor_weight, factor_is_removed);

	// Both the set-packing and simulated annealing solvers can miss the best
	// configuration, even for this small example.
	BOOST_CHECK(factor_is_removed[0] == false);
	BOOST_CHECK(factor_is_removed[1] == true);
	BOOST_CHECK(factor_is_removed[2] == false);
	BOOST_CHECK(factor_is_removed[3] == true);
	BOOST_CHECK_CLOSE_ABS(2.0, obj_sa, 1.0e-5);
#endif
#if 0
	double obj_sp = vac.ComputeDecompositionSP(factor_weight, factor_is_removed);

	BOOST_CHECK(factor_is_removed[0] == false);
	BOOST_CHECK(factor_is_removed[1] == true);
	BOOST_CHECK(factor_is_removed[2] == false);
	BOOST_CHECK(factor_is_removed[3] == true);
	BOOST_CHECK_CLOSE_ABS(2.0, obj_sp, 1.0e-5);
#endif
	double obj_exact = vac.ComputeDecompositionExact(factor_weight,
		factor_is_removed);
	BOOST_CHECK(factor_is_removed[0] == false);
	BOOST_CHECK(factor_is_removed[1] == true);
	BOOST_CHECK(factor_is_removed[2] == false);
	BOOST_CHECK(factor_is_removed[3] == true);
	BOOST_CHECK_CLOSE_ABS(2.0, obj_exact, 1.0e-5);
}

BOOST_AUTO_TEST_CASE(SimpleGrid)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 0.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N-by-N grid-structured model
	unsigned int N = 4;

	// Create a factor graph from the model (binary variables)
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

	Grante::VAcyclicDecomposition vac(&fg);
	std::vector<bool> factor_is_removed;
	unsigned int factor_count = fg.Factors().size();
	std::vector<double> factor_weight(factor_count, 1.0);

	boost::timer sp_timer;
	double obj_sp = vac.ComputeDecompositionSP(factor_weight, factor_is_removed);
	double time_sp = sp_timer.elapsed();

	boost::timer greedy_timer;
	double obj_greedy = vac.ComputeDecompositionGreedy(factor_weight, factor_is_removed);
	double time_greedy = greedy_timer.elapsed();

	std::cout << "Grid, objective SP " << obj_sp << ", Greedy "
		<< obj_greedy << std::endl;
	std::cout << "Grid, timing, SP " << time_sp
		<< ", Greedy " << time_greedy << std::endl;

	boost::timer exact_timer;
	double obj_exact = vac.ComputeDecompositionExact(factor_weight,
		factor_is_removed);
	double time_exact = exact_timer.elapsed();
	std::cout << "Exact " << obj_exact << ", time " << time_exact
		<< std::endl;
	BOOST_CHECK(std::fabs(obj_exact - 14.0) <= 1.0e-5); 
}

BOOST_AUTO_TEST_CASE(SimpleGridRandomCoeff)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 0.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N-by-N grid-structured model
	unsigned int N = 4;

	// Create a factor graph from the model (binary variables)
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

	Grante::VAcyclicDecomposition vac(&fg);
	std::vector<bool> factor_is_removed;
	unsigned int factor_count = fg.Factors().size();

	// Randomly set the factor weights
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	std::vector<double> factor_weight(factor_count, 0.0);
	for (unsigned int fi = 0; fi < factor_count; ++fi)
		factor_weight[fi] = randu();

	boost::timer sp_timer;
	double obj_sp = vac.ComputeDecompositionSP(factor_weight, factor_is_removed);
	double time_sp = sp_timer.elapsed();

	boost::timer greedy_timer;
	double obj_greedy = vac.ComputeDecompositionGreedy(factor_weight, factor_is_removed);
	double time_greedy = greedy_timer.elapsed();

	std::cout << "Grid, objective SP " << obj_sp << ", Greedy "
		<< obj_greedy << std::endl;
	std::cout << "Grid, timing, SP " << time_sp
		<< ", Greedy " << time_greedy << std::endl;

	boost::timer exact_timer;
	double obj_exact = vac.ComputeDecompositionExact(factor_weight,
		factor_is_removed);
	double time_exact = exact_timer.elapsed();
	std::cout << "Exact " << obj_exact << ", time " << time_exact
		<< std::endl;
	BOOST_CHECK(obj_exact >= obj_sp); 
	BOOST_CHECK(obj_exact >= obj_greedy); 
}

BOOST_AUTO_TEST_CASE(RandomLarge)
{
	Grante::FactorGraphModel model;
	unsigned int var_count = 15;

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

	// Create a factor graph from the model: 4 binary variables on a cycle
	std::vector<unsigned int> vc(var_count, 2);
	Grante::FactorGraph fg(&model, vc);

	// Random number generators: var index
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_int<unsigned int> rdestd(0, var_count-1);
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_vi(rgen, rdestd);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);

	unsigned int factor_count = 5 * var_count;
	for (unsigned int fi = 0; fi < factor_count; ++fi) {
		var_index[0] = rand_vi();
		do {
			var_index[1] = rand_vi();
		} while (var_index[1] == var_index[0]);
		Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
		fg.AddFactor(fac);
	}

	// Decompose
	Grante::VAcyclicDecomposition vac(&fg);
	std::vector<bool> factor_is_removed;

	// Randomly set the factor weights
	boost::mt19937 rgen2(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen2, rdestu);

	std::vector<double> factor_weight(factor_count, 0.0);
	for (unsigned int fi = 0; fi < factor_count; ++fi)
		factor_weight[fi] = randu();

	boost::timer sa_timer;
	double obj = vac.ComputeDecompositionSA(factor_weight, factor_is_removed);
	double time_sa = sa_timer.elapsed();

	boost::timer sp_timer;
	double obj_sp = vac.ComputeDecompositionSP(factor_weight, factor_is_removed);
	double time_sp = sp_timer.elapsed();

	boost::timer greedy_timer;
	double obj_greedy = vac.ComputeDecompositionGreedy(factor_weight,
		factor_is_removed);
	double time_greedy = greedy_timer.elapsed();

	std::cout << "SA " << obj << ", SP " << obj_sp << ", Greedy "
		<< obj_greedy << std::endl;
	std::cout << "Timing, SA " << time_sa << ", SP " << time_sp
		<< ", Greedy " << time_greedy << std::endl;

	// Exact
	boost::timer exact_timer;
	double obj_exact = vac.ComputeDecompositionExact(factor_weight,
		factor_is_removed);
	double time_exact = exact_timer.elapsed();
	std::cout << "Exact " << obj_exact << ", time " << time_exact
		<< std::endl;

	// Create a new factor graph by using only the retained factors
	Grante::FactorGraph fg2(&model, vc);
	for (unsigned int fi = 0; fi < factor_is_removed.size(); ++fi) {
		if (factor_is_removed[fi])
			continue;
		fg2.AddFactor(new Grante::Factor(*fg.Factors()[fi]));
	}
	BOOST_CHECK(Grante::FactorGraphStructurizer::IsForestStructured(&fg2));

	// Refined test for v-acyclicity: adding each removed factor individually
	// retains forest-structure
	for (unsigned int fi = 0; fi < factor_is_removed.size(); ++fi) {
		if (factor_is_removed[fi] == false)
			continue;

		fg2.AddFactor(new Grante::Factor(*fg.Factors()[fi]));
		BOOST_CHECK(Grante::FactorGraphStructurizer::IsForestStructured(&fg2));

		// Remove factor, the ugly way
		std::vector<Grante::Factor*>& factors =
			const_cast<std::vector<Grante::Factor*>&>(fg2.Factors());
		delete (*factors.rbegin());
		factors.resize(factors.size() - 1);
	}
}

BOOST_AUTO_TEST_CASE(SetPackingSimple)
{
	std::vector<std::tr1::unordered_set<unsigned int> > S;
	std::vector<double> S_weights;
	std::vector<bool> S_is_selected;
	std::tr1::unordered_set<unsigned int> E;

	// 0-1
	E.clear();
	E.insert(0);
	E.insert(1);
	S.push_back(E);
	S_weights.push_back(3);

	// 1-2
	E.clear();
	E.insert(1);
	E.insert(2);
	S.push_back(E);
	S_weights.push_back(2);

	// 2-3
	E.clear();
	E.insert(2);
	E.insert(3);
	S.push_back(E);
	S_weights.push_back(7);

	// 3-4
	E.clear();
	E.insert(3);
	E.insert(4);
	S.push_back(E);
	S_weights.push_back(9);

	// 0-4
	E.clear();
	E.insert(0);
	E.insert(4);
	S.push_back(E);
	S_weights.push_back(8);

	// 2-5
	E.clear();
	E.insert(2);
	E.insert(5);
	S.push_back(E);
	S_weights.push_back(1);

	// 3-5
	E.clear();
	E.insert(3);
	E.insert(5);
	S.push_back(E);
	S_weights.push_back(4);

	// 4-5
	E.clear();
	E.insert(4);
	E.insert(5);
	S.push_back(E);
	S_weights.push_back(4);

	double obj = Grante::VAcyclicDecomposition::ComputeSetPacking(
		S, S_weights, S_is_selected);
	BOOST_CHECK(std::fabs(obj - 15.0) <= 1.0e-5);
	BOOST_CHECK(S_is_selected[0] == false);
	BOOST_CHECK(S_is_selected[1] == false);
	BOOST_CHECK(S_is_selected[2] == true);
	BOOST_CHECK(S_is_selected[3] == false);
	BOOST_CHECK(S_is_selected[4] == true);
	BOOST_CHECK(S_is_selected[5] == false);
	BOOST_CHECK(S_is_selected[6] == false);
	BOOST_CHECK(S_is_selected[7] == false);
}

BOOST_AUTO_TEST_CASE(RandomSetPacking)
{
	unsigned int var_count = 1000;
	double alpha = 0.75;

	// Set packing problem parameters
	std::vector<std::tr1::unordered_set<unsigned int> > S;
	std::vector<double> S_weights;
	std::vector<bool> S_is_selected;
	std::tr1::unordered_set<unsigned int> E;

	// Random number generators: var index
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_int<unsigned int> rdestd(0, var_count-1);
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_vi(rgen, rdestd);

	// Create random sets
	unsigned int set_count = static_cast<unsigned int>(alpha * var_count);
	for (unsigned int si = 0; si < set_count; ++si) {
		E.clear();
		for (unsigned int sei = 0; sei < 3; ++sei)
			E.insert(rand_vi());
		S.push_back(E);
		S_weights.push_back(1.0);
	}

	Grante::VAcyclicDecomposition::ComputeSetPacking(
		S, S_weights, S_is_selected);
}

BOOST_AUTO_TEST_CASE(GridSetPacking)
{
	unsigned int xdim = 4;
	unsigned int ydim = 3;

	// Set packing problem parameters
	std::vector<std::tr1::unordered_set<unsigned int> > S;
	std::vector<double> S_weights;
	std::vector<bool> S_is_selected;
	std::tr1::unordered_set<unsigned int> E;

	for (unsigned int y = 0; y < ydim; ++y) {
		for (unsigned int x = 0; x < xdim; ++x) {
			if (y >= 1) {
				E.clear();
				E.insert((y-1)*xdim+x);
				E.insert(y*xdim+x);
				S.push_back(E);
				S_weights.push_back(1.0);
			}
			if (x >= 1) {
				E.clear();
				E.insert(y*xdim+x-1);
				E.insert(y*xdim+x);
				S.push_back(E);
				S_weights.push_back(1.0);
			}
		}
	}
	double obj = Grante::VAcyclicDecomposition::ComputeSetPacking(
		S, S_weights, S_is_selected);

	// The true optimum solution has objective 6, but LR achieves only 5
	BOOST_CHECK(obj >= (5.0 - 1.0e-8));
}

BOOST_AUTO_TEST_CASE(SimpleAcyclic)
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

	// Create a factor graph from the model: 3 binary variables, acyclic
	std::vector<unsigned int> vc(3, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);

	// 0-1
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	// 1-2
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// Test v-acyclic decomposition
	Grante::VAcyclicDecomposition vac(&fg);
	std::vector<bool> factor_is_removed;
	std::vector<double> factor_weight(2, 1.0);
	vac.ComputeDecompositionSP(factor_weight, factor_is_removed);
	BOOST_CHECK(factor_is_removed[0] == false);
	BOOST_CHECK(factor_is_removed[1] == false);
}

