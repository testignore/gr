
include Makefile.config
 
OBJS=Factor.o FactorGraph.o FactorGraphModel.o FactorType.o \
	FactorGraphStructurizer.o DisjointSet.o LogSumExp.o TreeInference.o \
	GibbsSampler.o FunctionMinimization.o FunctionMinimizationProblem.o \
	Likelihood.o InferenceMethod.o ParameterEstimationMethod.o \
	MaximumLikelihood.o FactorGraphObservation.o Prior.o NormalPrior.o \
	LaplacePrior.o MaximumPseudolikelihood.o FactorGraphUtility.o \
	Pseudolikelihood.o StudentTPrior.o \
	StructuredLossFunction.o StructuredHammingLoss.o StructuredSVM.o \
	StochasticFunctionMinimizationProblem.o StochasticFunctionMinimization.o \
	StructuredPerceptron.o RandomFactorGraphGenerator.o \
	VAcyclicDecomposition.o SubFactorGraph.o \
	FactorConditioningTable.o ConditionedFactorType.o Conditioning.o \
	MaximumCompositeLikelihood.o TreeCoverDecomposition.o \
	LinearProgrammingMAPInference.o FactorDataSource.o \
	FactorGraphPartialObservation.o BeliefPropagation.o \
	RBFNetworkRegression.o RBFNetwork.o NonlinearRBFFactorType.o \
	GibbsInference.o StructuredMeanFieldInference.o \
	ExpectationMaximization.o RandomSource.o \
	AISInference.o SimulatedAnnealingInference.o \
	MaximumCrissCrossLikelihood.o DiffusionInference.o \
	BruteForceExactInference.o LinearFactorType.o \
	NaivePiecewiseTraining.o ContrastiveDivergence.o \
	ContrastiveDivergenceTraining.o HyperbolicPrior.o \
	CompositeMinimization.o CompositeMinimizationProblem.o \
	SwendsenWangSampler.o SwendsenWangInference.o \
	ParallelTemperingInference.o SAMCInference.o TestModels.o \
	NaiveMeanFieldInference.o MaximumTreePseudoLikelihood.o \
	DisjointSetBT.o MultichainGibbsInference.o

###
all:	libgrante

clean:
	rm -f *.o
	rm -f FactorGraph_test TreeInference_test FunctionMinimization_test \
		Likelihood_test StructuredSVM_test VAcyclicDecomposition_test \
		SubFactorGraph_test Conditioning_test CompositeLikelihood_test \
		TreeCoverDecomposition_test LPMAP_test BeliefPropagation_test \
		RBFNetworkRegression_test NonlinearFactor_test Meanfield_test \
		EM_test RegressionTree_test RegressionTreeFactorType_test \
		AIS_test MSD_test Eulerian_test LinearFactorType_test \
		CD_test Prior_test \
		CompositeMinimization_test SW_test SAMC_test DisjointSetBT_test
	rm -f *.stackdump
	rm -f testdata.out/graph1.fg testdata.out/model1.fgm
	rm -f *.exe

###
### Production build targets
###
libgrante: $(OBJS)
	ar rc libgrante.a $(OBJS)
	ranlib libgrante.a

###
### Testcases
###

check:	libgrante FactorGraph_test TreeInference_test \
	FunctionMinimization_test Likelihood_test StructuredSVM_test \
	VAcyclicDecomposition_test SubFactorGraph_test Conditioning_test \
	CompositeLikelihood_test TreeCoverDecomposition_test LPMAP_test \
	BeliefPropagation_test RBFNetworkRegression_test NonlinearFactor_test \
	Meanfield_test EM_test AIS_test MSD_test Eulerian_test \
	LinearFactorType_test CD_test \
	Prior_test CompositeMinimization_test SW_test SAMC_test DisjointSetBT_test
#
	./FactorGraph_test && ./TreeInference_test \
		&& ./FunctionMinimization_test && ./Likelihood_test \
		&& ./StructuredSVM_test && ./VAcyclicDecomposition_test \
		&& ./SubFactorGraph_test && ./Conditioning_test \
		&& ./CompositeLikelihood_test && ./TreeCoverDecomposition_test \
		&& ./LPMAP_test && ./BeliefPropagation_test \
		&& ./RBFNetworkRegression_test && ./NonlinearFactor_test \
		&& ./Meanfield_test && ./EM_test \
		&& ./AIS_test && ./MSD_test && ./Eulerian_test \
		&& ./LinearFactorType_test \
		&& ./CD_test && ./Prior_test && ./CompositeMinimization_test \
		&& ./SW_test && ./SAMC_test && ./DisjointSetBT_test \
		&& echo "ALL TESTS PASSED SUCCESSFULLY"

FactorGraph_test:	FactorGraph_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o FactorGraph_test FactorGraph_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

TreeInference_test:	TreeInference_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o TreeInference_test \
		TreeInference_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

FunctionMinimization_test:	FunctionMinimization_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o FunctionMinimization_test \
		FunctionMinimization_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

CompositeMinimization_test:	CompositeMinimization_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o CompositeMinimization_test \
		CompositeMinimization_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

Likelihood_test:	Likelihood_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o Likelihood_test \
		Likelihood_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

StructuredSVM_test:	StructuredSVM_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o StructuredSVM_test \
		StructuredSVM_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

VAcyclicDecomposition_test:	VAcyclicDecomposition_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) \
		-o VAcyclicDecomposition_test VAcyclicDecomposition_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

SubFactorGraph_test:	SubFactorGraph_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o SubFactorGraph_test SubFactorGraph_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

Conditioning_test:	Conditioning_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o Conditioning_test Conditioning_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

CompositeLikelihood_test:	CompositeLikelihood_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o CompositeLikelihood_test \
		CompositeLikelihood_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

TreeCoverDecomposition_test:	TreeCoverDecomposition_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o TreeCoverDecomposition_test \
		TreeCoverDecomposition_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

LPMAP_test:	LPMAP_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o LPMAP_test \
		LPMAP_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

BeliefPropagation_test:	BeliefPropagation_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o BeliefPropagation_test \
		BeliefPropagation_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

RBFNetworkRegression_test:	RBFNetworkRegression_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o RBFNetworkRegression_test \
		RBFNetworkRegression_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

NonlinearFactor_test:	NonlinearFactor_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o NonlinearFactor_test \
		NonlinearFactor_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

Meanfield_test:	Meanfield_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o Meanfield_test \
		Meanfield_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

EM_test:	EM_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o EM_test \
		EM_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

AIS_test:	AIS_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o AIS_test \
		AIS_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

MSD_test:	MSD_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o MSD_test \
		MSD_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

Eulerian_test:	Eulerian_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o Eulerian_test \
		Eulerian_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

LinearFactorType_test:	LinearFactorType_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o LinearFactorType_test LinearFactorType_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

CD_test:	CD_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o CD_test \
		CD_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

Prior_test:	Prior_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o Prior_test \
		Prior_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

SW_test:	SW_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o SW_test \
		SW_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

SAMC_test:	SAMC_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o SAMC_test \
		SAMC_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

MTPLE_test:	MTPLE_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o MTPLE_test \
		MTPLE_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB) $(PERFTOOLS_LIB)

DisjointSetBT_test:	DisjointSetBT_test.o libgrante
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o DisjointSetBT_test DisjointSetBT_test.o \
		libgrante.a $(BOOST_LIB) $(BOOST_TEST_LIB)

