
#include <ctime>
#include <boost/random.hpp>

#include "DisjointSetBT.h"

#define BOOST_TEST_MODULE(DisjointSetBTTest)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Simple)
{
	// Basic functionality test on a set of four elements
	Grante::DisjointSetBT dset(4);

	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 4u);
	dset.Union(dset.Find(0), dset.Find(1));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 3u);
	dset.Deunion();
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 4u);

	dset.Union(dset.Find(0), dset.Find(1));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 3u);
	dset.Union(dset.Find(2), dset.Find(3));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 2u);
	dset.Deunion();
	dset.Deunion();
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 4u);

	dset.Union(dset.Find(0), dset.Find(1));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 3u);
	dset.Union(dset.Find(0), dset.Find(2));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 2u);
	dset.Union(dset.Find(0), dset.Find(3));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 1u);
	dset.Deunion();
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 2u);
	dset.Union(dset.Find(0), dset.Find(3));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 1u);
	dset.Deunion();
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 2u);
	dset.Deunion();
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 3u);
	dset.Union(dset.Find(2), dset.Find(3));
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 2u);
	dset.Deunion();
	dset.Deunion();
	BOOST_CHECK_EQUAL(dset.NumberOfDisjointSets(), 4u);
}

BOOST_AUTO_TEST_CASE(Large)
{
	// Generate a large set and randomly merge.  Take a snapshot.
	// Starting from the snapshot make a number of union/deunion passes, then
	// compare the result with the snapshot.
	unsigned int N = 1000;
	Grante::DisjointSetBT dset(N);

	// random number generator for 0...N-1
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_int<unsigned int> rdestd(0, N-1);
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_vi(rgen, rdestd);

	// 1. Perform a set of random merges
	unsigned int merge_count = 400;
	unsigned int mi = 0;
	while (mi < merge_count) {
		unsigned int root1 = dset.Find(rand_vi());
		unsigned int root2 = dset.Find(rand_vi());
		if (root1 == root2)
			continue;

		dset.Union(root1, root2);
		mi += 1;
	}

	// 2. Take a snapshot
	Grante::DisjointSetBT dset_snap(dset);

	// 3. Stress-test dset by performing sequential union/deunions
	for (unsigned int stress_i = 0; stress_i <= 5000; ++stress_i) {
		// Union
		unsigned int smerge_count = 50;
		unsigned int smi = 0;
		while (smi < smerge_count) {
			unsigned int root1 = dset.Find(rand_vi());
			unsigned int root2 = dset.Find(rand_vi());
			if (root1 == root2)
				continue;

			dset.Union(root1, root2);
			smi += 1;
		}

		// Deunion, undo'ing everything
		for (smi = 0; smi < smerge_count; ++smi)
			dset.Deunion();
	}

	// 4. Compare all pairwise tests
	BOOST_CHECK_EQUAL(dset_snap.NumberOfDisjointSets(),
		dset.NumberOfDisjointSets());
	for (unsigned int n1 = 0; n1 < N; ++n1) {
		for (unsigned int n2 = 0; n2 < N; ++n2) {
			bool dset_snap_eq = (dset_snap.Find(n1) == dset_snap.Find(n2));
			bool dset_eq = (dset.Find(n1) == dset.Find(n2));

			BOOST_CHECK_EQUAL(dset_snap_eq, dset_eq);
		}
	}
}

