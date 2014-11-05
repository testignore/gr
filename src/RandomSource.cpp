
#include <algorithm>
#include <limits>

#include "RandomSource.h"

namespace Grante {

// Global static variables
unsigned int RandomSource::initialized = 0;
boost::mt19937 RandomSource::random_sampler;

unsigned int RandomSource::seed_source_initialized = 0;
boost::uniform_int<boost::uint32_t> RandomSource::seed_randu;
boost::mt19937 RandomSource::seed_random_sampler;
boost::variate_generator<boost::mt19937,
	boost::uniform_int<boost::uint32_t> > RandomSource::seed_rand(
	RandomSource::seed_random_sampler, RandomSource::seed_randu);

boost::mt19937& RandomSource::GlobalRandomSampler() {
	if (initialized == 0) {
		initialized = static_cast<unsigned int>(std::time(0));
		random_sampler.seed(initialized);
	}
	// TODO: check how GlobalRandomSampler() is used in the project
	return (random_sampler);
}

boost::uint32_t RandomSource::GetGlobalRandomSeed() {
	boost::uint32_t seed;
	#pragma omp critical
	{
		if (seed_source_initialized == 0) {
			seed_source_initialized =
				static_cast<unsigned int>(std::time(0)) + 1903;
			seed_random_sampler.seed(seed_source_initialized);

			seed_randu = boost::uniform_int<boost::uint32_t>(0,
				std::numeric_limits<boost::uint32_t>::max());
			seed_rand = boost::variate_generator<boost::mt19937,
				boost::uniform_int<boost::uint32_t> >(seed_random_sampler,
				seed_randu);
		}
		seed = seed_rand();
	}
	return (seed);
}

void RandomSource::ShuffleRandom(std::vector<unsigned int>& vec) {
	boost::mt19937 sr_random_sampler;
	sr_random_sampler.seed(GetGlobalRandomSeed());
	shuffle_random sr(vec.size(), sr_random_sampler);
	std::random_shuffle(vec.begin(), vec.end(), sr);
}

RandomSource::RandomSource() {
}

// shuffle_random class
RandomSource::shuffle_random::shuffle_random(size_t N, boost::mt19937& gen)
	: N(N), gen(gen), dest(0, static_cast<boost::uint32_t>(N-1)),
	rand(gen, dest) {
}

std::ptrdiff_t RandomSource::shuffle_random::operator()(std::ptrdiff_t arg) {
	std::ptrdiff_t rv = static_cast<std::ptrdiff_t>(this->rand());
	// When using signed/unsigned combinations in boost::random, things can go
	// wrong, so better check.
	assert(rv >= 0);
	assert(rv < static_cast<std::ptrdiff_t>(N));
	return (rv);
}

}

