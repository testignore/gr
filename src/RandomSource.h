
#include <vector>
#include <ctime>

#include <boost/random.hpp>

namespace Grante {

class RandomSource {
public:
	static boost::mt19937& GlobalRandomSampler();

	static boost::uint32_t GetGlobalRandomSeed();

	// Permute the given vector randomly
	static void ShuffleRandom(std::vector<unsigned int>& vec);

private:
	static boost::mt19937 random_sampler;
	static unsigned int initialized;

	// One global random number source for seeding
	static boost::mt19937 seed_random_sampler;
	static boost::uniform_int<boost::uint32_t> seed_randu;
	static boost::variate_generator<boost::mt19937,
		boost::uniform_int<boost::uint32_t> > seed_rand;
	static unsigned int seed_source_initialized;

	RandomSource();

	class shuffle_random {
	private:
		size_t N;
		boost::mt19937& gen;
		boost::uniform_int<boost::uint32_t> dest;
		boost::variate_generator<boost::mt19937,
			boost::uniform_int<boost::uint32_t> > rand;

	public:
		shuffle_random(size_t N, boost::mt19937& gen);
		std::ptrdiff_t operator()(std::ptrdiff_t arg);
	};
};

}

