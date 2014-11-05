
#include <vector>
#include <iostream>
#include <ctime>
#include <cmath>

#include <boost/random.hpp>

#include "Prior.h"
#include "HyperbolicPrior.h"

#define BOOST_TEST_MODULE(PriorTest)
#include <boost/test/unit_test.hpp>

static const double pi = 3.14159265358979323846;

BOOST_AUTO_TEST_CASE(HyperbolicNormalizationTest)
{
	// Zero mean, unit std, univariate Normal distribution
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::normal_distribution<double> rdestn;
	boost::variate_generator<boost::mt19937,
		boost::normal_distribution<double> > randn(rgen, rdestn);

	unsigned int sample_count = 50000;
	double gsigma = 10.0;
	for (double delta = 0.25; delta <= 3.0; delta += 0.1) {
		std::cout << "Testing alpha = 0.25, delta = " << delta << std::endl;
		for (unsigned int dim = 1; dim <= 10; ++dim) {
			Grante::HyperbolicPrior hyp(dim, 0.25, delta);

			double vsum = 0.0;
			double vsum_logg = 0.0;
			double vsum_logp = 0.0;

			// Generate samples from multivariate Normal
			std::vector<double> X(dim);
			double X_logpgaussian;
			std::vector<double> dummy;
			for (unsigned int si = 0; si < sample_count; ++si) {
				double xcur_sqn = 0.0;
				for (unsigned int d = 0; d < dim; ++d) {
					X[d] = gsigma*randn();
					xcur_sqn += X[d] * X[d];
				}

				X_logpgaussian = -(0.5/(gsigma*gsigma))*xcur_sqn
					- 0.5*static_cast<double>(dim)*std::log(2.0*pi)
					- static_cast<double>(dim) * std::log(gsigma);

				// Evaluate prior -log p(w)
				double logp = -hyp.EvaluateNegLogP(X, dummy, 1.0);
				vsum += (logp - X_logpgaussian);// / std::abs(X_logpgaussian);
				vsum_logp += logp;
				vsum_logg += X_logpgaussian;
			}

			vsum /= static_cast<double>(sample_count);
			vsum_logp /= static_cast<double>(sample_count);
			vsum_logg /= static_cast<double>(sample_count);
			double div_stat = (vsum_logp - vsum_logg) /
				std::min(std::fabs(vsum_logp), std::fabs(vsum_logg));

#if 0
			std::cout << "   dim " << dim
				<< ", E_g[log p] = " << vsum_logp
				<< ", E_g[log g] = " << vsum_logg
				<< ", div " << div_stat
				<< std::endl;
#endif
			BOOST_CHECK(std::fabs(div_stat) <= 0.1);
		}
	}
}

