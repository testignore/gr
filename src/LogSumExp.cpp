
#include <algorithm>
#include <cmath>

#include "LogSumExp.h"

namespace Grante {

double LogSumExp::Compute(const std::vector<double>& x) {
	double xmax = *std::max_element(x.begin(), x.end());
	double lse = 0.0;
	for (std::vector<double>::const_iterator xi = x.begin();
		xi != x.end(); ++xi) {
		lse += std::exp(*xi - xmax);
	}
	return (xmax + std::log(lse));
}

double LogSumExp::ComputeNeg(const std::vector<double>& x) {
	double xmax = -*std::min_element(x.begin(), x.end());
	double lse = 0.0;
	for (std::vector<double>::const_iterator xi = x.begin();
		xi != x.end(); ++xi) {
		lse += std::exp(-*xi - xmax);
	}
	return (xmax + std::log(lse));
}

}

