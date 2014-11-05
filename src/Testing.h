#ifndef GRANTE_TESTING_H
#define GRANTE_TESTING_H

#include <cmath>

// Unfortunately, the boost testing libraries do not support an absolute
// deviation check for floating point numbers.  This macro emulates such test.
#define BOOST_CHECK_CLOSE_ABS(v_true,v_comp,tol) \
	BOOST_CHECK_MESSAGE(std::fabs((v_true)-(v_comp))<=(tol), \
		"true: " << (v_true) << ", is: " << (v_comp) << \
		", diff " << std::fabs((v_true)-(v_comp)) << " >= " << (tol));

#endif

