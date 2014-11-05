
include ../Makefile.config

MATLABROOT=/opt/matlab-r2009b
MEX=${MATLABROOT}/bin/mex
MEXINCLUDE=-I${MATLABROOT}/extern/include
MEXLIB=-L${MATLABROOT}/bin/glnxa64/ -lmat -lmex -lut -lmx -lmwservices
# Important: set LD_LIBRARY_PATH=/opt/matlab-r2009b/bin/glnxa64
#    before compiling

###
all:	debug_mex_grante_learn

debug_mex_grante_learn:	debug_mex_grante_learn.cpp
	$(CPP) -g -ggdb -o debug_mex_grante_learn $(MEXINCLUDE) -I.. \
		$(MEXLIB) $(BOOST_LIB) \
		debug_mex_grante_learn.cpp mex_grante_learn.cpp matlab_helpers.cpp \
		$(INCLUDE) -L.. -lgrante

