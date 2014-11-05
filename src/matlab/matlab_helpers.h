
#ifndef GRANTE_MATLAB_HELPERS_H
#define GRANTE_MATLAB_HELPERS_H

#include <vector>
#include <string>

#include <mex.h>

#include "FactorGraphModel.h"
#include "FactorGraph.h"

void MatlabCPPInitialize(bool verbose = true);
void MatlabCPPExit();
bool HasOption(const mxArray* opt_s, const std::string& opt_name);
const mxArray* GetOption(const mxArray* opt_s, const std::string& opt_name);
double GetScalarOption(const mxArray* opt_s, const std::string& opt_name);
std::string GetMatlabString(const mxArray* m_str);
double GetScalarDefaultOption(const mxArray* opt_s,
	const std::string& opt_name, double default_value);
unsigned int GetIntegerDefaultOption(const mxArray* opt_s,
	const std::string& opt_name, double default_value);
std::string GetStringDefaultOption(const mxArray* opt_s,
	const std::string& opt_name, const std::string& default_value);
void GetMatlabVector(const mxArray* vin, std::vector<unsigned int>& vout);
void GetMatlabVector(const mxArray* vin, std::vector<double>& vout);
void GetPartialMatlabVector(const mxArray* vin,
	std::vector<unsigned int>& var_subset,
	std::vector<unsigned int>& var_state);

bool matlab_parse_factorgraphmodel(const mxArray* par,
	Grante::FactorGraphModel& model);

Grante::FactorGraph* matlab_parse_factorgraphs(
	Grante::FactorGraphModel& model, const mxArray* fgs,
	unsigned int fgi);

bool matlab_parse_factorgraphs(Grante::FactorGraphModel& model,
	const mxArray* fgs, std::vector<Grante::FactorGraph*>& FG);

#endif

