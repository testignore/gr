
#include <algorithm>
#include <iostream>
#include <cassert>
#include <map>
#include <cmath>

#include <mex.h>
#include <boost/math/special_functions/fpclassify.hpp>

#include "FactorDataSource.h"
#include "LinearFactorType.h"
#include "NonlinearRBFFactorType.h"
#include "matlab_helpers.h"
#include "matlab_out.h"

matlab_out mp;
std::streambuf* matlab_out::mp_backup = 0;

void MatlabCPPInitialize(bool verbose) {
	matlab_out::OutInitialize(&mp, verbose);
}

void MatlabCPPExit() {
	matlab_out::OutExit();
}

bool HasOption(const mxArray* opt_s, const std::string& opt_name) {
	if (opt_s == 0)
		return (false);

	if (mxIsStruct(opt_s) == false) {
		mexErrMsgTxt("Passed options is not a structure.\n");
		assert(0);
		return (false);
	}
	int fn = mxGetFieldNumber(opt_s, opt_name.c_str());
	return (fn >= 0);
}

const mxArray* GetOption(const mxArray* opt_s, const std::string& opt_name) {
	assert(HasOption(opt_s, opt_name));
	int fn = mxGetFieldNumber(opt_s, opt_name.c_str());
	assert(fn >= 0);

	const mxArray* res = mxGetFieldByNumber(opt_s, 0, fn);
	assert(res != 0);
	return (res);
}

double GetScalarOption(const mxArray* opt_s,
	const std::string& opt_name) {
	const mxArray* opt = GetOption(opt_s, opt_name);
	if (mxIsNumeric(opt) == false || mxGetM(opt) != 1 || mxGetN(opt) != 1) {
		mexErrMsgTxt("Invalid use of a scalar-valued option.\n");
		return (0);
	}
	return (mxGetScalar(opt));
}

double GetScalarDefaultOption(const mxArray* opt_s,
	const std::string& opt_name, double default_value) {
	if (HasOption(opt_s, opt_name) == false)
		return (default_value);

	return (GetScalarOption(opt_s, opt_name));
}

std::string GetStringDefaultOption(const mxArray* opt_s,
	const std::string& opt_name, const std::string& default_value) {
	if (HasOption(opt_s, opt_name) == false)
		return (default_value);

	return (GetMatlabString(GetOption(opt_s, opt_name)));
}

unsigned int GetIntegerDefaultOption(const mxArray* opt_s,
	const std::string& opt_name, double default_value) {
	return (static_cast<unsigned int>(GetScalarDefaultOption(opt_s,
		opt_name, default_value)));
}

std::string GetMatlabString(const mxArray* m_str) {
	assert(mxIsChar(m_str));
	int char_count = static_cast<int>(mxGetNumberOfElements(m_str) + 1);
	char* buf = static_cast<char*>(mxCalloc(char_count, sizeof(char)));
	mxGetString(m_str, buf, char_count);

	std::string res(buf);
	mxFree(buf);

	return (res);
}

void GetMatlabVector(const mxArray* vin,
	std::vector<unsigned int>& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
		return;
	}
	size_t vec_count = mxGetNumberOfElements(vin);
	vout.resize(vec_count);
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n)
		vout[n] = static_cast<unsigned int>(vec_p[n]);
}

void GetMatlabVector(const mxArray* vin,
	std::vector<double>& vout) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}
	size_t vec_count = mxGetNumberOfElements(vin);
	vout.resize(vec_count);
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n)
		vout[n] = vec_p[n];
}

void GetMatlabVector(const mxArray* f_data,
	std::vector<double>& data_elem,
	std::vector<unsigned int>& data_idx) {
	assert(mxIsSparse(f_data) == true);
	assert(mxGetN(f_data) == 1);
	mwSize nzmax = mxGetNzmax(f_data);
	data_elem.resize(nzmax);
	data_idx.resize(nzmax);

	unsigned int nnz = 0;
	mwIndex* ir = mxGetIr(f_data);
	mwIndex* jc = mxGetJc(f_data);
	double* f_data_p = mxGetPr(f_data);

	for (size_t j = jc[0]; j < jc[1]; ++j, ++nnz) {
		data_elem[nnz] = f_data_p[j];
		data_idx[nnz] = static_cast<unsigned int>(ir[j]);
	}

	data_elem.resize(nnz);
	data_idx.resize(nnz);
}

void GetPartialMatlabVector(const mxArray* vin,
	std::vector<unsigned int>& var_subset,
	std::vector<unsigned int>& var_state) {
	if (mxIsDouble(vin) == false) {
		mexErrMsgTxt("Passed array is no double array.\n");
		assert(0);
	}

	size_t vec_count = mxGetNumberOfElements(vin);
	size_t non_nan_count = 0;
	const double* vec_p = mxGetPr(vin);
	for (unsigned int n = 0; n < vec_count; ++n)
		non_nan_count += (boost::math::isnan)(vec_p[n]) ? 0 : 1;

	var_subset.resize(non_nan_count);
	var_state.resize(non_nan_count);
	unsigned int oi = 0;
	for (unsigned int n = 0; n < vec_count; ++n) {
		if ((boost::math::isnan)(vec_p[n]))
			continue;

		assert(oi < var_subset.size());
		var_subset[oi] = n;
		var_state[oi] = static_cast<unsigned int>(vec_p[n]);
		oi += 1;
	}
}

// return false if failed, true if successful.
bool matlab_parse_factorgraphmodel(const mxArray* par,
	Grante::FactorGraphModel& model) {
	if (mxIsStruct(par) == false) {
		mexErrMsgTxt("Factorgraph model parameter must be a structure.\n");
		return (false);
	}

	// Parse model
	int fn_ftypes = mxGetFieldNumber(par, "factor_types");
	if (fn_ftypes < 0) {
		mexErrMsgTxt("model must contain a .factor_types field.\n");
		return (false);
	}
	const mxArray* model_ftypes = mxGetFieldByNumber(par, 0, fn_ftypes);
	if (mxIsStruct(model_ftypes) == false) {
		mexErrMsgTxt("model.factor_types must be an array of structures.\n");
		return (false);
	}
	size_t num_ftypes = mxGetNumberOfElements(model_ftypes);
	for (unsigned int fti = 0; fti < num_ftypes; ++fti) {
		const mxArray* ft_name = mxGetField(model_ftypes, fti, "name");
		if (ft_name == NULL) {
			mexErrMsgTxt("model.factor_types must contain a .name field.\n");
			return (false);
		}

		const mxArray* ft_card = mxGetField(model_ftypes, fti, "card");
		if (ft_card == NULL) {
			mexErrMsgTxt("model.factor_types must contain a .card field.\n");
			return (false);
		}
		if (mxIsDouble(ft_card) == false) {
			mexErrMsgTxt("model.factor_types.card must be a double array.\n");
			return (false);
		}

		// Might be NULL
		const mxArray* ft_weights = mxGetField(model_ftypes, fti, "weights");

		std::string ft_name_str = GetMatlabString(ft_name);
		std::vector<unsigned int> card;
		GetMatlabVector(ft_card, card);
		std::vector<double> weights;
		if (ft_weights != NULL)
			GetMatlabVector(ft_weights, weights);

		const mxArray* ft_type = mxGetField(model_ftypes, fti, "type");
		const mxArray* ft_A = mxGetField(model_ftypes, fti, "A");
		if (ft_type != NULL && mxIsEmpty(ft_type) == false) {
			std::string ft_type_str = GetMatlabString(ft_type);

			// Non-linear factor type
			const mxArray* ft_data_size = mxGetField(model_ftypes, fti, "data_size");
			if (ft_data_size == NULL || mxIsDouble(ft_data_size) == false ||
				mxGetNumberOfElements(ft_data_size) != 1) {
				mexErrMsgTxt("model.factor_types.data_size must be given as "
					"(1,1) double for non-linear factor types.\n");
				return (false);
			}
			unsigned int data_size =
				static_cast<unsigned int>(mxGetScalar(ft_data_size));
			assert(data_size > 0);
			const mxArray* ft_options = mxGetField(model_ftypes, fti, "options");
			// TODO
			if (ft_type_str == "rbfnet") {
				if (ft_options == NULL || mxIsDouble(ft_options) == false
					|| mxGetNumberOfElements(ft_options) != 2) {
					mexErrMsgTxt("For rbfnet factor types, "
						"model.factor_types.options must be provided as "
						"(1,2) double vector.\n");
					return (false);
				}
				double* ft_options_p = mxGetPr(ft_options);
				unsigned int rbf_basis_count =
					static_cast<unsigned int>(ft_options_p[0]);
				assert(rbf_basis_count > 0);
				double log_beta = ft_options_p[1];
				Grante::NonlinearRBFFactorType* factortype =
					new Grante::NonlinearRBFFactorType(ft_name_str, card,
						data_size, rbf_basis_count, log_beta);
				if (weights.empty() == false)
					factortype->InitializeWeights(weights);

				model.AddFactorType(factortype);
			} else {
				mexErrMsgTxt("Unsupported non-linear factor type.\n");
				return (false);
			}
		} else if (ft_A != NULL && mxIsEmpty(ft_A) == false) {
			unsigned int prod_card = 1;
			for (unsigned int fci = 0; fci < card.size(); ++fci)
				prod_card *= card[fci];
			if (mxIsDouble(ft_A) == false ||
				mxGetNumberOfElements(ft_A) != prod_card) {
				mexErrMsgTxt("When using factor type with .A field, "
					"the number of elements in A must equal the product "
					"cardinality of the factor.\n");
				return (false);
			}
			std::vector<unsigned int> A_ui;
			GetMatlabVector(ft_A, A_ui);
			std::vector<int> A(A_ui.size());
			std::copy(A_ui.begin(), A_ui.end(), A.begin());

			const mxArray* ft_data_size = mxGetField(model_ftypes, fti, "data_size");
			if (ft_data_size == NULL || mxIsDouble(ft_data_size) == false ||
				mxGetNumberOfElements(ft_data_size) != 1) {
				mexErrMsgTxt("model.factor_types.data_size must be given as "
					"(1,1) double for general linear factor types.\n");
				return (false);
			}
			unsigned int data_size =
				static_cast<unsigned int>(mxGetScalar(ft_data_size));

			Grante::LinearFactorType* factortype =
				new Grante::LinearFactorType(ft_name_str, card, weights,
				data_size, A);
			model.AddFactorType(factortype);
		} else {
			// Linear factor type
			Grante::FactorType* factortype =
				new Grante::FactorType(ft_name_str, card, weights);
			model.AddFactorType(factortype);
		}
	}
	return (true);
}

Grante::FactorGraph* matlab_parse_factorgraphs(
	Grante::FactorGraphModel& model, const mxArray* fgs,
	unsigned int fgi) {
	if (mxIsStruct(fgs) == false) {
		mexErrMsgTxt("Factor graphs parameter must be a structure.\n");
		return (NULL);
	}

	const mxArray* fg_card = mxGetField(fgs, fgi, "card");
	if (fg_card == NULL) {
		mexErrMsgTxt("Each element in factor_graphs must contain a .card "
			"field.\n");
		return (NULL);
	}

	// Create factor graph
	std::vector<unsigned int> fg_card_vec;
	GetMatlabVector(fg_card, fg_card_vec);
	Grante::FactorGraph* fg = new Grante::FactorGraph(&model, fg_card_vec);

	// Add data sources, if present
	std::map<unsigned int, const Grante::FactorDataSource*> ds_map;
	mxArray* datasources = mxGetField(fgs, fgi, "datasources");
	if (datasources != 0) {
		// Has datasources
		size_t num_ds = mxGetNumberOfElements(datasources);
		for (unsigned int dsi = 0; dsi < num_ds; ++dsi) {
			const mxArray* ds_id = mxGetField(datasources, dsi, "id");
			if (ds_id == 0) {
				mexErrMsgTxt("Data source has no id element.\n");
				return (0);
			}
			if (mxGetNumberOfElements(ds_id) != 1) {
				mexErrMsgTxt("Data source id must be a (1,1) array.\n");
				return (0);
			}
			unsigned int ds_id_val =
				static_cast<unsigned int>(mxGetScalar(ds_id));
			const mxArray* ds_data = mxGetField(datasources, dsi, "data");
			if (ds_data == 0) {
				mexErrMsgTxt("Data source has no or empty data elements.\n");
				return (0);
			}
			const Grante::FactorDataSource* new_ds = 0;
			if (mxIsSparse(ds_data)) {
				std::vector<double> data;
				std::vector<unsigned int> data_sparse_index;
				GetMatlabVector(ds_data, data, data_sparse_index);
				new_ds = new Grante::FactorDataSource(data, data_sparse_index);
			} else {
				std::vector<double> data;
				GetMatlabVector(ds_data, data);
				new_ds = new Grante::FactorDataSource(data);
			}
			ds_map[ds_id_val] = new_ds;
			fg->AddDataSource(new_ds);
		}
	}

	// Add factors
	const mxArray* fg_factors = mxGetField(fgs, fgi, "factors");
	if (fg_factors == NULL) {
		mexErrMsgTxt("Each element in factor_graphs must contain at "
			"least one factor.\n");
		return (NULL);
	}
	size_t num_fs = mxGetNumberOfElements(fg_factors);
	for (unsigned int fi = 0; fi < num_fs; ++fi) {
		// factor type (id or name)
		const mxArray* f_type = mxGetField(fg_factors, fi, "type");
		const Grante::FactorType* cur_ft = NULL;
		if (mxIsChar(f_type)) {
			std::string fname = GetMatlabString(f_type);
			cur_ft = model.FindFactorType(fname);
			if (cur_ft == NULL) {
				mexErrMsgTxt("Invalid factor specification: "
					"cannot find factor type with given name.\n");
				delete (fg);
				return (NULL);
			}
		} else {
			// Matlab index correction
			unsigned int fidx =
				static_cast<unsigned int>(mxGetScalar(f_type)) - 1;
			if (fidx >= model.FactorTypes().size()) {
				mexErrMsgTxt("Invalid factor specification: "
					"factor type index is too large.\n");
				delete (fg);
				return (NULL);
			}
			cur_ft = model.FactorTypes()[fidx];
		}

		// vars
		const mxArray* f_vars = mxGetField(fg_factors, fi, "vars");
		if (f_vars == NULL) {
			mexErrMsgTxt("Invalid factor specification: "
				".vars field missing.\n");
			delete (fg);
			return (NULL);
		}
		std::vector<unsigned int> vars_vec;
		GetMatlabVector(f_vars, vars_vec);
		for (unsigned int vi = 0; vi < vars_vec.size(); ++vi)
			vars_vec[vi] -= 1;	// Matlab index correction

		// data
		const mxArray* f_data = mxGetField(fg_factors, fi, "data");
		const mxArray* f_dsrc = mxGetField(fg_factors, fi, "dsrc");
		Grante::Factor* fac = NULL;
		if (f_dsrc != 0) {
			if (f_data != 0 && mxIsEmpty(f_data) == false) {
				mexErrMsgTxt("If .dsrc is used in a factor, "
					".data must be empty.\n");
				return (0);
			}
			if (mxGetNumberOfElements(f_dsrc) != 1) {
				mexErrMsgTxt("The .dsrc field must be a (1,1) array.\n");
				return (0);
			}
			unsigned int dsrc_id =
				static_cast<unsigned int>(mxGetScalar(f_dsrc));
			if (ds_map.find(dsrc_id) == ds_map.end()) {
				mexErrMsgTxt("Factor mentions a non-existing data source.\n");
				return (0);
			}
			const Grante::FactorDataSource* ds_p = ds_map[dsrc_id];
			fac = new Grante::Factor(cur_ft, vars_vec, ds_p);
		} else if (f_data != 0 && mxIsSparse(f_data)) {
			if (mxGetN(f_data) != 1) {
				mexErrMsgTxt("Sparse factor .data field must be (M,1).\n");
				delete (fg);
				return (0);
			}
			// sparse vector
			std::vector<double> data_elem;
			std::vector<unsigned int> data_idx;
			GetMatlabVector(f_data, data_elem, data_idx);
			fac = new Grante::Factor(cur_ft, vars_vec, data_elem, data_idx);
		} else {
			// dense vector
			std::vector<double> data_vec;
			if (f_data != NULL)
				GetMatlabVector(f_data, data_vec);

			fac = new Grante::Factor(cur_ft, vars_vec, data_vec);
		}
		fg->AddFactor(fac);
	}
	return (fg);
}

bool matlab_parse_factorgraphs(Grante::FactorGraphModel& model,
	const mxArray* fgs, std::vector<Grante::FactorGraph*>& FG) {
	if (mxIsStruct(fgs) == false) {
		mexErrMsgTxt("Factor graphs parameter must be a structure.\n");
		return (false);
	}
	size_t num_fgs = mxGetNumberOfElements(fgs);
	for (unsigned int fgi = 0; fgi < num_fgs; ++fgi) {
		Grante::FactorGraph* fg_cur =
			matlab_parse_factorgraphs(model, fgs, fgi);
		if (fg_cur == NULL) {
			for (unsigned int n = 0; n < FG.size(); ++n) {
				if (FG[fgi] != NULL)
					delete (FG[fgi]);
			}
			FG.clear();
			return (false);
		}
		FG.push_back(fg_cur);
	}
	return (true);
}

