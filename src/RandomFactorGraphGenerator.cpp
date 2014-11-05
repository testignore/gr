
#include <numeric>
#include <limits>
#include <tr1/unordered_map>
#include <cmath>
#include <ctime>
#include <cassert>

#include "RandomFactorGraphGenerator.h"
#include "FactorGraphStructurizer.h"

namespace Grante {

RandomFactorGraphGenerator::RandomFactorGraphGenerator(
	const FactorGraphModel* fg_model)
	: fg_model(fg_model),
		rgen(static_cast<const boost::uint32_t>(std::time(0))+1),
		randu(rgen, rdestu) {
}

FactorGraph* RandomFactorGraphGenerator::GenerateTreeStructured(
	const std::vector<double>& ft_dist, unsigned int factor_count) const {
	const std::vector<FactorType*>& ftypes = fg_model->FactorTypes();
	assert(ft_dist.size() == ftypes.size());
	assert(std::fabs(std::accumulate(ft_dist.begin(), ft_dist.end(), 0.0)
		- 1.0) < 1e-5);
	assert(factor_count > 0);

	// Create one variable selection random number generator per factor type
	boost::uint32_t btime = static_cast<const boost::uint32_t>(std::time(0));
	std::vector<boost::mt19937> vs_rgen;
	std::vector<boost::uniform_int<unsigned int> > vs_rdestd;
	std::vector<boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > > vs_rand_n;
	for (unsigned int fti = 0; fti < ftypes.size(); ++fti) {
		vs_rgen.push_back(boost::mt19937(btime + fti));
		vs_rdestd.push_back(boost::uniform_int<unsigned int>(0,
			static_cast<unsigned int>(ftypes[fti]->Cardinalities().size()-1)));
		vs_rand_n.push_back(boost::variate_generator<boost::mt19937,
			boost::uniform_int<unsigned int> >(vs_rgen[fti], vs_rdestd[fti]));
	}

	// List of all variables generated, indexed by cardinality
	std::tr1::unordered_map<unsigned int, std::vector<unsigned int> > gen_vars;
	unsigned int att_rgen_count = 0;

	// First determine a sequence of factors to be added
	std::vector<unsigned int> card;
	std::vector<Factor*> factors;
	for (size_t fi = 0; fi < factor_count; ) {
		// Sample factor type to add
		size_t ft = 0;
		while (true) {
			double uval = randu();
			double csum = 0.0;
			for (ft = 0; ft < ft_dist.size(); ++ft) {
				csum += ft_dist[ft];
				if (uval <= csum)
					break;
			}

			// Candidate factor type
			assert(ft < ftypes.size());
			const FactorType* ftype = ftypes[ft];

			// Adjacent variable indices
			unsigned int f_vc =
				static_cast<unsigned int>(ftype->Cardinalities().size());
			std::vector<unsigned int> var_index(f_vc);
			if (factors.empty()) {
				// Add factor to empty factor graph
				unsigned int v_base_idx =
					static_cast<unsigned int>(card.size());
				for (unsigned int vi = 0; vi < f_vc; ++vi) {
					unsigned int abs_var_index = v_base_idx + vi;
					var_index[vi] = abs_var_index;

					unsigned int var_card = ftype->Cardinalities()[vi];
					card.push_back(var_card);
					gen_vars[var_card].push_back(abs_var_index);
				}
			} else {
				// Determine attaching variable
				unsigned int attaching_id = vs_rand_n[ft]();
				unsigned int attaching_card =
					ftype->Cardinalities()[attaching_id];
				if (gen_vars[attaching_card].empty())
					continue;	// do not add this factor: nowhere to attach

				// Link factor to a random variable of the right cardinality
				boost::mt19937 att_rgen(btime + att_rgen_count);
				att_rgen_count += 1;
				boost::uniform_int<unsigned int> att_rdestd(0,
					static_cast<unsigned int>(gen_vars[attaching_card].size()-1));
				boost::variate_generator<boost::mt19937,
					boost::uniform_int<unsigned int> >
					att_rand_n(att_rgen, att_rdestd);
				unsigned int gv_idx = att_rand_n();
				assert(gv_idx < gen_vars[attaching_card].size());
				var_index[attaching_id] = gen_vars[attaching_card][gv_idx];

				// Add the new variables of this factor
				unsigned int v_base_idx = static_cast<unsigned int>(card.size());
				unsigned int abs_var_index = v_base_idx;
				for (unsigned int vi = 0; vi < f_vc; ++vi) {
					if (vi == attaching_id) {
						continue;
					} else {
						// Add variable
						unsigned int var_card = ftype->Cardinalities()[vi];
						var_index[vi] = abs_var_index;
						card.push_back(var_card);
						gen_vars[var_card].push_back(abs_var_index);
						abs_var_index += 1;
					}
				}
			}

			// Generate data
			std::vector<double> data;
			// FIXME/TODO: data_dim
#if 0
			if (data_dim > 0) {
				data.resize(data_dim);
				for (unsigned int d = 0; d < data_dim; ++d)
					data[d] = 2.0*randu() - 1.0;
			}
#endif

			factors.push_back(new Factor(ftype, var_index, data));
			fi += 1;	// One factor added
			break;
		}
	}

	// Now add all collected factors
	FactorGraph* fg = new FactorGraph(fg_model, card);
	for (std::vector<Factor*>::iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		fg->AddFactor(*fi);
	}
	assert(FactorGraphStructurizer::IsForestStructured(fg));

	return (fg);
}

}

