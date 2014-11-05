function [result, z_result] = grante_infer(model, factor_graphs, method, options);
%GRANTE_INFER Probabilistic inference: infer marginals for all factors and
%the log-partition function of the distribution.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 19th March 2010
%
% Input
%    model: Factor graph model structure (Grante::FactorGraphModel) containing
%       the following fields,
%       .factor_types: (1,T) array of structures with elements
%          .name: string containing a factor type identifier.
%          .card: (1,K) array containing cardinalities of adjacent variables.
%             K denotes the order of this factor (>= 1).
%          .weights: (optional) (1,D) double array containing parameters of
%             this factor type.  If there is no .weights element or it is
%             empty, then this factor type does not have parameters.
%        Tying, symmetry-constraints, etc. are supported by adding the fields
%          .data_size: (1,1) scalar containing the data size of factors
%             instantiating this type.  Must be >=1.
%          .A: (optional) (1,prod(card)) double array containing integer
%             elements.  An element of -1 fixes that energy element to be
%             zero.  An integer element >=0 indexes a tied group of energies.
%             For example, for a pairwise factor between two variables with
%             three states each, we can enforce symmetric factors by setting
%                A=[0 1 2;
%                   1 3 4; 
%                   2 4 5],
%             We could enforce a Potts type of energy by setting
%                A=[-1  0  0;
%                    0 -1  0;
%                    0  0 -1],
%             forcing the diagonal to zero.  When using a non-empty .A field,
%             the .weights size changes.  It becomes
%             numel(unique(A(find(A(:)>=0))))*data_size.  Currently the use of
%             .A prevents using nonlinear factor types.
%        Nonlinear factor types are supported as follows, by setting the
%        additional fields
%          .type: string, one of
%             'rbfnet', Radial Basis Function network with exp basis
%                functions of fixed bandwidth, but variable weights and
%                centers.  For learning (grante_learn) the basis will be
%                initialized by a random training sample subsample.
%          .data_size: (1,1) scalar containing the data size of factors
%             instantiating this type.  Must be >=1.
%          .options: additional options for non-linear functions, for
%             different non-linear maps this differs as follows.
%             rbfnet: this is a (1,2) vector [rbf_basis_count, log_beta] where
%                rbf_basis_count >= 1 is the number of RBF basis functions per
%                factor energy index and log_beta is explained in
%                NonlinearRBFFactorType.h.
%    factor_graphs: (1,N) array of structures describing factor graphs
%       (Grante::FactorGraph).  Each structure contains the following
%       fields:
%       .card: (1,V) array containing cardinalities of all V variables in the
%          factor graph.  For example, in the case of three binary variables,
%          this would be [2, 2, 2].
%       .factors: (1,F) array of structures, each containing the following
%          fields:
%          .type: either an index between 1 and T indexing into
%             model.factor_types, or a string with the name of the factor
%             type.
%          .vars: (1,K) array containing absolute variable indices mapping to
%             the K adjacent factor variables.  The variables must have the
%             same cardinalities as specified in model.factor_types(t).card.
%          .data: (optional) (M,1) data vector related to this factor
%             instance.  This is required only if the factor type has a
%             non-empty weight vector.  Refer to the documentation of
%             Grante::FactorType.  This element is allowed to be sparse.
%          .dsrc: (optional) (1,1) id indexing a datasource of this factor.
%             In case .dsrc is provided, .data must not be provided or must be
%             empty.
%       .datasources: (optional) (1,DS) array of structures containing data
%          vectors that are shareable across multiple factors.  Each structure
%          contains the following fields:
%          .id: An arbitrary index used to identify the data source within
%             this factor graph.
%          .data: (M,1) data vector.
%    method: A string describing the inference method to use.
%      Probabilistic inference methods
%       'treeinf' (for tree-structured factor graphs only),
%       'bp', plain sum-product belief propagation (general factor graphs),
%       'nmf', naive mean field (general factor graphs),
%       'smf', structured mean field (general factor graphs),
%       'spdiff', sum-product diffusion (general factor graphs),
%       'gibbs', Naive Gibbs sampling (for general factor graphs).
%       'mcgibbs', multiple-chain naive Gibbs sampling, using convergence
%          diagnostics to determine burn-in (for general factor graphs).
%       'ais', Annealed Importance sampling (for general factor graphs).
%       'samc', Stochastic Approximation Monte Carlo (generalized
%          Wang-Landau), for general factor graphs.
%       'sw', Generalized Swendsen-Wang (for pairwise factor graphs with all
%          variables having the same state space).
%       'bfexact', Brute-force exact inference (for general factor graphs).
%      MAP inference methods (energy minimization)
%       'maplp', MAP-MRF LP relaxation (general factor graphs),
%       'mapsa', simulated annealing approximate MAP (general factor graphs),
%       'mapbp', plain max-product belief propagation (general factor graphs),
%       'mapmsd', min-sum diffusion (general factor graphs),
%       'mapbfexact', brute-force exact inference (for general factor graphs).
%    options: (optional) structure with optional fields,
%       .gibbs_burnin: (1,1) scalar >=0, number of Gibbs burn in sweeps,
%          default: 100,
%       .gibbs_samples: (1,1) scalar >=1, number of Gibbs samples to obtain,
%          default: 1000,
%       .gibbs_spacing: (1,1) scalar >=0, number of Gibbs spacing sweeps
%          between samples, default: 0.
%
%       .mcgibbs_maxpsrf: (1,1) scalar >1, maximum tolerated potential scale
%          reduction factor, where values closer to one are more stringend.
%          Default: 1.01.
%       .mcgibbs_chains: (1,1) integer >=2, number of Gibbs chains to run in
%          parallel.  Default: 5.
%       .mcgibbs_samples: (1,1) scalar >=1, number of Gibbs samples to obtain,
%          default: 1000,
%       .mcgibbs_spacing: (1,1) scalar >=0, number of Gibbs spacing sweeps
%          between samples, default: 0.
%
%       .ais_k: (1,1) scalar >=2, number of annealing distributions to use,
%          default: 80,
%       .ais_sweeps: (1,1) scalar >=1, number of Gibbs sweeps to use for
%          updating the intermediate distribution,
%       .ais_samples: (1,1) scalar >=1, number of samples to obtain,
%          default: 100.
%
%       .samc_k: (1,1) scalar >=2, number of temperature levels.  Default: 20.
%       .samc_high_temp: (1,1) scalar >1.0, highest temperature in the ladder,
%          default: 20.0.
%       .samc_swap: (1,1) >0.0, <1.0, temperature swap probability.
%          Default: 0.5.
%       .samc_burnin: (1,1) scalar >=0, number of burnin sweeps,
%          default: 1000.
%       .samc_samples: (1,1) scalar >=0, number of samples, default: 1000.
%
%       .sw_qetemp: (1,1) scalar >0, temperature for factor appearance
%          probabilities.  See SwendsenWangSampler.h for documentation.
%          Default: 1.0.
%       .sw_burnin: (1,1) scalar >=0, number of SW burn in sweeps,
%          default: 50,
%       .sw_sweeps: (1,1) scalar >=1, number of SW sweeps to use for
%          updating the intermediate distribution,
%       .sw_samples: (1,1) scalar >=1, number of samples to obtain,
%          default: 100.
%
%       .sa_steps: (1,1) scalar >=1, number of simulated annealing steps,
%          default: 100,
%       .sa_t0: (1,1) scalar >0, simulated annealing initial temperature,
%          default: 10.0,
%       .sa_tfinal: (1,1) scalar >0, <.sa_t0, simulated annealing final
%          temperature, default: 0.05.
%
%       .lp_max_iter: (1,1) scalar >=0, number of subgradient steps to take,
%          default: 100,
%       .lp_conv_tol: (1,1) scalar >0.0, convergence tolerance,
%          Default: 1.0e-6.
%
%       .msd_max_iter: (1,1) scalar >=0, number of min-sum diffusion passes
%          over the model, default: 100,
%       .msd_conv_tol: (1,1) scalar >0.0, convergence tolerance,
%          default: 1.0e-5.
%
%       .nmf/smf_conv_tol: (1,1) scalar >=0.0, convergence tolerance wrt log_z
%          bound, default: 1.0e-6,
%       .nmf/smf_max_iter: (1,1) integer >=0, maximum number of mean field block
%          coordinate ascent iterations, use zero for no limit.  Default: 50.
%
%       .spdiff_max_iter: (1,1) scalar >=0, number of diffusion passes
%          over the model, default: 100,
%       .spdiff_conv_tol: (1,1) scalar >0.0, convergence tolerance,
%          default: 1.0e-5.
%
%       .bp_conv_tol: (1,1) scalar >=0.0, convergence tolerance wrt marginal
%          updates, default: 1.0e-5,
%       .bp_max_iter: (1,1) integer >=0, maximum number of belief propagation
%          sweeps, use zero for no limit.  Default: 100.
%
%       .verbose: (1,1) scalar, valued 0 or 1, where 1 means some extra
%          verbosity during inference, default: 0.
%
% Output
%    result: Result depends on inference method used.
%    'treeinf', 'bp', 'smf', 'gibbs', 'ais', 'samc', 'sw':
%       (1,N) array, copies from factor_graphs, with an additional
%       element (besides .card and .factors) for each factor graph:
%       .marginals: (1,F) cellarray of arrays.  marginals{fi} contains the
%          marginal distribution for the factor .factors(fi).  The array
%          marginals{fi} is a (Y_1,...,Y_K) double array.
%    'mapbp','maplp', 'mapsa', 'mapmsd':
%       (1,N) array, copies from factor_graphs, with two additional elements
%       for each factor graph:
%       .solution: (1,numel(card)) approximate labeling of optimal solution.
%       TODO: z_result for maplp and mapmsd, provide (2,:) lower bound
%       %.is_optimal_solution: (1,1) with value 0 or 1.  If 1, the .solution is
%       %   the optimal minimum energy state of the model, if 0 it is possibly
%       %   sub-optimal.
%    z_result: (1,N) double array containing the log-partition function for
%       'treeinf', 'bp' and 'ais', and the energies of the returned
%       solutions for 'maplp', 'mapsa' and 'mapbp'.  The 'gibbs' method will
%       return NaN, as it does not provide an estimate for logZ.

if nargin <= 3
	options=[];
end
[result, z_result] = mex_grante_infer(model, factor_graphs, method, options);

