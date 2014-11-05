function [fg_model_trained] = grante_learn(model, factor_graphs, ...
	observations, priors, method, options);
% GRANTE_LEARN Learn a discrete random field model using observation data.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 18th March 2010
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
%             (which is squareform(1:(k*(k-1)/2)) for a k-by-k table),
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
%    observations: (1,L) array of structures, each containing the following
%       fields:
%       .id: index into factor_graphs this observation refers to.
%       .labels: (1,V) discrete observation vector.  TODO: support expectation
%          observations.  labels are 1-based.
%      or
%       .expectations: (1,F) cellarray of expectation-observations (marginals).
%    priors: empty ([]) or an array of structures, each containing the
%       following fields:
%       .factor_type: either an index between 1 and T indexing into
%          model.factor_types, or a string with the name of the factor type.
%       .prior_name: 'normal', 'laplace', or 'studentt'.
%       .prior_opt: double array of prior-related parameters.
%          'normal': [sigma],
%          'laplace': [sigma],
%          'studentt': [degrees_of_freedom, sigma].
%    method: one of the following
%       'mle' (maximum likelihood),
%       'mple' (maximum pseudolikelihood),
%       'mcle' (maximum composite likelihood),
%       'mxxle' (maximum criss-cross likelihood), for 4-neighborhood grid
%          graphs only,
%       'npw' (naive piecewise), for all factor graphs,
%       'perceptron' (structured perceptron): does not support priors,
%       'avg_perceptron' (averaged structured perceptron): does not support
%          priors.
%       'ssvm' (structured SVM, margin rescaling, Hamming loss): you should
%          set the options.ssvm_c parameter.  Note: if you use ssvm, you
%          MUST USE a normal prior for all parameters!
%       'em' (expectation maximization), labels are allowed to contain NaN
%          values for missing-at-random values.  FIXME: only tree-structured
%          graphs are supported currently.  For tree-structured graphs we
%          perform exact inference to compute expectations.
%       'cd_obs' (contrastive divergence, fully observed).
%       'cd' (contrastive divergence, partially observed), labels are allowed
%          to contain NaN values for missing-at-random values.  All factor graphs
%          are supported.
%    options: (optional) structure with method-dependent options,
%       .max_iter: (1,1) scalar >=1, maximum number of learning iterations,
%          default: 1000,
%       .conv_tol: (1,1) scalar >=0.0, convergence tolerance
%          (method-dependent), default: 1.0e-5.
%       .mle_infer_method: probabilistic inference method to use for MLE.  The
%          method parameter must be 'mle' and mle_infer_method must be one of
%          'bp', 'nmf', 'smf', 'ais', 'bfexact'.  The inference method must be
%          able to provide the log-partition function, so 'gibbs' cannot be
%          used directly.
%          When mle_infer_method is set, all applicable parameters from
%          grante_infer can also be used for that method.
%       .opt_method: for 'mle', 'mple', 'mcle', 'mxxle', and 'npw' methods,
%          this selects the numerical optimization procedure, one of the
%          following:
%          'lbfgs' (default): limited memory bfgs,
%          'gradient': simple gradient method (slow),
%          'bb': Barzilai-Borwein method (no convergence guarantee),
%          'fista': proximal method, for sparsity priors.
%       .em_subiter: (1,1) scalar >=1, maximum number of iterations in EM
%          maximization subproblem, default: 100,
%       .em_subtol: (1,1) scalar >=0.0, convergence tolerance for EM
%          maximization subproblem, default: 1.0e-6,
%       .ssvm_c: (1,1) scalar >0.0, C parameter of the structured SVM
%          objective:  Omega(w) + (C/N) \sum_{n=1}^{N} \Delta_n(f(x_n;w),y_n)
%          Default: 1.0.
%       .mcle_cover: (1,1) double, integer-valued >= 0.  Default is 0 for
%          uniform single-covering decomposition.  Values >=1 lead to
%          randomized covering.  When the value is >1, multiple randomized
%          coverings are performed.
%       .cd_k: (1,1) double, integer-valued >= 1.  Contrastive divergence
%          number of Gibbs sweeps used.  Default is 1.
%       .cd_minibatchsize: (1,1) double, integer-valued >= 0.  Size as number
%          of instances of the mini batches used.  If zero, all instances are
%          used (batch).  Default: 10.
%       .cd_stepsize: (1,1) double, >0.  Constant stepsize used within CD.
%          Default: 1.0e-2.

% TODO: check parameters
if nargin < 6
	options = [];
end
if strcmp(method,'ssvm')
	if numel(priors) ~= numel(model.factor_types)
		error(['Using "ssvm" training you must use normal priors for all factor types.']);
	end
	for i=1:numel(priors)
		if ~strcmp(priors(i).prior_name, 'normal')
			error(['Currently all priors must be "normal" priors when using "ssvm".']);
		end
	end
end
[fg_model_trained] = mex_grante_learn(model, factor_graphs, observations, ...
	priors, method, options);
