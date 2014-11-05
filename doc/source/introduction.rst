
Introduction
============

Grante is a small C++ library for solving estimation and inference problems in
probabilistic discrete undirected graphical models.

The library has a native C++ interface as well as a wrapper to expose most
functionality to the Matlab language.

It is particularly suited to learn about discrete graphical models, to rapidly
prototype small to medium-sized models, and to compare different inference and
estimation methods.


Benefits
--------

Grante allows its users to quickly set up sophisticated models describing
probability distributions over large finite sets.  The model parameters can be
estimated from observations and accurate predictions can be made on novel data
points.

To use the library successfully, a basic knowledge of probabilistic graphical
models is required.  A tutorial explaining basic concepts is available
(`PDF <http://www.nowozin.net/sebastian/cvpr2011tutorial/>`_).


Features
--------

General:

+ Factor graph representation for discrete state models.
+ Unary, pairwise, and high-order factors.
+ Support for linear data-dependent factors and parameter sharing in factors.
+ Support for complex tying patterns within factor energies (symmetry,
  sparse energy tables, Potts, etc).
+ Support for non-linear data-dependent factors.
+ Support for sparse factor data and for shared data among multiple factors.
+ Multi-core support through OpenMP.
+ Few dependencies: only boost 1.45 or higher and a C++0x compiler are required.


Inference:

+ Exact inference, sampling and MAP for tree-structured factor graphs.
+ (Metropolized) Gibbs sampling inference for general factor graphs.
+ Sum-product and max-product Loopy Belief Propagation, sequential and
  parallel schedules.
+ Naive Mean Field for general factor graphs.
+ Structured Mean Field for general factor graphs (using v-acyclic
  decompositions).
+ Annealed Importance Sampling (AIS) inference for general factor graphs.
+ Parallel Tempering (Replica-Exchange Monte Carlo) inference for general
  factor graphs.
+ Stochastic Approximation Monte Carlo (SAMC) inference for general factor
  graphs.
+ Generalized Swendsen-Wang MCMC inference for pairwise factor graphs where
  each variable has the same state space.
+ Approximate MAP-MRF Linear Programming inference (tree-based).
+ Approximate MAP-MRF inference by simulated annealing.
+ Min-sum/sum-product diffusion.
+ Conditioning of distributions by discrete observations and
  factor-decomposable expectations.


Learning:

+ Maximum (Conditional) Likelihood Learning for tree-structured factor
  graphs.
+ Maximum Pseudolikelihood Learning for general factor graphs.
+ Maximum Composite Likelihood Learning for general factor graphs,
  specialized variant for 4-neighborhood grid graphs.
+ Contrastive divergence learning from fully observed and partially observed
  data.
+ Structured SVM for factor-decomposable loss functions.  The default
  structured loss is the Hamming loss.  Batch and stochastic training, as
  well as BMRM-training are supported.
+ Structured Perceptron (averaged).
+ Prior distributions (and regularizers): multivariate Normal (L2),
  multivariate Laplace (L1), multivariate Student-t, multivariate Hyperbolic.
+ Expectation Maximization (EM) for partially observed data.


Optimization:

+ Barzilai-Borwein method for unconstrained differentiable minimization.
+ Limited memory BFGS method for unconstrained differentiable minimization.
+ FISTA composite minimization method for sparsity priors.
+ Subgradient method for unconstrained continuous strictly convex
  minimization.
+ Stochastic subgradient method for incremental unconstrained continuous
  strictly convex minimization.
+ Approximately optimal v-acyclic decompositions of factor graphs.
+ Approximate tree-covering decomposition.


Interfaces:

+ Matlab interface for learning, inference, and sampling.
+ Efficient saving and loading of models and factor graph instances.

