function [d_kl]=grante_compute_kl(model_q, model_p, fg_p, fg_q);
%GRANTE_COMPUTE_KL Approximately compute the Kullback-Leibler divergence
%D(q||p) for two given model distributions over the same variables.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 25th January 2011.

% Note that we have
%    D_KL(q||p) = < E_p(y) - E_q(y) >_{y~q} + log Z_p - log Z_q.

options=[];
options.ais_k=200;
options.ais_sweeps=1;
options.ais_samples=500;
[result,logz_p]=grante_infer(model_p,fg_p,'ais',options);
[result,logz_q]=grante_infer(model_q,fg_q,'ais',options);

sample_count=5000;
options.gibbs_burnin=100;
options.gibbs_spacing=2;
[states]=grante_sample(model_q,fg_q,'gibbs',sample_count,options);

[E_q]=grante_evaluate(model_q,fg_q,states);
[E_p]=grante_evaluate(model_p,fg_p,states);

d_kl=mean(E_p-E_q) + logz_p - logz_q;

