function [d_kl]=grante_compute_kl_inf(model_q, model_p, fg_p, fg_q, ...
	inf_method, inf_options);
%GRANTE_COMPUTE_KL_INF Approximately compute the Kullback-Leibler divergence
%D(q||p) for two given model distributions over the same variables.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 1th September 2011.

% Note that we have
%    D_KL(q||p) = < E_p(y) - E_q(y) >_{y~q} + log Z_p - log Z_q.

[result_p,logz_p]=grante_infer(model_p,fg_p,inf_method,inf_options);
[result_q,logz_q]=grante_infer(model_q,fg_q,inf_method,inf_options);

mean_E_q=mean_energy(model_q,fg_q,result_q(1));
mean_E_p=mean_energy(model_p,fg_p,result_q(1));

d_kl=(mean_E_p+logz_p) - (mean_E_q+logz_q);

function [me]=mean_energy(model,fg,inf_res);
me=grante_evaluate(model,fg,inf_res.marginals);

