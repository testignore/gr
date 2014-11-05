function [d_e,d_ec]=grante_compute_estatistic(model_q, model_p, fg_p, fg_q);
%GRANTE_COMPUTE_ESTATISTIC Compute the energy statistic between two
%distributions.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 31th January 2011.
%
%    D_ES(q,p) = 2 < |y_q - y_p| >_{y_q~q,y_p~p}
%                - < |y_q - y'_q| >_{y_q~q,y'_q~q}
%                - < |y_p - y'_p| >_{y_p~p,y'_p~p}.
%
% See (Szekely, Rizzo, "Testing for equal distributions in high dimensions",
% 2004).
% Also informative: http://en.wikipedia.org/wiki/E-statistic
%
% Output
%    d_e: The energy statistic between q and p.
%    d_ec: The E-coefficient of inhomogeneity, >=0, <=1.

sample_count=5000;
options.gibbs_burnin=100;
options.gibbs_spacing=2;
[states_q]=grante_sample(model_q,fg_q,'gibbs',sample_count,options);
[states_p]=grante_sample(model_p,fg_p,'gibbs',sample_count,options);

states_qp=[states_q,states_p]';	% form concatenated version
D=squareform(pdist(states_qp, ...
	@(x1,x2)(sqrt(sum(repmat(x1,size(x2,1),1)~=x2,2)))));

D_qq=(1.0/sample_count^2)*sum(sum(D(1:sample_count,1:sample_count)));
D_pp=(1.0/sample_count^2)*sum(sum(D((sample_count+1):end,(sample_count+1):end)));
D_pq=(2.0/sample_count^2)*sum(sum(D(1:sample_count,(sample_count+1):end)));

%d_e=((sample_count^2)/(2.0*sample_count))*(D_pq-D_pp-D_qq);
d_e=D_pq - D_pp - D_qq;
d_ec=d_e / D_pq;

