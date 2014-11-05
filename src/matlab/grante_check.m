function grante_check(model, factor_graphs, observations);
% GRANTE_CHECK Check a factor graph model and factor graphs.
%
% Input
%    model: factor graph model, as in grante_infer.m,
%    factor_graphs: an array of factor graph structures, as described in
%       grante_infer.m,
%    observations: (optional) an array of observation structures, as described
%       in grante_learn.m.
%
% Output
% An error with explanation will be thrown in case the factor graph or model
% is showing some misspecification.  This does not guarantee the resulting
% model and factor graphs are error free but does some basic checking.
% We also check the dynamic range of numeric features and provide a warning in
% case these are large.

check_model(model);
for n=1:numel(factor_graphs)
	check_factor_graph(model, factor_graphs(n));
end
if nargin >= 3
	check_observations(model, factor_graphs, observations);
end

function check_model(model);
% 1. Check order of weights
for fti=1:numel(model.factor_types)
    if ~isfield(model.factor_types(fti),'weights') || ...
        isempty(model.factor_types(fti).weights)
        data_dims=0;
        fti_w_s=model.factor_types(fti).card;
    else
    	data_dims=ndims(model.factor_types(fti).weights) - ...
        	numel(model.factor_types(fti).card);
        if data_dims < 0
        	error(['Factor type "', model.factor_types(fti).name, '" has ', ...
        		'a weight vector with too few dimensions.']);
        end
        fti_w_s=size(model.factor_types(fti).weights);
    end
	for vci=1:numel(model.factor_types(fti).card)
		if model.factor_types(fti).card(vci) ~= fti_w_s(data_dims + vci)
			error(['Factor type "', model.factor_types(fti).name, ...
				'" has a weight vector of wrong dimensionality: ', ...
				'it must be ordered (D1,...,Dm,Y1,...,Yk).']);
		end
	end
end

function check_factor_graph(model, fg);
% 1. Check factors refer to variables of correct cardinality
var_used=zeros(1,numel(fg.card));
for fi=1:numel(fg.factors)
	ft=[];
	for fti=1:numel(model.factor_types)
		if strcmp(model.factor_types(fti).name, fg.factors(fi).type)
			ft=fti;
			break;
		end
	end
	if isempty(ft)
		error(['Factor ', num2str(fi), ' has unknown factor type "', ...
			fg.factors(fi).type]);
	end
	if ~isa(fg.factors(fi).vars,'double')
		error(['factor ', num2str(fi), ' vars is no double type.']);
	end
	if numel(fg.factors(fi).vars) ~= numel(model.factor_types(ft).card)
		error(['factor ', num2str(fi), ...
			' has wrong number of adjacent variables.']);
	end
	vcard=zeros(1,numel(fg.factors(fi).vars));
	for fvi=1:numel(fg.factors(fi).vars)
		if fg.factors(fi).vars(fvi) > numel(fg.card)
			error(['Factor ', num2str(fi), ' has invalid variable indices.']);
		end
		vcard(fvi) = fg.card(fg.factors(fi).vars(fvi));
	end
	if ~isempty(find(vcard ~= model.factor_types(ft).card))
		error(['Factor ', num2str(fi), ' uses variables of wrong ', ...
			'cardinality.']);
	end

	% Mark used variables
	var_used(fg.factors(fi).vars) = var_used(fg.factors(fi).vars) + 1;

	% Check data is proper
	if ~isempty(fg.factors(fi).data)
		if ~isempty(find(isnan(fg.factors(fi).data))) || ...
			~isempty(find(isinf(fg.factors(fi).data)))
			error(['Factor ', num2str(fi), ' has NaN or Inf elements in ', ...
				'data vector.']);
		end

		if max(abs(fg.factors(fi).data(:))) >= 1.0e4
			warning(['Factor ', num2str(fi), ' has a data element with ', ...
				'magnitude ', num2str(max(abs(fg.factors(fi).data(:)))), ...
				' that could cause numerical issues.']);
		end
	end

	% 2. Check factor data sizes match if factor type has empty weight
	if isempty(model.factor_types(ft).weights)
		if numel(fg.factors(fi).data) ~= prod(model.factor_types(ft).card)
			error(['Factor ', num2str(fi), ' has wrong energy table: ', ...
				'has ', num2str(numel(fg.factors(fi).data)), ' elements, ', ...
				'should have ', num2str(prod(model.factor_types(ft).card)), ...
				' elements.']);
		end
	else
		if issparse(fg.factors(fi).data) && nnz(fg.factors(fi).data) == 0
			error(['Factor ', num2str(fi), ...
				' has sparse data which is empty, this is not supported.']);
		end
	end
end
% 3. Check for isolated variables
if ~isempty(find(var_used == 0))
	error(['Factor graph has ', num2str(sum(var_used == 0)), ...
		' variables not appearing in any factor: ', ...
		num2str(find(var_used == 0))]);
end
% 4. TODO: check that same factor type is not used twice or more at the same
% sites


function check_observations(model, fg, obs);
if numel(obs) ~= numel(fg)
	error(['Number of observations (', num2str(numel(obs)), ...
		') and number of factor graphs (', num2str(numel(fg)), ...
		') do not agree.']);
end
for n=1:numel(obs)
	if obs(n).id < 1 || obs(n).id > numel(fg)
		error(['Observation ', num2str(n), ' has an id (', ...
			num2str(obs(n).id), ' outside the valid range (1-', ...
			num2str(numel(fg)), ').']);
	end
	fg_id=obs(n).id;
	if numel(obs(n).labels) ~= numel(fg(fg_id).card)
		error(['Observation ', num2str(n), ' has wrong label ', ...
			'vector length.']);
	end
	if find(obs(n).labels == 0)
		error(['Observation ', num2str(n), ' has zero labels.']);
	end
	if find(obs(n).labels > fg(fg_id).card)
		error(['Observation ', num2str(n), ' has label values outside ', ...
			'the legal variable cardinalities.']);
	end
end

