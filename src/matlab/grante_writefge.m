function grante_writefge(model, fg, fge_filename);
%GRANTE_WRITEFGE -- Write factor graph instance in FGE format.
var_count=numel(fg.card);

% <variablecountN>        # Number of variables in the model
% <lc1 [lc2 [... lcN]]>   # Number of states for each variable
fid=fopen(fge_filename,'w');
fprintf(fid, '%d\n', var_count);
for vi=1:var_count
	if vi > 1
		fprintf(fid, ' ');
	end
	fprintf(fid, '%d', fg.card(vi));
end
fprintf(fid, '\n');
fprintf(fid, '%d\n', numel(fg.factors));

% Followed by a list of factors, each factor having:
% <varcount>
% <var1 [var2 [...]]> # one-based indices
% <Energies>      # Stacked energies, in first-moves-fastest order (Matlab)
for fi=1:numel(fg.factors)
	fi_var_count=numel(fg.factors(fi).vars);
	fprintf(fid, '%d\n', fi_var_count);
	for vi=1:fi_var_count
		if vi > 1
			fprintf(fid, ' ');
		end
		fprintf(fid, '%d', fg.factors(fi).vars(vi));
	end
	fprintf(fid, '\n');

	% Write energies
	E=compute_effective_energies(model, fg.factors(fi));
	E=E(:);	% first-moves-fastest order
	assert(numel(E) == prod(fg.card(fg.factors(fi).vars)));
	for ei=1:numel(E)
		if ei > 1
			fprintf(fid, ' ');
		end
		fprintf(fid, '%10.10f', E(ei));
	end
	fprintf(fid, '\n');
end
fclose(fid);


function [E]=compute_effective_energies(model, factor);
% 1. Find factor type
fti=[];
for efti=1:numel(model.factor_types)
	if strcmp(model.factor_types(efti).name, factor.type)
		fti=efti;
		break;
	end
end
assert(~isempty(fti));

% 2. If a fixed factor, we are done
if isempty(model.factor_types(fti).weights)
	E=factor.data;
	return;
end

% 3. Factor depends on global parameters, compute linear energies
esize=model.factor_types(fti).card;
if numel(esize) == 1
	esize=[esize,1];
end
E=zeros(esize);
for ei=1:numel(E)
	E(ei)=model.factor_types(fti).weights(:,ei)'*factor.data;
end

