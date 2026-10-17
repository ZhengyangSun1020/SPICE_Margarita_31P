% Export the AMARES fit of the hybrid MID00283 file to a plain .mat for Python (arrays are [i j rep] in the file's (col,row) layout)
base = 'F:\Chichen_ConceptB1\AliceConceptB1\CONCEPT\CONCEPT_2D_3D_recon_15112024';
tb = fullfile(base, 'CONCEPT_code_recon\RodgersSpectroToolsV2Dev\main_Ferenc_pre01112022');
addpath(tb); addpath(genpath(fullfile(tb, 'SyntheticSpec'))); addpath(genpath(fullfile(tb, 'Utils'))); warning('off', 'all');
S = load(fullfile(base, 'SpecObj_mat', 'meas_MID00283_FID221488_200FOV_10x10_fitted.mat')); F = S.meas_MID00283_FID221488_200FOV_10x10_fitted;
pkn = F.pk.initialValues; disp(fieldnames(pkn)'); disp(struct2table(pkn));
idx = find(~cellfun(@isempty, F.fit), 1); x = F.fit{idx}; fprintf('fit cell: %s %s\n', class(x), mat2str(size(x))); if isstruct(x), disp(fieldnames(x)'); end
fprintf('fitStatus example: '); y = F.fitStatus(~cellfun(@isempty, F.fitStatus)); if ~isempty(y), disp(y{1}); else, disp('all empty'); end
A = F.ApodSpectrum; n = numel(A{find(~cellfun(@isempty, A), 1)}); spec = complex(nan(n, 10, 10, 52));
for k = 1:numel(A), if ~isempty(A{k}), [i, j, r] = ind2sub(size(A), k); spec(:, i, j, r) = A{k}; end, end
fitspec = complex(nan(n, 10, 10, 52)); fit_ok = false;
for k = 1:numel(F.fit)
    z = F.fit{k}; if isempty(z), continue; end; [i, j, r] = ind2sub(size(F.fit), k);
    if isnumeric(z) && numel(z) == n, fitspec(:, i, j, r) = z; fit_ok = true;
    elseif isstruct(z), ff = fieldnames(z); for q = 1:numel(ff), zz = z.(ff{q}); if isnumeric(zz) && numel(zz) == n, fitspec(:, i, j, r) = zz(:); fit_ok = true; break; end; end
    end
end
fprintf('fit model spectra exported: %d\n', fit_ok);
FS = F.fitStatus; sz = size(FS); exitf = nan(sz); relnorm = nan(sz); resid = complex(nan(n, sz(1), sz(2), sz(3))); infid = complex(nan(n, sz(1), sz(2), sz(3))); xfit = nan(22, sz(1), sz(2), sz(3));
for k = 1:numel(FS)
    z = FS{k}; if isempty(z), continue; end; [i, j, r] = ind2sub(sz, k);
    exitf(k) = z.EXITFLAG; relnorm(k) = z.relativeNorm; resid(:, i, j, r) = z.residual; infid(:, i, j, r) = z.inputFid; xfit(1:numel(z.xFit), i, j, r) = z.xFit;
end
fprintf('fitStatus filled: %d of %d; EXITFLAG counts: ', nnz(~isnan(exitf)), numel(exitf)); u = unique(exitf(~isnan(exitf))); for q = u', fprintf('%d:%d  ', q, nnz(exitf == q)); end; fprintf('\n');
nanlike = nan(size(F.PCr.Ampl)); getf = @(s, f) local_get(F, s, f, nanlike);
fprintf('info: '); disp(F.info); fprintf('PK peaks: %s\n', strjoin(cellfun(@(c) strjoin(cellstr(c), '/'), {pkn.peakName}, 'uni', 0), ', '));
z = FS{find(~cellfun(@isempty, FS), 1)}; fprintf('beginTime %.3g ms\n', 1e3 * z.exptParams.beginTime);
out = struct('exitflag', exitf, 'relnorm', relnorm, 'resid', resid, 'infid', infid, 'xfit', xfit,'PCr', F.PCr.Ampl, 'PCr_CRLB', F.PCr.CRLB, 'PCr_LW', F.PCr.LW, 'PCr_CS', F.PCr.CS, 'Pi', F.Pi.Ampl, 'Pi_CS', F.Pi.CS, 'Pi_LW', F.Pi.LW, ...
    'Pi2', getf('Pi2', 'Ampl'), 'Pi2_CS', getf('Pi2', 'CS'), 'ATP', F.ATP.Ampl, 'ATP_LW', F.ATP.LW, 'PDE', getf('PDE', 'Ampl'), 'Pi_tot', getf('Pi_tot', 'Ampl'), 'pH', F.pH, 'pH2', local_get2(F, 'pH2', nanlike), 'pH_mean', local_get2(F, 'pH_mean', nanlike), ...
    'Tau', F.Tau, 'Tau_R2', F.Tau_R2, 'spec', spec, 'fitspec', fitspec, 'rest', F.info.rest, 'ee', F.info.ee, 'rec', F.info.rec, 'tRep', F.info.tRep);
save('E:\claude_tmp\study\amares\fitted_MID00283_hybrid.mat', '-struct', 'out', '-v7'); disp('saved');
function v = local_get(F, s, f, d)
if isfield(F, s) && isfield(F.(s), f), v = F.(s).(f); else, v = d; end
end
function v = local_get2(F, s, d)
if isfield(F, s), v = F.(s); else, v = d; end
end
