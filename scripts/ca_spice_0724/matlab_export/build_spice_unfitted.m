function build_spice_unfitted(ds, varargin)
% Build an "unfitted"-format file (obj_dynamic: 1 x nRep cell, each .data.spec = SynthSpec) from our SPICE recon, for
% Leg_eval_twix.m-style fitting. Header/geometry/scan info is taken from the scan's own MATLAB recon object (format
% only); the signals are our SPICE FIDs (variant D, soft boundary), PCr referenced to 0 ppm.
% Usage: build_spice_unfitted('283')
%        build_spice_unfitted('644', 'template', <MATLAB recon .mat>, 'input', <MID00644_SPICE.mat>, 'outdir', <dir>, 'layout', 'matlab')
% layout: 'display' = signals(:,row,col), row 1 = top of localizer (paper figures);
%         'matlab'  = signals(:,col,row) = transpose of display, the array layout of reconUsingDFT_2D_splitCoils_VE_dynamic
%                     (verified on MID00644: MATLAB array = transpose of our display grid), shown upright by quickViewDynamicCRT / PlotCsi.
base = 'F:\Chichen_ConceptB1\AliceConceptB1\CONCEPT\CONCEPT_2D_3D_recon_15112024';
tb = fullfile(base, 'CONCEPT_code_recon\RodgersSpectroToolsV2Dev\main_Ferenc_pre01112022');
addpath(tb); addpath(genpath(fullfile(tb, 'SyntheticSpec'))); addpath(genpath(fullfile(tb, 'Utils'))); warning('off', 'all');
p = inputParser; p.addParameter('template', ''); p.addParameter('input', ''); p.addParameter('outdir', 'E:\claude_tmp\study\recon_for_matlab');
p.addParameter('layout', 'display'); p.addParameter('prefix', 'SPICE'); p.parse(varargin{:}); opt = p.Results;   % prefix: output name / coil string, e.g. 'hybrid'
% The F: file named MID00277 is a copy of MID00279 (8x8, 30 mm, 72 reps), so MID00277 borrows the MID00279 header
% (same session: TR, FA, f0, patient, localizer refs) with its own geometry: 10x10, 24 mm, same 240 mm FOV and centre.
tds = ds; if strcmp(ds, '277'), tds = '279'; end
if isempty(opt.template)
    tpl = dir(fullfile(base, 'SpecObj_mat', sprintf('unfitted_meas_MID00%s_*.mat', tds))); tplPath = fullfile(tpl(1).folder, tpl(1).name); outName = tpl(1).name(10:end);
else
    tplPath = opt.template; outName = sprintf('meas_MID00%s_SPICE.mat', ds);
end
if isempty(opt.input), opt.input = sprintf('E:\\claude_tmp\\study\\recon_for_matlab\\MID00%s_SPICE.mat', ds); end
T = load(tplPath); fnT = fieldnames(T); o = T.(fnT{1}); nRepT = numel(o);
E = load(opt.input);  % fid(:,row,col,rep), row 1 = top of localizer
nRep = size(E.fid, 4); N = size(E.fid, 2); sp0 = o{1}.data.spec;
geo = struct('rows', sp0.rows, 'columns', sp0.columns, 'pixelSpacing', sp0.pixelSpacing, 'imagePositionPatient', sp0.imagePositionPatient);
if ~strcmp(tds, ds)
    ps = 240 / N; d = reshape(sp0.imageOrientationPatient(1:6), 3, 2); dpos = (ps - sp0.pixelSpacing(1)) / 2 * (d(:, 1) + d(:, 2));
    geo = struct('rows', N, 'columns', N, 'pixelSpacing', [ps; ps], ...
        'imagePositionPatient', reshape(sp0.imagePositionPatient(:) + dpos, size(sp0.imagePositionPatient)));
    outName = sprintf('meas_MID00%s_FID221482_240FOV_10x10_5rings.mat', ds);
    fprintf('MID00%s: header from MID00%s, geometry -> %dx%d, %.0f mm, imagePositionPatient %s\n', ds, tds, N, N, ps, mat2str(geo.imagePositionPatient(:)', 5));
else
    assert(nRep == nRepT, 'rep count differs: ours %d, template %d', nRep, nRepT);
end
mk = @(F, spT) struct('signals', {{F}}, 'dwellTime', spT.dwellTime, 'imagingFrequency', spT.imagingFrequency, ...
    'columns', geo.columns, 'rows', geo.rows, 'slices', spT.slices, 'pixelSpacing', geo.pixelSpacing, ...
    'sliceThickness', spT.sliceThickness, 'imageOrientationPatient', spT.imageOrientationPatient, ...
    'sliceNormal', spT.sliceNormal, 'imagePositionPatient', geo.imagePositionPatient, ...
    'coilStrings', {{opt.prefix}}, 'twixIdentifiers', spT.twixIdentifiers, 'info', {spT.info(1)}, ...
    'coilCombination', [opt.prefix ' (single combined signal)']);
evalc('s1 = SynthSpec(''externalReconStruct'', mk(zeros(size(E.fid, 1), N, N), sp0));');   % axes as the class defines them
t = s1.timeAxis(:); ppm = s1.ppmAxis(:);
if numel(t) ~= numel(sp0.timeAxis), fprintf('MID00%s: FID length %d (header %d)\n', ds, numel(t), numel(sp0.timeAxis)); end
% --- orientation: 'display' keeps the paper-figure layout; 'matlab' transposes to signals(:,col,row) ---
if strcmp(opt.layout, 'matlab'), orient = @(F) permute(F, [1 3 2]); orientTxt = 'signals(:,col,row) = transpose of the display grid: MATLAB reconUsingDFT layout, upright in quickViewDynamicCRT/PlotCsi';
else, orient = @(F) F; orientTxt = 'signals(:,row,col): row 1 = top of localizer (anterior), col 1 = image left; same as paper figures'; end
% --- PCr reference: leg-summed rest spectrum peak -> 0 ppm ---
leg = orient(permute(double(E.leg_phantom_mask == 1), [3 1 2])); leg = squeeze(leg) > 0;
rest = double(E.rest_reps(:)'); acc = 0;
for r = rest, S = abs(specFft(orient(double(E.fid(:, :, :, r))))); acc = acc + sum(S(:, leg), 2); end
w = find(abs(ppm) < 2); [~, j] = max(acc(w)); j = w(j); a = acc(j - 1); b = acc(j); c = acc(j + 1);
p0 = ppm(j) + 0.5 * (a - c) / (a - 2 * b + c) * (ppm(2) - ppm(1)); dHz = -p0 * sp0.imagingFrequency;
fprintf('MID00%s: PCr found at %+.3f ppm -> shifted by %+.2f Hz to 0 ppm\n', ds, p0, dHz);
% --- build new objects ---
obj_dynamic = cell(1, nRep);
for r = 1:nRep
    rT = min(r, nRepT); spT = o{rT}.data.spec; F = orient(double(E.fid(:, :, :, r))) .* exp(2i * pi * dHz * t);
    ers = mk(F, spT);
    if ~strcmp(tds, ds), ers.info{1} = local_retitle(ers.info{1}, '240FOV_8x8_5rings', '240FOV_10x10_5rings'); end
    evalc('synsp = SynthSpec(''externalReconStruct'', ers);');
    d = o{rT}.data; d.spec = synsp;
    obj_dynamic{r} = struct('data', d, 'title', sprintf('%s MID00%s rep %d', opt.prefix, ds, r));
end
spice_info = struct('source', [opt.prefix ' recon (Python), exported from ' opt.input], 'method', E.method, ...
    'spice_lambda', E.spice_lambda, 'spice_rank', E.spice_rank, 'pcr_reference_ppm_before', p0, ...
    'orientation', orientTxt, ...
    'template_header_from', tplPath, 'muscle_mask', orient(permute(double(E.muscle_mask), [3 1 2])), ...
    'leg_phantom_mask', orient(permute(double(E.leg_phantom_mask), [3 1 2])), 'mask_labels', E.mask_labels, ...
    'rest_reps', E.rest_reps, 'exercise_reps', E.exercise_reps, 'recovery_reps', E.recovery_reps);
spice_info.muscle_mask = squeeze(spice_info.muscle_mask); spice_info.leg_phantom_mask = squeeze(spice_info.leg_phantom_mask);
if isfield(E, 'gm_voxels_rowcol')                                           % voxel lists as indices (i, j) into signals{1}(:, i, j) of THIS layout
    if strcmp(opt.layout, 'matlab'), sw = @(x) double(x(:, [2 1])); else, sw = @(x) double(x); end
    spice_info.gm_voxels = sw(E.gm_voxels_rowcol); spice_info.soleus_voxels = sw(E.soleus_voxels_rowcol);
    spice_info.voxel_index_note = 'gm_voxels / soleus_voxels: [i j] indices into signals{1}(:, i, j)';
end
if isfield(E, 'hybrid_mu'), spice_info.hybrid_mu = E.hybrid_mu; spice_info.scale_to_nufft = E.scale_to_nufft; end
out = fullfile(opt.outdir, ['unfitted_' opt.prefix '_' outName]);
save(out, 'obj_dynamic', 'spice_info', '-v7.3');
% --- self-check: reload, PCr position and signal size as Leg_eval_twix would see it ---
C = load(out); s = C.obj_dynamic{1}.data.spec; x = s.signals{1, 1}; Sp = abs(specFft(x));
[~, iq] = max(sum(Sp(:, :), 2)); fprintf('MID00%s: saved %s\n   signals{1,1} %s, dwell %.3g s, f0 %.4f MHz, summed-spectrum max at %.3f ppm, TR %g, FA %g\n', ...
    ds, out, mat2str(size(x)), s.dwellTime, s.imagingFrequency, s.ppmAxis(iq), s.info{1}.csa.MrPhoenixProtocol.alTR{1, 1}, s.info{1}.csa.MrPhoenixProtocol.adFlipAngleDegree{1, 1});
% layout check against the template's own signals (only meaningful when the template holds a real, valid recon)
lb = exp(-pi * 10 * s.timeAxis(:)); pm = @(X) squeeze(max(abs(specFft(double(X) .* lb)), [], 1));
mT = 0; mS = 0; nOk = 0;
for r = rest
    xt = double(o{r}.data.spec.signals{end}); if ~all(isfinite(xt(:))) || max(abs(xt(:))) > 1, continue; end
    mT = mT + pm(xt); mS = mS + pm(C.obj_dynamic{r}.data.spec.signals{1}); nOk = nOk + 1;
end
if nOk, c = corrcoef(mT(:), mS(:)); ct = corrcoef(mT(:), reshape(mS.', [], 1));
    fprintf('   rest max-|S| map vs template recon (%d valid reps): r = %.3f as saved, r = %.3f if transposed\n', nOk, c(1, 2), ct(1, 2));
else, fprintf('   template signals invalid -> layout not checkable against it\n'); end
end

function s = local_retitle(s, a, b)
% replace protocol-name strings in the top-level header fields (MID00277 borrows the MID00279 header)
for f = {'SeriesDescription', 'ProtocolName'}, if isfield(s, f{1}) && ischar(s.(f{1})), s.(f{1}) = strrep(s.(f{1}), a, b); end, end
end

function G = local_orient(F, k, fl)
% F: [nT rows cols] in our display orientation -> template array orientation
G = permute(F, [2 3 1]); G = rot90(G, k); if fl, G = fliplr(G); end; G = permute(G, [3 1 2]);
end
