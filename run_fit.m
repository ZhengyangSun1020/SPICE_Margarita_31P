%% run_fit.m — run the OXSA/AMARES leg pipeline on SPICE-reconstructed data
%  Requires: OXSA toolbox on the MATLAB path (AMARES.*), and Leg_eval_twix.m.
%  Input:    cells_MID00644_r10.mat  (built by scripts/pack_cells_for_oxsa.py)

clear; close all;

% --- point these at your paths ---------------------------------------------
addpath(genpath('C:\path\to\OXSA'));                 % <-- EDIT: OXSA toolbox
addpath('F:\Chichen_ConceptB1\AliceConceptB1\CONCEPT\CONCEPT_2D_3D_recon_15112024');  % Leg_eval_twix.m
load('cells_MID00644_r10.mat', 'cells');             % 1x52 cell of nested structs

% --- protocol / phase durations --------------------------------------------
% 52 acquisitions @ 14 s each = 728 s = 12.1 min.  Default 1-5-6 min protocol.
% duration = # acquisitions per phase and MUST sum to numel(cells) (=52).
% [4 22 26] => rest reps 1-4, exercise reps 5-26, recovery reps 27-52
% (consistent with the Pi/PCr peak observed at rep ~25 = end of exercise).
protocol = [1 5 6];          % minutes: rest / exercise / recovery
duration = [4 22 26];        % <-- EDIT if your protocol timing differs

assert(sum(duration) == numel(cells), 'duration must sum to nAcq (%d)', numel(cells));

% --- run the fit -----------------------------------------------------------
[obj, RESULTS, INFO, Volunteer_info] = Leg_eval_twix(cells, protocol, duration);

% --- save with your normal naming convention -------------------------------
twixPath2D = 'F:\AliceConceptB1\in_vivo_3T\dynamic_crt_10_10\meas_MID00644_FID214692_LV_crt_dynamic_10x10_5rings.dat';
[~, baseName, ~] = fileparts(twixPath2D);
saveName = ['SPICE_R10_' baseName];        % prefix marks the SPICE rank-10 fit
savePath = 'F:\Chichen_ConceptB1\AliceConceptB1\CONCEPT\CONCEPT_2D_3D_recon_15112024\SpecObj_mat';
if ~exist(savePath,'dir'); mkdir(savePath); end
save(fullfile(savePath, saveName), 'obj', 'RESULTS', 'INFO', 'Volunteer_info', '-v7.3');
fprintf('Done. Saved %s\n', fullfile(savePath, [saveName '.mat']));

% --- quick sanity check on the first exercise voxel fit --------------------
% Expect PCr chemShift ~0, Pi ~4.8, ATP ~-16 (gamma).  If ATP comes out near
% +16, the FID is mirrored -> set CONJ_FID=False in pack_cells_for_oxsa.py and
% re-pack.  Voxel (ix,iy)=(xi,yi); e.g. voxel id 55 -> ix=6,iy=6 (1-indexed).
acq = 25;  ix = 6;  iy = 6;
if ~isempty(obj.fit{ix,iy,acq})
    fprintf('acq %d vox(%d,%d): PCr.CS=%.2f  Pi.CS=%.2f  ATP.CS=%.2f  pH=%.2f\n', ...
        acq, ix, iy, obj.PCr.CS(ix,iy,acq), obj.Pi.CS(ix,iy,acq), ...
        obj.ATP.CS(ix,iy,acq), obj.pH(ix,iy,acq));
end
