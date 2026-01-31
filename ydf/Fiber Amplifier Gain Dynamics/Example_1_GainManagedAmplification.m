clear all;
% Check to make sure script is working by matching Chen et. al.

run('units.m')
baseDir = fileparts(mfilename('fullpath'));
addpath(fullfile(baseDir,'Utils'));
addpath(fullfile(baseDir,'Plotting Function'));

%% Check doping concentration to Match the absorption at 976 and 920 given the commercial data of yb1200 6/125;
sys = struct; signal = struct; fiber = struct; pumps = struct;
fiber.Type = 'gain'; fiber.RareEarthDopant = 'yb'; fiber.Lz = 5*m;
fiber.DoubleCladded = 'yes'; 
fiber.NT = 9.6e25; % Match with Liekki Absorption
% fiber.NT = 25e25; % Match with RP fiber Power
% fiber.NT = 1e25; % Match with Chen paper
% Matching with 976;
sys.PlotResults = 'yes';
signal.InterCavityPower = 0;
signal.E0 = 1.75*nJ;
pumps.lambda = [976,976]*nm; pumps.Direction = {'f','b'}; pumps.Power0 = [1,0]*W;
sys.WinT = 17.5*ps; sys.nt = 2^12;
dz = 0.001;
[sys, signal, fiber, pumps] = Fiber_Prop_Lz_v1(sys, signal, fiber, pumps, dz);


options.units = 'gain';
options.ymin = 0;
options.ymax = 50;
sys = plot_FiberPropResults(sys,signal,pumps,fiber);
sys = plot_PowerOutput(sys,signal,pumps,fiber,options);
fig1 = figure(sys.fignum-1);

%% ------------------------------------------------------------------------
%  Correct PyNLO-style plots + pump depletion + inversion
%  (Fixes wavelength flipping by sorting wl and permuting spectrum to match)
% ------------------------------------------------------------------------
run('units.m');

%% =========================
% 1) Pump depletion + signal + N2
%% =========================
z_m = sys.z(:);

P_sig = signal.Powerz(:);
if isfield(pumps,'Powerz')
    P_pump = sum(pumps.Powerz,2);     % total pump power (all channels)
else
    P_pump = zeros(size(P_sig));
end

Nmin = min([numel(z_m), numel(P_sig), numel(P_pump)]);
z_m    = z_m(1:Nmin);
P_sig  = P_sig(1:Nmin);
P_pump = P_pump(1:Nmin);

N2 = [];
zN2_m = [];
if isfield(fiber,'N2z')
    N2 = fiber.N2z(:);
    N2 = N2(1:min(numel(N2), numel(z_m)-1));
    zN2_m = z_m(2:numel(N2)+1);
end

figure('Color','w','Name','pump depletion + inversion');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(z_m, P_pump, 'LineWidth', 2); hold on;
plot(z_m, P_sig,  'LineWidth', 2);
grid on;
xlabel('position (m)');
ylabel('power (W)');
legend('pump','signal','Location','best');

nexttile;
if ~isempty(N2)
    plot(zN2_m, N2, 'LineWidth', 2);
    grid on;
    xlabel('position (m)');
    ylabel('population inversion');
else
    text(0.1,0.5,'fiber.N2z not found','FontSize',12); axis off;
end

%% ------------------------------------------------------------------------
% PyNLO-style plots (linear + dB), robust wavelength axis (no weird flips)
% Paste at end of Example_1_GainManagedAmplification.m (replace old block)
% ------------------------------------------------------------------------
run('units.m');   % defines c, ps, mm, um, etc.

db_floor = -40;       % like PyNLO vmin
do_norm  = true;      % normalize so max = 1 (linear) or 0 dB (dB)

% =========================
% (A) Power evolution + N2
% =========================
z_m = sys.z(:);                              % [m]
P_sig = signal.Powerz(:);                    % [W]
if isfield(pumps,'Powerz')
    P_pump = sum(pumps.Powerz,2);            % [W] total pump
else
    P_pump = zeros(size(P_sig));
end

Nmin = min([numel(z_m), numel(P_sig), numel(P_pump)]);
z_m   = z_m(1:Nmin);
P_sig = P_sig(1:Nmin);
P_pump= P_pump(1:Nmin);

N2 = []; zN2_m = [];
if isfield(fiber,'N2z')
    N2 = fiber.N2z(:);
    N2 = N2(1:min(numel(N2), numel(z_m)-1));
    zN2_m = z_m(2:numel(N2)+1);
end

% ---- linear power + N2 (matches your python power plot) ----
figure('Name','power evolution (linear)','Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(z_m, P_pump, 'LineWidth', 2); hold on;
plot(z_m, P_sig,  'LineWidth', 2);
grid on; xlabel('position (m)'); ylabel('power (W)');
legend('pump','signal','Location','best');

nexttile;
if ~isempty(N2)
    plot(zN2_m, N2, 'LineWidth', 2);
    grid on; xlabel('position (m)'); ylabel('population inversion');
else
    text(0.1,0.5,'fiber.N2z not found'); axis off;
end

% ---- OPTIONAL: dB view of pump/signal power (relative to their own max) ----
figure('Name','power evolution (dB)','Color','w');
Pp_db = 10*log10(P_pump + eps); Pp_db = Pp_db - max(Pp_db);
Ps_db = 10*log10(P_sig  + eps); Ps_db = Ps_db - max(Ps_db);

tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(z_m, Pp_db, 'LineWidth', 2); hold on;
plot(z_m, Ps_db, 'LineWidth', 2);
grid on; xlabel('position (m)'); ylabel('power (dB, rel.)');
ylim([db_floor 5]);
legend('pump','signal','Location','best');

nexttile;
if ~isempty(N2)
    plot(zN2_m, N2, 'LineWidth', 2);
    grid on; xlabel('position (m)'); ylabel('population inversion');
else
    text(0.1,0.5,'fiber.N2z not found'); axis off;
end

% =========================================
% (B) Spectral + temporal evolution (dB + linear)
% =========================================
% Requires saved arrays from Fiber_Prop_Lz_v1:
%   signal.hz_3d, signal.vz_3d   (time-domain fields vs sys.t)
%   signal.Hz_3d, signal.Vz_3d   (freq-domain fields vs sys.w)
if ~(isfield(signal,'Hz_3d') && isfield(signal,'hz_3d'))
    warning('Missing signal.Hz_3d/hz_3d arrays. Cannot make evolution plots.');
    return;
end

% ---- grids ----
t_ps = (sys.t(:)/ps).';                % [ps] row
z_mm = (sys.z(:)/mm);                  % [mm] col

% IMPORTANT: sys.w is already fftshift'ed in this codebase (see Fiber_Prop_Lz_v1.m)
omega0   = 2*pi*c./sys.lambda0;        % [rad/s]
omegaAbs = sys.w(:) + omega0;          % [rad/s] col (monotonic around 0)

% Use frequency (Hz) to match PyNLO's dv/dλ conversion
nuAbs    = omegaAbs/(2*pi);            % [Hz]
lam_m    = c ./ nuAbs;                 % [m]
lam_um   = lam_m/um;                   % [um]

% dv/dλ magnitude
dv_dl = c ./ (lam_m.^2);               % [Hz/m]

% ---- powers ----
P_t  = abs(signal.hz_3d).^2 + abs(signal.vz_3d).^2;     % Nz x Nt
P_nu = abs(signal.Hz_3d).^2 + abs(signal.Vz_3d).^2;     % Nz x Nw

% Convert to "per wavelength" like PyNLO: P(λ) = P(ν) * |dν/dλ|
P_lam = P_nu .* (dv_dl.' + 0);                           % Nz x Nw

% ---- enforce monotonic wavelength axis & permute spectrum columns to match ----
% This removes "flipped / scrambled" axes permanently.
[lam_um_s, idxLam] = sort(lam_um, 'ascend');
P_lam = P_lam(:, idxLam);

% ---- match lengths safely ----
Nz = min([size(P_t,1), size(P_lam,1), numel(z_mm)]);
Nt = min([size(P_t,2), numel(t_ps)]);
Nw = min([size(P_lam,2), numel(lam_um_s)]);

z_mm    = z_mm(1:Nz);
t_ps    = t_ps(1:Nt);
lam_um_s= lam_um_s(1:Nw).';

P_t  = P_t(1:Nz, 1:Nt);
P_lam= P_lam(1:Nz, 1:Nw);

% ---- helpers ----
if do_norm
    P_tn   = P_t  ./ (max(P_t(:))  + eps);
    P_lamn = P_lam./ (max(P_lam(:))+ eps);
else
    P_tn   = P_t;
    P_lamn = P_lam;
end

P_t_dB   = 10*log10(P_tn   + eps);  P_t_dB   = P_t_dB   - max(P_t_dB(:));
P_lam_dB = 10*log10(P_lamn + eps);  P_lam_dB = P_lam_dB - max(P_lam_dB(:));

% =========================
% Figure: dB (PyNLO-like)
% =========================
figure('Name','evolution (dB)','Color','w');
ax0 = subplot(3,2,1);
ax1 = subplot(3,2,2);
ax2 = subplot(3,2,[3 5]);
ax3 = subplot(3,2,[4 6]);

% spectra in/out
axes(ax0);
plot(lam_um_s, P_lam_dB(1,:),  'b', 'LineWidth', 1.5); hold on;
plot(lam_um_s, P_lam_dB(end,:), 'g', 'LineWidth', 1.5);
grid on; ylabel('Power (dB)'); ylim([db_floor 10]);
xlabel('wavelength (\mum)');

% time in/out
axes(ax1);
plot(t_ps, P_t_dB(1,:),   'b', 'LineWidth', 1.5); hold on;
plot(t_ps, P_t_dB(end,:), 'g', 'LineWidth', 1.5);
grid on; ylabel('Power (dB)'); ylim([db_floor 10]);
xlabel('Time (ps)');

% spectral evolution heatmap
axes(ax2);
imagesc(lam_um_s, z_mm, P_lam_dB);
axis xy; grid on;
caxis([db_floor 0]);
xlabel('wavelength (\mum)');
ylabel('Propagation Distance (mm)');

% temporal evolution heatmap
axes(ax3);
imagesc(t_ps, z_mm, P_t_dB);
axis xy; grid on;
caxis([db_floor 0]);
xlabel('Time (ps)');
ylabel('Propagation Distance (mm)');

colormap(turbo);

% =========================
% Figure: linear (normalized)
% =========================
figure('Name','evolution (linear)','Color','w');
ax0 = subplot(3,2,1);
ax1 = subplot(3,2,2);
ax2 = subplot(3,2,[3 5]);
ax3 = subplot(3,2,[4 6]);

% spectra in/out
axes(ax0);
plot(lam_um_s, P_lamn(1,:),   'b', 'LineWidth', 1.5); hold on;
plot(lam_um_s, P_lamn(end,:), 'g', 'LineWidth', 1.5);
grid on; 
if do_norm
    ylab_lin = 'Power (norm.)';
else
    ylab_lin = 'Power (linear)';
end
ylabel(ylab_lin);
xlabel('wavelength (\mum)');
if do_norm, ylim([0 1.05]); end

% time in/out
axes(ax1);
plot(t_ps, P_tn(1,:),   'b', 'LineWidth', 1.5); hold on;
plot(t_ps, P_tn(end,:), 'g', 'LineWidth', 1.5);
grid on;
if do_norm
    ylab_lin = 'Power (norm.)';
else
    ylab_lin = 'Power (linear)';
end
ylabel(ylab_lin);
xlabel('Time (ps)');
if do_norm, ylim([0 1.05]); end

% spectral evolution heatmap
axes(ax2);
imagesc(lam_um_s, z_mm, P_lamn);
axis xy; grid on;
if do_norm, caxis([0 1]); end
xlabel('wavelength (\mum)');
ylabel('Propagation Distance (mm)');

% temporal evolution heatmap
axes(ax3);
imagesc(t_ps, z_mm, P_tn);
axis xy; grid on;
if do_norm, caxis([0 1]); end
xlabel('Time (ps)');
ylabel('Propagation Distance (mm)');

colormap(turbo);


rmpath(strcat(cd,'/Plotting Function'))