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
% PyNLO-style plots (matches simple_ydfa.py -> sim.plot("wvl"))
% ------------------------------------------------------------------------
run('units.m');   % defines c, ps, mm, um, etc.

% =========================
% (A) Power evolution + N2
% =========================
z = sys.z(:);                            % [m]
P_sig = signal.Powerz(:);                % [W]

if isfield(pumps,'Powerz')
    P_pump = sum(pumps.Powerz,2);        % [W] total pump
else
    P_pump = zeros(size(P_sig));
end

Nmin = min([numel(z), numel(P_sig), numel(P_pump)]);
z = z(1:Nmin);  P_sig = P_sig(1:Nmin);  P_pump = P_pump(1:Nmin);

N2 = [];
zN2 = [];
if isfield(fiber,'N2z')
    N2 = fiber.N2z(:);
    N2 = N2(1:min(numel(N2), numel(z)-1));
    zN2 = z(2:numel(N2)+1);
end

figure('Name','power evolution','Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot(z, P_pump, 'LineWidth', 2); hold on;
plot(z, P_sig,  'LineWidth', 2);
grid on; xlabel('position (m)'); ylabel('power (W)');
legend('pump','signal','Location','best');

nexttile;
if ~isempty(N2)
    plot(zN2, N2, 'LineWidth', 2);
    grid on; xlabel('position (m)'); ylabel('population inversion');
else
    text(0.1,0.5,'fiber.N2z not found'); axis off;
end

% =========================================
% (B) Spectral + temporal evolution (PyNLO)
% =========================================
% Uses:
%   Time:  signal.hz_3d, signal.vz_3d vs sys.t
%   Freq:  signal.Hz_3d, signal.Vz_3d vs sys.w  (rad/s, relative)
%   Carrier: sys.lambda0

if ~(isfield(signal,'Hz_3d') && isfield(signal,'hz_3d'))
    warning('Missing signal.Hz_3d/hz_3d arrays. Cannot make PyNLO-style evolution plot.');
    return;
end

% --- grids ---
t_ps = (sys.t(:) / ps).';                                     % [ps], row
omega_abs = fftshift(sys.w(:)) + 2*pi*c./sys.lambda0;         % [rad/s], col
lam_um = (2*pi*c ./ omega_abs) / um;                          % [um], col
z_mm = (sys.z(:) / mm);                                       % [mm], col

% --- time-domain power (Nz x Nt) ---
P_t = abs(signal.hz_3d).^2 + abs(signal.vz_3d).^2;
P_t_dB = 10*log10(P_t + eps);
P_t_dB = P_t_dB - max(P_t_dB(:));

% --- frequency-domain power (Nz x Nw), shift to match fftshift(sys.w) ---
P_w = abs(signal.Hz_3d).^2 + abs(signal.Vz_3d).^2;
P_w = fftshift(P_w, 2);

% Convert to "per wavelength" like PyNLO: p_wl = |A|^2 * dv/dλ (here ω-grid -> use |dω/dλ|)
% ω = 2πc/λ  => |dω/dλ| = 2πc/λ^2
domega_dlambda = (2*pi*c) ./ ((lam_um*um).^2);                % [rad/s per m], col
P_lam = P_w .* (domega_dlambda.' + 0);                        % broadcast across z

P_lam_dB = 10*log10(P_lam + eps);
P_lam_dB = P_lam_dB - max(P_lam_dB(:));

% --- match lengths (just in case) ---
Nz = min([size(P_t_dB,1), size(P_lam_dB,1), numel(z_mm)]);
Nt = min([size(P_t_dB,2), numel(t_ps)]);
Nw = min([size(P_lam_dB,2), numel(lam_um)]);

z_mm = z_mm(1:Nz);
P_t_dB   = P_t_dB(1:Nz, 1:Nt);
P_lam_dB = P_lam_dB(1:Nz, 1:Nw);
t_ps = t_ps(1:Nt);
lam_um = lam_um(1:Nw);

% --- figure layout like PyNLO plot_results() ---
figure('Name','spectral & temporal evolution (PyNLO style)','Color','w');
ax0 = subplot(3,2,1);  % spectrum in/out
ax1 = subplot(3,2,2);  % time in/out
ax2 = subplot(3,2,[3 5]);  % spectral evolution
ax3 = subplot(3,2,[4 6]);  % temporal evolution

% Top-left: spectrum in/out (blue=input, green=output)
axes(ax0);
plot(lam_um, P_lam_dB(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(lam_um, P_lam_dB(end,:), 'g', 'LineWidth', 1.5);
grid on; ylabel('Power (dB)'); ylim([-50 10]);
set(gca,'XDir','reverse');  % PyNLO: invert_xaxis()
xlabel('wavelength (\mum)');

% Top-right: time in/out (blue=input, green=output)
axes(ax1);
plot(t_ps, P_t_dB(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(t_ps, P_t_dB(end,:), 'g', 'LineWidth', 1.5);
grid on; ylabel('Power (dB)'); ylim([-50 10]);
xlabel('Time (ps)');

% Bottom-left: spectral evolution heatmap (vmin=-40, vmax=0)
axes(ax2);
imagesc(lam_um, z_mm, P_lam_dB);
axis xy; grid on;
set(gca,'XDir','reverse');  % PyNLO: invert_xaxis()
caxis([-40 0]);
xlabel('wavelength (\mum)');
ylabel('Propagation Distance (mm)');

% Bottom-right: temporal evolution heatmap (vmin=-40, vmax=0)
axes(ax3);
imagesc(t_ps, z_mm, P_t_dB);
axis xy; grid on;
caxis([-40 0]);
xlabel('Time (ps)');
ylabel('Propagation Distance (mm)');

% Optional: apply a decent colormap
colormap(turbo);



rmpath(strcat(cd,'/Plotting Function'))