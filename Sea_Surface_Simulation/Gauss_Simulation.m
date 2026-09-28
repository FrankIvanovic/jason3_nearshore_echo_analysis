%% Gauss_Simulation.m
% =========================================================================
%  Simulation of a 2-D random rough surface with a GAUSSIAN spectrum.
%  -------------------------------------------------------------------------
%  Reference (mathematical basis):
%     "基于电磁散射特性的雷达高度计回波仿真与分", Sec. 2.1.2, eqs. (2.4)-(2.13),
%     pp. 17-19.  (parameters identical to Fig. 2.2 of that thesis)
%  Method:
%     Linear-filtering (FFT) method.  Complex Gaussian white noise is filtered
%     in the spatial-frequency domain by the square root of the power spectrum
%     and inverse-FFT'd; twice its real part gives the rough-surface height.
%  Parameters (same as the reference, Fig. 2.2):
%        incident frequency   f  = 1 GHz      ->  wavelength lambda = c/f
%        correlation length   lx = ly = 1.5*lambda
%        RMS height           sigma = 0.1*lambda
%        sampling interval    dx = dy = lambda/20
%        surface length       Lx = Ly = 20*lx   (satisfies S > 15l*15l)
%  Output: Z(x,y) - 2-D Gaussian-spectrum rough-surface height matrix,
%          plus a 3-D surface plot and an autocorrelation check.
%
%  NOTE: comments/labels are in ASCII because this MATLAB session reads source
%  files as the OS locale (GBK) and rejects non-ASCII bytes. Greek letters in
%  plot text are written with TeX escapes (\sigma, \lambda, ...) which are
%  ASCII in source and render as Greek in figures.
% =========================================================================

clear; clc; close all;
rng(2026);                          % fixed seed for reproducibility

%% ------ 1. Physical and geometric parameters ---------------------------
c      = 3e8;                       % speed of light (m/s)
f      = 1e9;                       % incident frequency 1 GHz
lambda = c / f;                     % wavelength = 0.3 m

lx     = 1.5 * lambda;              % correlation length in x
ly     = 1.5 * lambda;              % correlation length in y
sigma  = 0.1 * lambda;              % RMS (root-mean-square) height

dx     = lambda / 20;               % sampling interval in x
dy     = lambda / 20;               % sampling interval in y

Lx     = 20 * lx;                   % surface length in x = 20*lx
Ly     = 20 * ly;                   % surface length in y = 20*ly

%% ------ 2. Sampling grid ----------------------------------------------
%   Number of samples N = L / d; force even for an efficient FFT.
Nx = round(Lx / dx);
Ny = round(Ly / dy);
Nx = Nx + mod(Nx, 2);
Ny = Ny + mod(Ny, 2);

fprintf('wavelength      lambda = %.3f m\n', lambda);
fprintf('correlation len lx=ly  = %.3f m\n', lx);
fprintf('RMS height      sigma  = %.4f m  (= %.3f*lambda)\n', sigma, sigma/lambda);
fprintf('sample step     dx=dy  = %.4f m  (= lambda/%.0f)\n', dx, lambda/dx);
fprintf('surface size    Lx=Ly  = %.2f m  (= %.0f*lx)\n', Lx, Lx/lx);
fprintf('sample counts   Nx=Ny  = %d  (%dx%d grid)\n\n', Nx, Nx, Ny);

%% ------ 3. Spatial-frequency grid (FFT order) --------------------------
%   Angular spatial frequency K (rad/m).  K = 2*pi*m / L, m in fft order:
%   0,1,...,N/2-1, -N/2,...,-1
dkx = 2*pi / Lx;
dky = 2*pi / Ly;
kx  = dkx * [0:Nx/2-1, -Nx/2:-1];
ky  = dky * [0:Ny/2-1, -Ny/2:-1];
[KX, KY] = meshgrid(kx, ky);        % KX, KY are Ny x Nx

%% ------ 4. 2-D Gaussian power spectrum W(Kx,Ky)  -- eq. (2.13) ---------
%   W(Kx,Ky) = sigma^2 * lx*ly / (4*pi) * exp[-(Kx^2*lx^2 + Ky^2*ly^2)/4]
W = sigma^2 * lx * ly / (4*pi) .* exp(-(KX.^2 .* lx.^2 + KY.^2 .* ly.^2) / 4);

%% ------ 5. Spectral-domain filter coefficients (linear filtering) -------
%   Let b(m,n) = H(m,n) .* gamma_hat(m,n), where gamma_hat is unit-variance
%   complex Gaussian white noise.  H is normalized so that the variance of
%   Z = 2*Re{ifft2(b)} equals sigma^2 exactly:
%       E[Z^2] = sum( W * dkx*dky ) = sigma^2        (discrete Parseval)
%   This corresponds to eqs. (2.6) and (2.9) of the reference, but using the
%   standard ifft2 normalization so the output statistics match the spectrum.
H = (Nx*Ny / sqrt(2)) * sqrt(W .* dkx .* dky);

% Complex Gaussian random numbers gamma_hat = (a + j*b)/sqrt(2),
% a,b ~ N(0,1), so that E[|gamma_hat|^2] = 1.   [cf. eq. (2.6)]
gamma_hat = (randn(Ny, Nx) + 1j*randn(Ny, Nx)) / sqrt(2);

%% ------ 6. Inverse FFT -> rough-surface height  -- eqs. (2.10)(2.11) ----
%   zeta_hat(p,q) = IFFT2{ b(m,n) }              -- eq. (2.10)
%   Z(p,q)        = 2 * Re{ zeta_hat(p,q) }      -- eq. (2.11)
b        = H .* gamma_hat;
zeta_hat = ifft2(b);                % complex in general
Z        = 2 * real(zeta_hat);      % take real part and double -> real surface

% coordinate axes (m)
x = (0:Nx-1) * dx;
y = (0:Ny-1) * dy;

%% ------ 7. Statistical checks ------------------------------------------
%   (a) RMS height
sigma_emp = std(Z(:), 1);
fprintf('--- statistical checks ---\n');
fprintf('RMS height  theory sigma   = %.5f m\n', sigma);
fprintf('RMS height  sim.  std(Z)   = %.5f m   (ratio %.3f)\n', sigma_emp, sigma_emp/sigma);

%   (b) correlation length: normalized autocorrelation along x, find 1/e point.
%       Theory: R(tau)/R(0) = exp(-tau^2/lx^2), so R = 1/e at tau = lx.
Zc  = Z - mean(Z(:));
R2D = real(fftshift(ifft2(abs(fft2(Zc)).^2)));    % 2-D autocorrelation
cx  = Ny/2 + 1;                                    % center row (cut at y = 0)
rxx = R2D(cx, Nx/2+1:end);                         % along +x (zero lag -> +x)
rxx = rxx / rxx(1);
idx = find(rxx < exp(-1), 1, 'first');
if ~isempty(idx)
    lx_emp = (idx-1) * dx;
    fprintf('corr. length theory lx   = %.5f m\n', lx);
    fprintf('corr. length sim.  lx_emp= %.5f m   (ratio %.3f)\n', lx_emp, lx_emp/lx);
end
fprintf('surface mean  mean(Z)      = %.2e (should be ~0)\n\n', mean(Z(:)));

%% ------ 8. Plots -------------------------------------------------------
%   Fig 1: 3-D rough-surface plot (counterpart of Fig. 2.2 in the reference).
%          The grid is dense, so subsample for display.
step = 4;
Xs = x(1:step:end);
Ys = y(1:step:end);
Zs = Z(1:step:end, 1:step:end);

figure('Name','2-D Gaussian-spectrum rough surface','Position',[100 100 760 560]);
surf(Xs, Ys, Zs, 'EdgeColor','none');
colorbar; colormap(jet);
xlabel('x (m)'); ylabel('y (m)'); zlabel('height z (m)');
title(sprintf('2-D Gaussian-spectrum rough surface  (\\sigma=%.3f\\lambda, l=%.2f\\lambda, L=20l, %dx%d)', ...
              sigma/lambda, lx/lambda, Nx, Ny));
view(-37.5, 30); grid on; axis tight;

%   Fig 2: normalized autocorrelation along x - simulation vs theory
figure('Name','autocorrelation check','Position',[900 200 640 460]);
tau = (0:numel(rxx)-1) * dx;
plot(tau, rxx, 'b-', 'LineWidth', 1.8); hold on;
plot(tau, exp(-(tau/lx).^2), 'r--', 'LineWidth', 1.8);
xlabel('\tau_x (m)'); ylabel('R(\tau_x,0) / R(0)');
title('normalized autocorrelation along x: simulation vs theory');
legend('simulated autocorrelation', 'theory  exp(-\tau^2/l^2)', 'Location','northeast');
grid on; xlim([0 Lx/2]);

fprintf('Simulation done. Two figures generated.  (math: reference pp.17-19, eqs. 2.4-2.13)\n');
