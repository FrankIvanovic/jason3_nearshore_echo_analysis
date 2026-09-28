# -*- coding: utf-8 -*-
"""PM 谱随机海面生成（FFT 线性滤波法）。

数学基础: 《基于电磁散射特性的雷达高度计回波仿真与分析》
  式(2.14) 一维 PM 谱, 式(2.15) U10 -> U19.5 换算, 式(2.17) 二维方向谱。
数值方法与 Sea_Surface_Simulation/Gauss_Simulation.m 相同（Thorsos 线性滤波法）:
单位方差复高斯白噪声在空间频率域被 sqrt(W) 滤波后逆 FFT, 取实部加倍。

尺度说明（重要）:
  PM 谱是宽谱, 有限网格 (L, dx) 只能表示 k_min=2*pi/L ~ k_max=pi/dx 内的波。
  低风速时 PM 谱能量集中于短重力波, 网格解析率 capture = exp(-a/k_max^2)
  （a = beta*g^2/U19.5^4）急剧下降, 因此本模块默认把海面高度场归一化到
  完整 PM 理论 sigma_h（SWH=4*sigma_h）, 用以补偿未解析波段的方差;
  代价是已解析长波的斜率被同步放大。低风速场景需两尺度分解处理（后续工作）。
"""

import numpy as np

ALPHA = 8.10e-3   # PM 谱经验常数 alpha
BETA = 0.74       # PM 谱经验常数 beta
G = 9.81          # 重力加速度 (m/s^2)


def u19_5_from_u10(u10: float) -> float:
    """10 m 高度风速 -> 19.5 m 高度风速, 论文式(2.15)。"""
    return u10 * (1.0 + np.sqrt(5e-4 * u10**0.5) / 0.4 * np.log(10.0))


def pm_sigma_h(u19_5: float) -> float:
    """一维 PM 谱理论均方根高度 sigma_h = sqrt(alpha*U^4/(4*beta*g^2)) (m)。"""
    return np.sqrt(ALPHA * u19_5**4 / (4.0 * BETA * G**2))


def pm_swh(u19_5: float) -> float:
    """有效波高 SWH = 4 * sigma_h (m)。"""
    return 4.0 * pm_sigma_h(u19_5)


def pm_spectrum_1d(k: np.ndarray, u19_5: float) -> np.ndarray:
    """一维 PM 谱, 论文式(2.14): W(k) = alpha/(2 k^3) exp(-beta g^2/(k^2 U^4))。"""
    with np.errstate(over="ignore", divide="ignore"):
        w = ALPHA / (2.0 * np.abs(k) ** 3) * np.exp(-BETA * G**2 / (np.abs(k)**2 * u19_5**4))
    return np.where(k > 0, w, 0.0)


def pm_spectrum_2d(kx: np.ndarray, ky: np.ndarray, u19_5: float,
                   phi_v: float = 0.0) -> np.ndarray:
    """二维方向 PM 谱, 论文式(2.17):

        W(kx,ky) = alpha/(2 K^4) exp(-beta g^2/(K^2 U^4)) cos^2(theta_k - phi_v)/pi

    输入数组为 FFT 序（fftfreq 角波数），直流项置 0。
    注: 式(2.17) 的 cos^2/pi 展宽在全平面内积分为 1/2, 即二维方向谱
    仅承载一维 PM 谱一半的方差; 该差异并入 capture 统计, 不单独修正。
    """
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    kx2 = kx**2
    ky2 = ky**2
    # 广播到 (Ny, Nx)
    K2 = ky2[:, None] + kx2[None, :]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        wk = ALPHA / (2.0 * K2**2) * np.exp(-BETA * G**2 / (K2 * u19_5**4))
        theta_k = np.arctan2(ky[:, None], kx[None, :])
        w = wk * np.cos(theta_k - phi_v) ** 2 / np.pi
    w[0, 0] = 0.0
    return w


def generate_pm_surface(u19_5: float, L: float, N: int, phi_v: float = 0.0,
                        seed: int | None = None, normalize_to_pm: bool = True):
    """生成二维 PM 谱随机海面高度场 eta(x, y)。

    参数:
        u19_5  : 19.5 m 高度风速 (m/s)
        L, N   : 海面边长 (m) 与每边采样数, dx = L/N
        phi_v  : 风向 (rad)
        seed   : 随机种子
        normalize_to_pm : True 时将 eta 归一化到完整 PM 理论 sigma_h
                          （补偿网格截断 + 方向谱展宽, 见模块 docstring）

    返回:
        eta  : (N, N) 海面高度 (m), 行对应 y, 列对应 x
        meta : dict(sigma_pm, sigma_grid, capture, swh, dx, ...)
               sigma_grid = sqrt(sum(W dkx dky)) 为本网格谱期望的均方根高度,
               capture    = (sigma_grid/sigma_pm)^2, 已解析方差比例
    """
    rng = np.random.default_rng(seed)
    dx = L / N
    dkx = 2.0 * np.pi / L

    # FFT 序角波数网格 (rad/m), 与 Gauss_Simulation.m 第 3 节一致
    n_idx = np.arange(N)
    kx = dkx * np.where(n_idx < N - N // 2, n_idx, n_idx - N)
    kx[0] = 0.0
    ky = kx.copy()

    W = pm_spectrum_2d(kx, ky, u19_5, phi_v)
    sigma_grid = float(np.sqrt(W.sum() * dkx * dkx))
    sigma_pm = pm_sigma_h(u19_5)

    # 线性滤波: b = H * gamma_hat, eta = 2*Re{ifft2(b)}  (Gauss_Simulation.m 第 5-6 节)
    H_f = (N * N / np.sqrt(2.0)) * np.sqrt(W * dkx * dkx)
    gamma_hat = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2.0)
    eta = 2.0 * np.real(np.fft.ifft2(H_f * gamma_hat))

    if normalize_to_pm and sigma_grid > 0:
        eta *= sigma_pm / np.std(eta)
    elif sigma_grid > 0:
        eta *= sigma_grid / np.std(eta)

    meta = {
        "u19_5": u19_5,
        "sigma_pm": sigma_pm,
        "swh": pm_swh(u19_5),
        "sigma_grid": sigma_grid,
        "capture": (sigma_grid / sigma_pm) ** 2,
        "dx": dx,
        "L": L,
        "N": N,
        "phi_v": phi_v,
        "normalized_to": "pm" if normalize_to_pm else "grid",
    }
    return eta, meta
