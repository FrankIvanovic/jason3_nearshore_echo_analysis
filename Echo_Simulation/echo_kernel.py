# -*- coding: utf-8 -*-
"""逐面元回波核: GO/KA 后向散射系数 + 雷达方程按距离门累加。

物理模型（论文第四章）:
  1. 每个面元的双程延迟 tau = 2R/c, R = sqrt(x^2 + y^2 + (H-h)^2)   式(4.1)
  2. 天线高斯功率方向图 G(theta), 半功率角 = beam_deg/2            式(4.2)-(4.4)
  3. GO 后向散射系数 sigma0 = rho*pi*p(tan theta,0)/cos^4 theta     式(4.5)
     其中斜率 PDF 为各向异性 Cox-Munk 形式, 斜率方差随风速线性变化  式(4.7)-(4.10)
  4. 雷达方程: P(tau_n) = lambda^2 Pt/((4 pi)^3 R^4) * G^2 * sigma0 * dS,
     按面元延迟分箱到精细时间网格, 与雷达点目标响应 sinc^2 卷积后
     每门积分得到 104 门波形。

两尺度口径: 波高/延迟分布由 PM 长波海面承载; sigma0 的斜率统计用含全部
波段的 Cox-Munk 总方差（式(4.9)(4.10), 短波贡献不可解析, 由统计量承载）。
use_local_tilt=True 时面元局部入射角由 PM 长波法向确定（长波倾斜对 sigma0
的调制）, False 时用几何入射角（与 Brown 模型假设严格一致, 用于对照验证）。
"""

import numpy as np

from altimeter_params import C


def fresnel_rho(eps_sea: complex) -> float:
    """法向菲涅尔反射系数模方 rho = |(1-sqrt(eps))/(1+sqrt(eps))|^2, 式(4.6)。"""
    r = (1 - np.sqrt(eps_sea)) / (1 + np.sqrt(eps_sea))
    return float(np.abs(r) ** 2)


def slope_variances(u19_5: float):
    """Cox-Munk 型斜率方差, 论文式(4.9)(4.10), U 为 19.5 m 高度风速。

    sigma_u2: 顺风向, sigma_c2: 交叉风向。
    """
    sigma_u2 = 0.00078545 * u19_5 + 0.0092407
    sigma_c2 = 0.00052799 * u19_5 + 0.0097295
    return sigma_u2, sigma_c2


def sigma0_go(theta: np.ndarray, phi: np.ndarray, u19_5: float,
              phi_v: float, rho: float) -> np.ndarray:
    """GO 后向散射系数, 论文式(4.5)-(4.10):

        sigma0 = rho * pi * p(tan theta, 0) / cos^4 theta
        p = exp(-tan^2 theta / v) / (2 pi sigma_u sigma_c)
        v = 2 / (cos^2(phi-phi_v)/sigma_u2 + sin^2(phi-phi_v)/sigma_c2)
    """
    sigma_u2, sigma_c2 = slope_variances(u19_5)
    v = 2.0 / (np.cos(phi - phi_v) ** 2 / sigma_u2
               + np.sin(phi - phi_v) ** 2 / sigma_c2)
    tan_t = np.tan(theta)
    p = np.exp(-tan_t ** 2 / v) / (2.0 * np.pi * np.sqrt(sigma_u2 * sigma_c2))
    return rho * np.pi * p / np.cos(theta) ** 4


def gain_power(theta: np.ndarray, sigma_theta: float) -> np.ndarray:
    """单程高斯功率方向图 G(theta) = exp(-theta^2/(2 sigma_theta^2))。"""
    return np.exp(-theta ** 2 / (2.0 * sigma_theta ** 2))


def ptr_kernel(p, q: int) -> np.ndarray:
    """雷达理想点目标响应 S_r(tau) = (sin(pi B tau)/(pi B tau))^2, 论文式(5.4)。

    离散卷积意义下归一化（采样点和为 1, 能量守恒）, 半宽 +-6/B,
    采样间隔 = gate_dt/q。
    """
    dt_f = p.gate_dt / q
    half = int(np.ceil(6.0 / (p.fs * dt_f)))          # 6/B 折合 q*6 个精细格
    t = np.arange(-half, half + 1) * dt_f
    k = np.sinc(p.fs * t) ** 2
    k /= k.sum()
    return k


def gauss_kernel(sigma: float, dt_f: float) -> np.ndarray:
    """高斯核（海面波高概率密度卷积, 论文式(5.1)(5.3) 取偏度 lambda_s=0）。

    同样按采样点和为 1 归一化, 保证与 PTR、逐面元路径电平一致。
    """
    half = max(1, int(np.ceil(4.0 * sigma / dt_f)))
    t = np.arange(-half, half + 1) * dt_f
    k = np.exp(-t ** 2 / (2.0 * sigma ** 2))
    k /= k.sum()
    return k


def echo_fine_density(eta, X, Y, dx, p, u19_5, phi_v,
                      use_local_tilt=False, q=4, sigma0_map=None):
    """面元功率 -> 精细时间网格功率密度（未与 PTR 卷积）。

    sigma0_map: 可选的逐面元 σ0 覆盖（近海复合场景用, 见 coastal.py）。
    传入时跳过内部海面 σ0 计算（eta 仍参与延迟 R 的计算）。

    返回长度 n_gates*q 的数组, 精细格中心 t_i = tau0 + t_start - gate_dt/2
    + (i+0.5)*gate_dt/q。
    """
    Ny, Nx = eta.shape
    dS = dx * dx
    rho = p.fresnel_rho

    # 几何: 双程延迟与几何入射角, 式(4.1)
    R = np.sqrt(X * X + Y * Y + (p.H - eta) ** 2)
    theta_geom = np.arctan(np.sqrt(X * X + Y * Y) / p.H)
    phi_loc = np.arctan2(Y, X)

    # 双程功率方向图（误指向角取 0, 天线指向天底点）
    G2 = gain_power(theta_geom, p.sigma_theta) ** 2

    if sigma0_map is not None:
        s0 = sigma0_map
    elif use_local_tilt:
        # 长波倾斜: 面元法向 n = (-dx_eta,-dy_eta,1)/norm, 视线单位矢 u = (-x,-y,H-h)/R
        d_dy, d_dx = np.gradient(eta, dx, dx)      # axis0=y, axis1=x
        norm = np.sqrt(1.0 + d_dx * d_dx + d_dy * d_dy)
        cos_loc = (X * d_dx + Y * d_dy + (p.H - eta)) / (R * norm)
        theta_loc = np.arccos(np.clip(cos_loc, 1e-6, 1.0))
        s0 = sigma0_go(theta_loc, phi_loc, u19_5, phi_v, rho)
    else:
        s0 = sigma0_go(theta_geom, phi_loc, u19_5, phi_v, rho)

    # 雷达方程逐面元权重: lambda^2 Pt G^2 sigma0 dS / ((4 pi)^3 R^4)
    w = p.lam ** 2 * p.Pt / ((4.0 * np.pi) ** 3) * G2 * s0 * dS / R ** 4

    # 按双程延迟分箱到精细时间网格
    dt_f = p.gate_dt / q
    n_f = p.n_gates * q
    t_lo = p.tau0 + p.t_start - p.gate_dt / 2.0
    idx = np.floor((2.0 * R / C - t_lo) / dt_f).astype(np.int64)
    mask = (idx >= 0) & (idx < n_f)
    dens = np.bincount(idx[mask], weights=w[mask], minlength=n_f) / dt_f
    return dens


def apply_ptr(dens: np.ndarray, p, q: int = 4) -> np.ndarray:
    """与理想点目标响应 sinc^2 卷积（压缩后脉冲的时间展宽）。"""
    return np.convolve(dens, ptr_kernel(p, q), mode="same")


def density_to_gates(dens: np.ndarray, p, q: int = 4) -> np.ndarray:
    """精细密度 -> 104 门波形（门内积分功率）。"""
    dt_f = p.gate_dt / q
    return dens.reshape(p.n_gates, q).mean(axis=1) * p.gate_dt
