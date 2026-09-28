# -*- coding: utf-8 -*-
"""反演器集: 数值 Brown 拟合 + 阈值法（含 Brown 标定）+ σ⁰->风速。

阈值法（工程标准）:
    SSH = (t_50% - t_ref)·c/2,  t_ref 为零波高 Brown 波形的 50% 点
    SWH = a·W_10-90 + b,  (a,b) 由 Brown 模型族标定（宽度和 SWH 近似线性）

Brown 拟合: 正向模型复用 Echo_Simulation 的数值链（平坦冲激响应缓存,
仅重做波高核卷积 + 平移插值）, 三参数 (A, t0, σ_h)。

σ⁰->风速: 平台窗功率 / σ⁰≡1 响应 -> 有效天底 σ⁰ -> Cox-Munk 正演关系
σ⁰(U) = ρ/(2√(σ_u²σ_c²)) 二分解出 U（与仿真同公式, 闭环自洽）。
"""

import numpy as np
from scipy.optimize import least_squares

from altimeter_params import C
from echo_kernel import (echo_fine_density, apply_ptr, density_to_gates,
                         gauss_kernel, slope_variances)

L_SIM = 32e3  # 与 Echo_Simulation 两期一致的空间截断


def _flat_grid(n_flat: int = 512, L: float = L_SIM):
    axis = np.linspace(-L / 2.0, L / 2.0, n_flat, endpoint=False)
    return np.meshgrid(axis, axis), L / n_flat


class BrownFitModel:
    """数值 Brown 正向模型（平坦响应缓存）+ 三参数最小二乘拟合。"""

    def __init__(self, p, u19_5: float = 8.0, L: float = L_SIM,
                 n_flat: int = 512, q: int = 4, phi_v: float = 0.0,
                 sigma0_map=None):
        self.p = p
        self.q = q
        self.dt_f = p.gate_dt / q
        self.n_f = p.n_gates * q
        self.t_fine = p.t_start - p.gate_dt / 2.0 + (np.arange(self.n_f) + 0.5) * self.dt_f
        self.gate_centers = p.t_start + (np.arange(p.n_gates) + 0.5) * p.gate_dt
        (X, Y), dx = _flat_grid(n_flat, L)
        dens = echo_fine_density(np.zeros((n_flat, n_flat)), X, Y, dx, p,
                                 u19_5, phi_v, q=q, sigma0_map=sigma0_map)
        self.flat = apply_ptr(dens, p, q)
        self.flat = self.flat / self.flat.max()   # 归一到峰值 1: 拟合波形归一化
        # 后 scale~O(1) 才在 bounds 内 (真实功率量级 1e-22 会使拟合卡死)

    def waveform(self, sigma_h: float, t0: float = 0.0,
                 scale: float = 1.0) -> np.ndarray:
        sig_tau = 2.0 * sigma_h / C
        conv = np.convolve(self.flat, gauss_kernel(sig_tau, self.dt_f), "same")
        return scale * np.interp(self.gate_centers - t0, self.t_fine, conv)

    def fit(self, wf: np.ndarray, guess_list=None) -> dict:
        """三参数 (A, t0, σ_h) 拟合。返回 dict(A, t0, sigma_h, swh, resid_rms)。

        diff_step 用绝对步长: t0 初值为 0 时相对步长会使雅可比列全零,
        拟合退化为初值挑选（已踩坑验证）。
        """
        wfn = wf / np.max(wf)
        if guess_list is None:
            guess_list = [np.array([1.0, 0.0, 0.4])]
        best = None
        for x0 in guess_list:
            r = least_squares(
                lambda x: self.waveform(x[2], x[1] * 1e-9, x[0]) - wfn,
                x0, bounds=([0.2, -40, 1e-3], [5.0, 40, 4.0]),
                diff_step=[0.05, 0.5, 0.02],
                method="trf", xtol=1e-12, ftol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
        return dict(A=float(best.x[0]), t0=float(best.x[1]) * 1e-9,
                    sigma_h=float(best.x[2]), swh=4.0 * float(best.x[2]),
                    resid_rms=float(np.sqrt(np.mean(best.fun ** 2))))


# ---------------------------------------------------------------- 阈值法 ----

def smooth3(w: np.ndarray) -> np.ndarray:
    return np.convolve(w, np.ones(3) / 3.0, "same")


def _gate_axis_ns(p) -> np.ndarray:
    return (p.t_start + (np.arange(p.n_gates) + 0.5) * p.gate_dt) * 1e9


def _cross(wfn: np.ndarray, ax_ns: np.ndarray, level: float) -> float:
    i = int(np.argmax(wfn >= level))
    if i == 0:
        return float(ax_ns[0])
    return float(ax_ns[i - 1] + (level - wfn[i - 1]) / (wfn[i] - wfn[i - 1])
                 * (ax_ns[i] - ax_ns[i - 1]))


def calibrate_threshold(model: BrownFitModel,
                        swh_list=None) -> tuple:
    """Brown 模型族标定（Brown 族真值 SSH=0, SWH=4σ_h）:

        SWH = a·W_10-90(ns) + b          （前沿宽度 -> 波高）
        SSH = a2·t_50%(ns) + b2          （50% 点位置 -> 高度, 消 PTR/门积分
                                           非对称引起的 ~20 cm 系统差）

    返回 ((a,b), (a2,b2), t_ref(s), (widths, swhs, t50s))。
    """
    if swh_list is None:
        swh_list = np.linspace(0.25, 8.0, 14)
    p = model.p
    ax = _gate_axis_ns(p)
    widths, t50s = [], []
    for swh in swh_list:
        wfn = smooth3(model.waveform(swh / 4.0))
        wfn = wfn / np.max(wfn)
        widths.append(_cross(wfn, ax, 0.9) - _cross(wfn, ax, 0.1))
        t50s.append(_cross(wfn, ax, 0.5))
    a, b = np.polyfit(widths, swh_list, 1)
    # SSH 标定: 50% 点对对称卷积核近似不动, 系统差取标定族均值
    t50_ref = float(np.mean(t50s))
    a2 = C / 2.0 * 1e-9          # m per ns
    b2 = -a2 * t50_ref
    wfn0 = smooth3(model.waveform(0.002))
    t_ref = _cross(wfn0 / np.max(wfn0), ax, 0.5) * 1e-9
    return (float(a), float(b)), (a2, b2), t_ref, \
        (np.array(widths), np.array(swh_list), np.array(t50s))


def threshold_retrieve(wf: np.ndarray, p, cal: tuple, t_ref: float):
    """阈值法反演。cal=((a,b)_swh, (a2,b2)_ssh)。返回 (ssh_m, swh_m, t50_ns, width_ns)。"""
    ax = _gate_axis_ns(p)
    wfn = smooth3(wf)
    wfn = wfn / np.max(wfn)
    t50 = _cross(wfn, ax, 0.5)
    width = _cross(wfn, ax, 0.9) - _cross(wfn, ax, 0.1)
    (a, b), (a2, b2) = cal
    return (a2 * t50 + b2, a * width + b, t50, width)


# ------------------------------------------------------------ σ⁰ -> 风速 ----

PLATEAU_WINDOW_NS = (60.0, 160.0)


def unit_sigma0_plateau(p, q: int = 4, L: float = L_SIM,
                        n_flat: int = 512) -> float:
    """σ⁰≡1 平坦响应的平台窗均值（几何增益常数, 与风速无关）。"""
    (X, Y), dx = _flat_grid(n_flat, L)
    dens = echo_fine_density(np.zeros((n_flat, n_flat)), X, Y, dx, p, 8.0, 0.0,
                             q=q, sigma0_map=np.ones_like(X))
    wf = density_to_gates(apply_ptr(dens, p, q), p, q)
    ax = _gate_axis_ns(p)
    sel = (ax >= PLATEAU_WINDOW_NS[0]) & (ax <= PLATEAU_WINDOW_NS[1])
    return float(wf[sel].mean())


def sigma0_nadir(u19_5: float, rho: float) -> float:
    """天底 GO σ⁰ 正演, 与 Echo_Simulation 第一期指标表同公式。"""
    su2, sc2 = slope_variances(u19_5)
    return rho / (2.0 * np.sqrt(su2 * sc2))


def u19_5_from_sigma0(s0: float, rho: float) -> float:
    lo, hi = 0.5, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sigma0_nadir(mid, rho) > s0:   # σ⁰ 随风速单调下降
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wind_from_waveform(wf: np.ndarray, p, p_unit: float):
    """平台窗功率 -> (U19.5_est, sigma0_est)。"""
    ax = _gate_axis_ns(p)
    sel = (ax >= PLATEAU_WINDOW_NS[0]) & (ax <= PLATEAU_WINDOW_NS[1])
    s0 = float(wf[sel].mean()) / p_unit
    return u19_5_from_sigma0(s0, p.fresnel_rho), s0
