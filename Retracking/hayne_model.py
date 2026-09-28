# -*- coding: utf-8 -*-
"""Hayne/Amarouche 二阶近似解析回波模型（论文 5.1.1 节式 5.7-5.15）+ 拟合反演器。

模型结构（海面偏度 λ_s=0、忽略热噪声）:
    W(t) = 2*W1(t) - W2(t)                      式(5.8)
    W_i(t) = A0 * E_xi * exp(-D_i*(U_i + D_i/2)) * Phi(U_i)
    Phi = 正态 CDF（λ_s=0 时论文方括号恰为 [1+erf(U/√2)]/2）
    E_xi = exp(-(4/γ)·sin²ξ)                     误指向角衰减
    γ: 天线宽度参数。论文式(5.11) 的 PDF 提取有乱码; 这里采用与
       Echo_Simulation 天线模型自洽的**双程口径**: 数值链双程功率方向图
       G²(θ)=exp(-θ²/σ_θ²) ≈ exp(-(4/γ)sin²θ)  =>  γ = 4σ_θ² (rad²),
       由此 δ = 4c/(γH) = c/(σ_θ²H) 与数值平坦响应的实测尾部衰减率一致
       (数值 8.76e5 vs 解析 8.44e5 /s, 相对差 3.7%)。
    δ = (4/γ)(c/H)·cos(2ξ),  β = (4/γ)√(c/H)·sin(2ξ)   式(5.12)
    α1 = δ - β²/8,  α2 = δ                       式(5.13)
    D_i = α_i·σ_c,  U_i = (t - t0)/σ_c - D_i     式(5.14)(5.15)
    σ_c = √(σ_p² + σ_s²),  σ_p = 0.513/B (PTR 高斯近似, 式 5.5),  σ_s = 2σ_h/c

自验证锚点: ξ=0 时 β=0, α1=α2, W 退化为"平坦响应×高斯波高PDF"的一阶模型。
与数值 Brown 卷积波形的对比: 前沿/平台区(0-120 ns)重合 <0.5% RMS, 全窗
RMS 4-5%（差异集中于尾段, 来源于 σ⁰ 场角度结构与纯指数衰减近似的差）。
SWH = 2c·√(σ_c² - σ_p²)。

拟合反演器: 最小二乘联合估计 (A, t0, σ_c, ξ) 四参数（论文 5.1.2 节流程）。
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr

from altimeter_params import C


def _term(u: np.ndarray, d: float) -> np.ndarray:
    """exp(-D(U+D/2))·Phi(U), 用 log_ndtr 保证 U 很负时数值稳定。"""
    return np.exp(-d * (u + d / 2.0) + log_ndtr(u))


class HayneModel:
    """二阶近似解析模型正向 + 拟合。时间轴单位: 秒, 相对 2H/c。"""

    def __init__(self, p, q: int = 4):
        self.p = p
        self.sigma_p = 0.513 / p.fs                 # PTR 高斯近似宽度 (s), 式(5.5)
        self.gamma = 4.0 * p.sigma_theta ** 2        # 天线宽度参数 (rad², 双程口径)
        self.gate_centers = p.t_start + (np.arange(p.n_gates) + 0.5) * p.gate_dt

    def waveform(self, sigma_c: float, t0: float = 0.0, xi: float = 0.0,
                 scale: float = 1.0) -> np.ndarray:
        """二阶模型门波形。sigma_c (s), t0 (s, 前缘相对 2H/c), xi (rad)。"""
        g = self.gamma
        delta = 4.0 * C / (g * self.p.H) * np.cos(2.0 * xi)
        beta = 4.0 / g * np.sqrt(C / self.p.H) * np.sin(2.0 * xi)
        a1 = delta - beta ** 2 / 8.0
        a2 = delta
        d1 = a1 * sigma_c
        d2 = a2 * sigma_c
        e_xi = np.exp(-4.0 * np.sin(xi) ** 2 / g)
        u = (self.gate_centers - t0) / sigma_c
        w = 2.0 * _term(u - d1, d1) - _term(u - d2, d2)
        return scale * e_xi * w

    def swh_from_sigma_c(self, sigma_c: float) -> float:
        """σ_c -> SWH (m)。σ_c² = σ_p² + (2σ_h/c)², SWH = 4σ_h。"""
        var = sigma_c ** 2 - self.sigma_p ** 2
        return 2.0 * C * np.sqrt(max(var, 0.0))

    def fit(self, wf: np.ndarray, fit_xi: bool = True,
            t0_range_ns=(-40.0, 40.0), t0_init_ns=0.0,
            fix_sigma_c: float | None = None,
            guess_list=None) -> dict:
        """最小二乘拟合（论文 5.1.2 节）。

        wf: 已去热噪声的 104 门波形（任意单位）。
        fix_sigma_c=None: 四参数 (A, t0, σ_c, ξ) 联合估计。
            注意 ξ 与 σ_c 强耦合（误指向角展宽与波高展宽互相模仿）,
            联合拟合时 ξ 通常被 σ_c 吸收而不可辨识。
        fix_sigma_c=σ: 固定 σ_c, 拟合 (A, t0, ξ) —— 用于先由 ξ=0/阈值法
            确定 σ_c 后再估计等效误指向角。
        diff_step 用绝对步长: t0/ξ 初值为 0 时相对步长会使雅可比列全零。
        """
        wfn = wf / np.max(wf)
        if fix_sigma_c is not None:
            if guess_list is None:
                guess_list = [np.array([1.0, t0_init_ns, 0.0])]
            best = None
            for x0 in guess_list:
                r = least_squares(
                    lambda x: self.waveform(fix_sigma_c, x[1] * 1e-9,
                                            np.deg2rad(x[2]) if fit_xi else 0.0,
                                            x[0]) - wfn,
                    x0, bounds=([0.2, t0_range_ns[0], 0.0],
                                [5.0, t0_range_ns[1], 1.5 if fit_xi else 1e-9]),
                    diff_step=[0.05, 0.5, 0.05],
                    method="trf", xtol=1e-12, ftol=1e-12)
                if best is None or r.cost < best.cost:
                    best = r
            xi_rad = np.deg2rad(best.x[2]) if fit_xi else 0.0
            return dict(A=float(best.x[0]), t0=float(best.x[1]) * 1e-9,
                        sigma_c=float(fix_sigma_c), xi=float(xi_rad),
                        swh=self.swh_from_sigma_c(float(fix_sigma_c)),
                        resid_rms=float(np.sqrt(np.mean(best.fun ** 2))))
        xi_hi = 1.5 if fit_xi else 0.0   # 度
        if guess_list is None:
            guess_list = [np.array([1.0, t0_init_ns, 3.0, 0.0])]
        best = None
        for x0 in guess_list:
            if not fit_xi:
                x0 = np.array([x0[0], x0[1], x0[2], 0.0])
            r = least_squares(
                lambda x: (self.waveform(x[2] * 1e-9, x[1] * 1e-9,
                                         np.deg2rad(x[3]) if fit_xi else 0.0,
                                         x[0]) - wfn),
                x0,
                bounds=([0.2, t0_range_ns[0], 1.7, 0.0],
                        [5.0, t0_range_ns[1], 80.0, xi_hi if fit_xi else 1e-9]),
                diff_step=[0.05, 0.5, 0.1, 0.05],
                method="trf", xtol=1e-12, ftol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
        xi_rad = np.deg2rad(best.x[3]) if fit_xi else 0.0
        return dict(A=float(best.x[0]), t0=float(best.x[1]) * 1e-9,
                    sigma_c=float(best.x[2]) * 1e-9, xi=float(xi_rad),
                    swh=self.swh_from_sigma_c(float(best.x[2]) * 1e-9),
                    resid_rms=float(np.sqrt(np.mean(best.fun ** 2))))
