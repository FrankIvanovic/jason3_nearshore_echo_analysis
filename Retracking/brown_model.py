# -*- coding: utf-8 -*-
"""Brown 平均回波模型（数值实现）。

论文 5.1.1 节式(5.1)-(5.4):
    W(t) = P_FS(t) * q_s(eta) * S_r(tau)
平坦海面冲激响应 P_FS 与海面波高概率密度 q_s、雷达点目标响应 S_r 的卷积。

实现方式: P_FS 用与逐面元仿真完全相同的代码路径计算（高度取 0 的平面,
同一几何/增益/sigma0/分箱, 空间域截断也与逐面元仿真一致）, 从而系综平均
的逐面元结果应收敛到本模型 —— 该一致性即仿真正确性的对照验证。
q_s 取高斯（偏度 lambda_s = 0）, 标准差 sigma_tau = 2*sigma_h/c。
"""

import numpy as np

from altimeter_params import C
from echo_kernel import echo_fine_density, apply_ptr, gauss_kernel


def brown_waveform(p, u19_5, phi_v, sigma_h, L, n_flat=512, q=4):
    """Brown 模型平均波形。

    L: 与逐面元仿真一致的空间域边长（截断相同, 尾部一致性才可比）。
    sigma_h: 海面均方根高度 (m), 应与逐面元仿真海面的实际 sigma_h 一致。
    返回 (104,) 门功率数组。
    """
    dx_f = L / n_flat
    axis = np.linspace(-L / 2.0, L / 2.0, n_flat, endpoint=False)
    X, Y = np.meshgrid(axis, axis)

    flat = echo_fine_density(np.zeros((n_flat, n_flat)), X, Y, dx_f, p,
                             u19_5, phi_v, use_local_tilt=False, q=q)
    flat = apply_ptr(flat, p, q)

    sigma_tau = 2.0 * sigma_h / C
    dt_f = p.gate_dt / q
    conv = np.convolve(flat, gauss_kernel(sigma_tau, dt_f), mode="same")

    from echo_kernel import density_to_gates
    return density_to_gates(conv, p, q)
