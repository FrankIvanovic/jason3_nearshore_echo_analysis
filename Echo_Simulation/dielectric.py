# -*- coding: utf-8 -*-
"""海水介电常数 Debye 模型（论文 2.4.1 节, 式 2.20-2.23）与陆地介质常数。

海水相对介电常数 eps(S, T, f) 依赖盐度 S(千分比)、温度 T(摄氏度)、频率 f(Hz)。
Klein-Swift 型 Debye 弛豫模型（eps_s = 4.9 + eps1 为静态介电常数, eps_inf = 4.9）:
    Re eps = 4.9 + (eps1 - 4.9)/(1+(2 pi f tau)^2)
    Im eps = (2 pi f tau)(eps1 - 4.9)/(1+(2 pi f tau)^2) + sigma/(2 pi f eps0)
论文基准: f=13.575 GHz, T=20 degC, S=35 permil -> eps = 47.0823 + 39.0651j,
对应法向菲涅尔反射系数模方 rho = 0.6173。

陆地介质: 论文 4.3 节直接给出结果值（Wang 土壤模型在给定含水量/温度下的输出）,
沙质土壤在 Ku 波段实部远低于海水 -> 反射率低, 但表面更粗糙 -> 天底 σ0 可高于或
低于海面, 取决于其谱参数（见 coastal.py）。
"""

import numpy as np

EPS0 = 8.85419e-12  # 真空介电常数 (F/m)


def seawater_eps(salinity: float, temperature: float, freq: float) -> complex:
    """海水相对介电常数, 论文式(2.20)-(2.23)。

    参数:
        salinity    : 盐度 S (千分比, permil), 海水典型 35
        temperature : 温度 T (摄氏度), 范围约 -2 ~ 30
        freq        : 频率 (Hz)
    """
    t = float(temperature)
    s = float(salinity)

    # 式(2.21): 静态介电常数相关的 eps1(S, T)
    eps1 = ((87.134 - 1.949e-1 * t - 1.276e-2 * t**2 + 2.491e-4 * t**3)
            * (1.0 + 1.613e-5 * t * s - 3.656e-3 * s + 3.210e-5 * s**2
               - 4.232e-7 * s**3))

    # 式(2.22): 弛豫时间 tau(S, T) (s)
    tau = ((1.768e-11 - 6.086e-13 * t + 1.104e-14 * t**2 - 8.111e-17 * t**3)
           * (1.0 + 2.282e-5 * t * s - 7.638e-4 * s - 7.76e-6 * s**2
              + 1.105e-8 * s**3))

    # 式(2.23): 离子电导率 sigma(S, T) (S/m), 温差 Delta = 25 - T
    delta = 25.0 - t
    sigma_cond = (s * (0.182521 - 1.46192e-3 * s + 2.09324e-5 * s**2
                       - 1.28205e-7 * s**3)
                  * np.exp(-delta * (2.033e-2 + 1.266e-4 * delta + 2.464e-6 * delta**2
                                     - s * (1.849e-5 - 2.551e-7 * delta
                                            + 2.551e-8 * delta**2))))

    w = 2.0 * np.pi * freq
    denom = 1.0 + (w * tau) ** 2
    # 实部: eps_inf + (eps_s - eps_inf)/(1+(w tau)^2), 其中 eps_s = 4.9 + eps1
    re = 4.9 + (eps1 - 4.9) / denom
    im = (w * tau * (eps1 - 4.9) / denom + sigma_cond / (w * EPS0))
    return complex(re, im)


def fresnel_rho(eps: complex) -> float:
    """法向菲涅尔反射系数模方 rho = |(1-sqrt(eps))/(1+sqrt(eps))|^2, 式(4.6)。"""
    r = (1 - np.sqrt(eps)) / (1 + np.sqrt(eps))
    return float(np.abs(r) ** 2)


# ---- 陆地介质（论文 4.3 节给出的 Wang 土壤模型结果值, Ku 波段）----
# 沙滩: 沙质土壤, 4.3.1 节 eps_s = (22.5032, 15.7495)
LAND_EPS_BEACH = 22.5032 + 15.7495j
# 岛屿: 沙壤土, 含水量 mv=0.5, T=20 degC, 4.3.3 节 eps_t = (22.4128, 13.8675)
LAND_EPS_ISLAND = 22.4128 + 13.8675j
# 礁石: 论文未单独给出, 取岩石表面按沙壤土介电常数近似（其强回波由更粗糙的
# 指数谱参数贡献, 见 coastal.py; 论文图 4.11 礁石峰高于海洋平台, 与该组合一致）
LAND_EPS_REEF = LAND_EPS_ISLAND
