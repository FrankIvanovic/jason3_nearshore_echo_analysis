# -*- coding: utf-8 -*-
"""近海复合场景（陆地 + 海面）与陆地 GO 散射（两尺度口径）。

论文 4.3 节: 沙滩/礁石/岛屿等陆地用指数谱粗糙面描述（式 2.18/2.19），与 PM 海面
复合，复合模型为沙滩条带/方形礁石/圆形岛屿（论文图 4.9）。

本模块不显式生成陆地高度场: 陆地微粗糙相关长度 l = 1.5*lambda ≈ 3.3 cm 远小于
仿真面元 dx ≈ 15.6 m，逐面元无法也无需解析 —— 采用两尺度口径:
  * 面元尺度上陆地是高度平台（海拔偏移，改变双程延迟 -> 回波提前）;
  * 微粗糙通过指数谱斜率方差（截断波数 k_cut = 2*pi/lambda，短于雷达波长的
    分量对 Ku 散射无额外贡献）的各向同性 GO σ0 体现:
        sigma0 = rho_land * exp(-tan^2 th / (2 sig_s^2)) / (2 pi sig_s^2 cos^4 th)
指数谱斜率方差随谱参数 sigma^2 增大:
  沙滩/礁石 sigma = 0.1*lambda -> sig_s^2 ~ 8.3e-3 -> sigma0(0) ~ 14.6 dB（强于海面）
  岛屿   sigma = 0.5*lambda -> sig_s^2 ~ 0.21   -> sigma0(0) ~ 0.1 dB （弱于海面）
与论文图 4.10（沙滩使回波下降沿抬升）、图 4.11（礁石峰高于海洋平台）、
图 4.12（岛屿回波弱于海洋）定性一致。
k_cut 取 2*pi/lambda 为两尺度截断的常规口径; 若取 pi/lambda, 陆地 σ0 约降 4 dB
（敏感性在 README 说明）。
"""

import numpy as np

from dielectric import fresnel_rho, LAND_EPS_BEACH, LAND_EPS_ISLAND, LAND_EPS_REEF
from echo_kernel import sigma0_go


def exp_slope_variance(sigma: float, l_corr: float, k_cut: float) -> float:
    """可分离二维指数谱（论文式 2.19）的单方向斜率方差，截断于 [0, k_cut]。

        sigma_s^2 = ∫∫ Kx^2 W dKx dKy
                  = sigma^2 l^2/(2 pi) * [∫ k^2/(1+k^2 l^2)] * [∫ 1/(1+k^2 l^2)]
    """
    a = l_corr
    i1 = (k_cut - np.arctan(a * k_cut) / a) / a**2   # ∫₀^K k²/(1+k²a²) dk
    i0 = np.arctan(a * k_cut) / a                    # ∫₀^K 1/(1+k²a²) dk
    return sigma**2 * a * a / (2.0 * np.pi) * i1 * i0


def land_sigma0(theta: np.ndarray, rho: float, sig_s2: float) -> np.ndarray:
    """陆地各向同性 GO 后向散射系数（Cox-Munk 形式, 斜率方差 = sig_s2）。"""
    tan_t = np.tan(theta)
    pdf = np.exp(-tan_t ** 2 / (2.0 * sig_s2)) / (2.0 * np.pi * sig_s2)
    return rho * np.pi * pdf / np.cos(theta) ** 4


def land_spec(kind: str, lam: float) -> dict:
    """陆地类型参数（论文 4.3 节）。height 为海拔偏移 (m)。

    礁石高度 +2 m 来自论文 4.3.2; 岛屿高度论文未给出, 取 +12 m（中等海岛假设,
    只影响回波提前量, 可调）。
    """
    if kind == "beach":
        return dict(eps=LAND_EPS_BEACH, sigma=0.1 * lam, l=1.5 * lam,
                    height=0.5, name="沙滩")
    if kind == "reef":
        return dict(eps=LAND_EPS_REEF, sigma=0.1 * lam, l=1.5 * lam,
                    height=2.0, name="礁石")
    if kind == "island":
        return dict(eps=LAND_EPS_ISLAND, sigma=0.5 * lam, l=1.5 * lam,
                    height=12.0, name="岛屿")
    raise ValueError(kind)


def beach_mask(X: np.ndarray, Y: np.ndarray, L: float, fraction: float) -> np.ndarray:
    """沙滩条带: +x 侧条带, 面积占比 = fraction（论文图 4.9(a)）。"""
    return (X > (0.5 - fraction) * L).astype(float)


def square_mask(X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
                half: float) -> np.ndarray:
    """方形礁石, 中心 (cx, cy), 半边长 half（论文图 4.9(b)）。"""
    return ((np.abs(X - cx) <= half) & (np.abs(Y - cy) <= half)).astype(float)


def disk_mask(X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
              radius: float) -> np.ndarray:
    """圆形岛屿, 中心 (cx, cy), 半径 radius（论文图 4.9(c)）。"""
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2).astype(float)


def hetero_sigma0(X: np.ndarray, Y: np.ndarray, mask: np.ndarray,
                  p, u19_5: float, phi_v: float, spec: dict) -> np.ndarray:
    """复合场景的逐面元 σ0 图: 海面用 Cox-Munk(式 4.5-4.10), 陆地用指数谱 GO。

    两类面元都用几何入射角（长波倾斜不调制陆地平台）, 与 Brown 对照口径一致。
    """
    theta = np.arctan(np.sqrt(X * X + Y * Y) / p.H)
    phi = np.arctan2(Y, X)
    s_sea = sigma0_go(theta, phi, u19_5, phi_v, p.fresnel_rho)
    sig_s2 = exp_slope_variance(spec["sigma"], spec["l"], 2.0 * np.pi / p.lam)
    s_land = land_sigma0(theta, fresnel_rho(spec["eps"]), sig_s2)
    return np.where(mask > 0, s_land, s_sea)
