# -*- coding: utf-8 -*-
"""雷达高度计系统参数（Jason-3 / 论文口径）。

参数与参考文献《基于电磁散射特性的雷达高度计回波仿真与分析》
（华中科技大学硕士学位论文）第四章保持一致：
  H = 1336 km, Ku 波段 13.575 GHz, Pt = 50 W, 波束宽度 2.2 度,
  采样率 320 MHz, 104 个距离门（与 Jason SGDR Ku 波段波形门数一致）。
距离门时间轴相对"平均海平面双程延迟 2H/c"偏移 t_start。

海水介电常数取论文 2.4.1 节 Debye 模型结果（13.58 GHz, 20 degC, 35 permil）:
eps = 47.0823 + j*39.0651, 对应法向菲涅尔反射系数平方 rho = 0.6173。
"""

from dataclasses import dataclass, field
import numpy as np

C = 3.0e8  # 光速 (m/s)


@dataclass
class AltimeterParams:
    H: float = 1336e3            # 轨道高度 (m)
    f: float = 13.575e9          # Ku 波段频率 (Hz)
    Pt: float = 50.0             # 峰值发射功率 (W)
    beam_deg: float = 2.2        # 3dB 波束宽度 (deg), 论文口径
    fs: float = 320e6            # 采样率 = 带宽 (Hz)
    n_gates: int = 104           # 距离门数
    t_start: float = -50e-9      # 首门中心相对 2H/c 的偏移 (s)
    eps_sea: complex = 47.0823 + 1j * 39.0651   # 海水相对介电常数

    # ---- 派生量 ----
    @property
    def lam(self) -> float:
        """波长 (m)。"""
        return C / self.f

    @property
    def gate_dt(self) -> float:
        """距离门间隔 (s) = 1/fs = 3.125 ns。"""
        return 1.0 / self.fs

    @property
    def sigma_theta(self) -> float:
        """高斯功率方向图标准差 (rad)。

        G(theta) = exp(-theta^2 / (2 sigma_theta^2)),
        在 theta = beam_deg/2 处恰为半功率 (-3 dB)。
        """
        half = np.deg2rad(self.beam_deg) / 2.0
        return np.tan(half) / np.sqrt(2.0 * np.log(2.0))

    @property
    def fresnel_rho(self) -> float:
        """法向菲涅尔反射系数的模方 rho = |R(0)|^2, 论文式(4.6)。"""
        r = (1 - np.sqrt(self.eps_sea)) / (1 + np.sqrt(self.eps_sea))
        return float(np.abs(r) ** 2)

    @property
    def tau0(self) -> float:
        """平均海平面双程延迟 2H/c (s), 距离门时间轴的零点。"""
        return 2.0 * self.H / C

    def gate_times(self) -> np.ndarray:
        """104 个距离门中心相对 tau0 的时间偏移 (s)。"""
        return self.t_start + np.arange(self.n_gates) * self.gate_dt


def default_params() -> AltimeterParams:
    return AltimeterParams()
