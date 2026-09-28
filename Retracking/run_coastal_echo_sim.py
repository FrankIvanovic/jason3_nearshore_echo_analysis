# -*- coding: utf-8 -*-
"""近海复合场景回波仿真主脚本（对应论文 4.3 节）+ 海况参数（盐度/温度）扫描。

场景:
  A. 论文呈现风格波形: 单次实现、逐门折线+散斑, U19.5=4/8/12（对照论文图 4.3）
  B. 沙滩占比扫描: +x 侧沙滩条带占比 0~80%, U19.5=8（对照论文 4.3.1/图 4.10）
  C. 礁石位置扫描: 2 km 方形礁石位于天底点/2.2/4.4/8.8 km（对照 4.3.2/图 4.11）
  D. 岛屿位置扫描: 半径 1.5 km 圆形岛屿位于天底点/2.2/4.4/6.6 km（对照 4.3.3/图 4.12）
  E. 海水盐度/温度扫描: Debye 介电常数（式 2.20-2.23）-> rho -> σ0/平台功率

说明:
  * B/C/D 用 M 次海面实现系综平均（common random numbers, 突出确定性陆地特征）;
  * C/D 场景窗口前移到 -150 ns 以容纳礁石/岛屿高于海面导致的回波提前;
  * E 中 GO 模型 σ0 ∝ rho 为纯乘性, 波形形状不变, 仅平台功率平移,
    故无需重新仿真海面。

用法:
  python run_coastal_echo_sim.py           # 完整 (2048 网格, M=6, 约 3-4 min)
  python run_coastal_echo_sim.py --quick   # 冒烟 (512 网格, M=2)
"""

import argparse
import time
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from altimeter_params import AltimeterParams
from pm_surface import generate_pm_surface, pm_swh
from echo_kernel import echo_fine_density, apply_ptr, density_to_gates
from coastal import (land_spec, beach_mask, square_mask, disk_mask,
                     hetero_sigma0, exp_slope_variance)
from dielectric import seawater_eps, fresnel_rho

# ---------------- 仿真配置 ----------------
L_DOMAIN = 32e3
N_GRID = 2048
M_ENS = 6
SEED0 = 2026
U_COASTAL = 8.0          # 近海场景海面风速 U19.5 (m/s), 论文 4.3 节口径
WINDS_THESIS = [4.0, 8.0, 12.0]
Q_FINE = 4
OUT = Path(__file__).resolve().parent / "output"


def simulate(p, eta, X, Y, dx, u19_5, phi_v, sigma0_map=None):
    """单实现 104 门波形（可选 σ0 覆盖）。"""
    dens = echo_fine_density(eta, X, Y, dx, p, u19_5, phi_v,
                             sigma0_map=sigma0_map, q=Q_FINE)
    dens = apply_ptr(dens, p, Q_FINE)
    return density_to_gates(dens, p, Q_FINE)


def grid(n_grid):
    axis = np.linspace(-L_DOMAIN / 2.0, L_DOMAIN / 2.0, n_grid, endpoint=False)
    return np.meshgrid(axis, axis)


def surfaces_for(u19_5, n_grid, m, seed0=SEED0):
    return [generate_pm_surface(u19_5, L_DOMAIN, n_grid, 0.0, seed=seed0 + k)[0]
            for k in range(m)]


def ensemble_waveform(p, surfs, X, Y, dx, u19_5, config_fn):
    """对 M 个实现取平均; config_fn(eta) -> (eta_total, sigma0_map)。"""
    acc = np.zeros(p.n_gates)
    for eta in surfs:
        eta_t, s0 = config_fn(eta)
        acc += simulate(p, eta_t, X, Y, dx, u19_5, 0.0, sigma0_map=s0)
    return acc / len(surfs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    n_grid = 512 if args.quick else N_GRID
    m_ens = 2 if args.quick else M_ENS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = AltimeterParams()
    p_wide = replace(p, t_start=-150e-9)   # C/D 场景加宽窗口（回波提前）
    dx = L_DOMAIN / n_grid
    X, Y = grid(n_grid)
    t_ref = p.gate_times() * 1e9
    t_wide = p_wide.gate_times() * 1e9

    print("=" * 78)
    print("近海复合场景回波仿真 (PM 海面 + 陆地指数谱 GO, 两尺度口径)")
    print("=" * 78)
    print(f"  H={p.H/1e3:.0f} km  f={p.f/1e9:.3f} GHz  海面域 {L_DOMAIN/1e3:.0f} km, "
          f"N={n_grid}, M={m_ens}, U19.5={U_COASTAL:.0f} m/s (SWH={pm_swh(U_COASTAL):.2f} m)")
    t_wall = time.time()

    # ================= 场景 A: 论文呈现风格 =================
    print("\n[A] 论文呈现风格: 单次实现逐门波形 (U19.5=4/8/12)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    thesis_meta = {}
    for ax, u in zip(axes, WINDS_THESIS):
        eta, meta = generate_pm_surface(u, L_DOMAIN, n_grid, 0.0, seed=SEED0)
        wf = simulate(p, eta, X, Y, dx, u, 0.0)
        wf_n = wf / wf.max()
        ax.plot(np.arange(1, p.n_gates + 1), wf_n, "b.-", ms=4, lw=0.9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("采样点"); ax.set_ylabel("归一化回波功率")
        ax.set_title(f"风速 U19.5={u:.0f} m/s 的回波波形")
        ax.grid(True, alpha=0.3)
        # 前沿 10-90 宽度（门数）
        w10 = np.argmax(wf_n >= 0.1); w90 = np.argmax(wf_n >= 0.9)
        thesis_meta[u] = dict(edge_gates=w90 - w10, swh=meta["swh"])
        print(f"    U={u:.0f}: SWH={meta['swh']:.2f} m, "
              f"前沿(10-90)={w90-w10} 门 = {(w90-w10)*p.gate_dt*1e9:.1f} ns")
    fig.suptitle("论文呈现风格: 单次实现、逐门折线+散斑（对照论文图 4.3）", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "thesis_style_waveforms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 场景 B: 沙滩占比扫描 =================
    print("\n[B] 沙滩条带占比扫描 (U19.5=8, 系综 M=%d)" % m_ens)
    spec_b = land_spec("beach", p.lam)
    sb2 = exp_slope_variance(spec_b["sigma"], spec_b["l"], 2 * np.pi / p.lam)
    print(f"    沙滩: eps={spec_b['eps']}, sigma={spec_b['sigma']*1e3:.2f} mm, "
          f"sig_s2={sb2:.2e}, sigma0(0)={10*np.log10(fresnel_rho(spec_b['eps'])/(2*sb2)):.1f} dB")
    surfs = surfaces_for(U_COASTAL, n_grid, m_ens)
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    beach_wf = {}
    for f in fractions:
        mask = beach_mask(X, Y, L_DOMAIN, f) if f > 0 else np.zeros_like(X)
        s0 = hetero_sigma0(X, Y, mask, p, U_COASTAL, 0.0, spec_b) if f > 0 else None
        wf = ensemble_waveform(p, surfs, X, Y, dx, U_COASTAL,
                               lambda e, m=mask, s=s0: (e + m * spec_b["height"], s))
        beach_wf[f] = wf
        print(f"    占比 {f*100:.0f}%: 峰值={wf.max():.3e}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(fractions)))
    plateau_sel = (t_ref >= 60) & (t_ref <= 160)
    tail_sel = (t_ref >= 220) & (t_ref <= 268)
    plat_dB, tail_dB = [], []
    sea_plateau = beach_wf[0.0][plateau_sel].mean()
    sea_tail = beach_wf[0.0][tail_sel].mean()
    for c, f in zip(colors, fractions):
        wf = beach_wf[f]
        ax1.plot(t_ref, wf / wf.max(), color=c, lw=1.8,
                 label=f"沙滩占比 {f*100:.0f}%")
        plat_dB.append(10 * np.log10(wf[plateau_sel].mean() / sea_plateau))
        tail_dB.append(10 * np.log10(wf[tail_sel].mean() / sea_tail))
    ax1.set_xlabel("时间延迟 (ns)"); ax1.set_ylabel("归一化回波功率")
    ax1.set_title("不同沙滩占比的平均回波（各白归一化, 对照论文图 4.10）")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.plot(np.array(fractions) * 100, tail_dB, "o-", color="C3", lw=1.8,
             label="尾窗 220-268 ns（远区 r≈9-10 km）")
    ax2.plot(np.array(fractions) * 100, plat_dB, "s-", color="C0", lw=1.8,
             label="平台 60-160 ns（近足印）")
    ax2.set_xlabel("沙滩占比 (%)")
    ax2.set_ylabel("功率相对纯海面 (dB)")
    ax2.set_title("尾窗对远区沙滩立即响应;\n平台在沙滩侵入天底足印后才抬升(上限≈σ0差1.4dB)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"沙滩(指数谱 σ=0.1λ, εs=(22.50,15.75))复合海面 U19.5={U_COASTAL:.0f} m/s",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "beach_fraction_scan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 场景 C: 礁石位置扫描 =================
    print("\n[C] 礁石位置扫描 (2km 方块, +2 m, U19.5=8, 宽窗)")
    spec_c = land_spec("reef", p.lam)
    positions_km = [0.0, 2.2, 4.4, 8.8]
    reef_wf = {}
    for r_km in positions_km:
        mask = square_mask(X, Y, r_km * 1e3, 0.0, 1e3)
        s0 = hetero_sigma0(X, Y, mask, p_wide, U_COASTAL, 0.0, spec_c)
        wf = ensemble_waveform(p_wide, surfs, X, Y, dx, U_COASTAL,
                               lambda e, m=mask, s=s0: (e + m * spec_c["height"], s))
        reef_wf[r_km] = wf
        print(f"    位置 {r_km:.1f} km: 完成")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True, sharey=True)
    sea_ref = reef_wf[0.0].max()
    for ax, r_km in zip(axes.ravel(), positions_km):
        wf = reef_wf[r_km]
        ax.plot(t_wide, wf / sea_ref, "C0-", lw=1.6)
        ax.axvline(0.0, color="0.5", ls=":", lw=0.8)
        ax.set_title(f"礁石距天底点 {r_km:.1f} km", fontsize=11)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("时间延迟 (ns)")
    for ax in axes[:, 0]:
        ax.set_ylabel("归一化回波功率 (对清洁海面峰值)")
    fig.suptitle(f"2 km 方形礁石(+2 m)位于不同位置的回波（对照论文图 4.11, "
                 f"U19.5={U_COASTAL:.0f} m/s, 系综 M={m_ens}）", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "reef_position_scan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 场景 D: 岛屿位置扫描 =================
    print("\n[D] 岛屿位置扫描 (半径 1.5 km 圆, +12 m, U19.5=8, 宽窗, dB 尺度)")
    spec_d = land_spec("island", p.lam)
    sd2 = exp_slope_variance(spec_d["sigma"], spec_d["l"], 2 * np.pi / p.lam)
    print(f"    岛屿: sigma={spec_d['sigma']*1e3:.2f} mm, sig_s2={sd2:.3f}, "
          f"sigma0(0)={10*np.log10(fresnel_rho(spec_d['eps'])/(2*sd2)):.2f} dB (弱于海面)")
    island_wf = {}
    for r_km in [0.0, 2.2, 4.4, 6.6]:
        mask = disk_mask(X, Y, r_km * 1e3, 0.0, 1.5e3)
        s0 = hetero_sigma0(X, Y, mask, p_wide, U_COASTAL, 0.0, spec_d)
        wf = ensemble_waveform(p_wide, surfs, X, Y, dx, U_COASTAL,
                               lambda e, m=mask, s=s0: (e + m * spec_d["height"], s))
        island_wf[r_km] = wf
        print(f"    位置 {r_km:.1f} km: 完成")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True, sharey=True)
    for ax, r_km in zip(axes.ravel(), positions_km[:3] + [6.6]):
        wf = island_wf[r_km]
        ax.plot(t_wide, 10 * np.log10(np.maximum(wf / sea_ref, 1e-6)), "C0-", lw=1.6)
        ax.axvline(0.0, color="0.5", ls=":", lw=0.8)
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax.set_title(f"岛屿距天底点 {r_km:.1f} km", fontsize=11)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("时间延迟 (ns)")
    for ax in axes[:, 0]:
        ax.set_ylabel("回波功率 (dB, 相对清洁海面峰值)")
    ax.set_ylim(-45, 3)
    fig.suptitle(f"半径 1.5 km 圆形岛屿(+12 m)位于不同位置的回波（对照论文图 4.12, "
                 f"U19.5={U_COASTAL:.0f} m/s, dB 尺度显示弱岛屿回波）", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "island_position_scan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 场景 E: 盐度/温度扫描 =================
    print("\n[E] 海水盐度/温度 -> 介电常数 -> rho -> σ0(平台功率)")
    s_scan = [0, 10, 20, 30, 35, 40]
    t_scan = [-2, 0, 10, 20, 30]
    eps_s = [seawater_eps(s, 20, p.f) for s in s_scan]
    eps_t = [seawater_eps(35, t, p.f) for t in t_scan]
    rho0 = fresnel_rho(seawater_eps(35, 20, p.f))
    d_s = [10 * np.log10(fresnel_rho(e) / rho0) for e in eps_s]
    d_t = [10 * np.log10(fresnel_rho(e) / rho0) for e in eps_t]
    print(f"    基准 (35 permil, 20 degC): rho={rho0:.4f}")
    for s, e, d in zip(s_scan, eps_s, d_s):
        print(f"    S={s:2d} permil: eps=({e.real:6.2f},{e.imag:6.2f})  "
              f"rho={fresnel_rho(e):.5f}  dPlat={d:+.4f} dB")
    for t, e, d in zip(t_scan, eps_t, d_t):
        print(f"    T={t:2d} degC  : eps=({e.real:6.2f},{e.imag:6.2f})  "
              f"rho={fresnel_rho(e):.5f}  dPlat={d:+.4f} dB")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    ax = axes[0, 0]
    ax.plot(s_scan, [e.real for e in eps_s], "o-", color="C0", label="实部 ε'")
    axp = ax.twinx()
    axp.plot(s_scan, [e.imag for e in eps_s], "s--", color="C1", label="虚部 ε''")
    ax.set_xlabel("盐度 S (permil)"); ax.set_ylabel("ε'"); axp.set_ylabel("ε''")
    ax.set_title("海水介电常数随盐度 (T=20°C)")
    ax = axes[0, 1]
    ax.plot(t_scan, [e.real for e in eps_t], "o-", color="C0", label="实部 ε'")
    axp = ax.twinx()
    axp.plot(t_scan, [e.imag for e in eps_t], "s--", color="C1", label="虚部 ε''")
    ax.set_xlabel("温度 T (°C)"); ax.set_ylabel("ε'"); axp.set_ylabel("ε''")
    ax.set_title("海水介电常数随温度 (S=35‰)")
    ax = axes[1, 0]
    ax.plot(s_scan, d_s, "o-", color="C3")
    ax.set_xlabel("盐度 S (permil)")
    ax.set_ylabel("平台功率相对基准 (dB)")
    ax.set_title(f"回波平台随盐度（Ku 波段不敏感, 全程 <0.01 dB）")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(t_scan, d_t, "o-", color="C3")
    ax.set_xlabel("温度 T (°C)")
    ax.set_ylabel("平台功率相对基准 (dB)")
    ax.set_title("回波平台随温度（-2~30°C 约 0.2 dB）")
    ax.grid(True, alpha=0.3)
    for ax in axes.ravel():
        ax.ticklabel_format(useOffset=False)
    fig.suptitle("海况参数敏感性: Debye 介电常数(式 2.20-2.23) -> rho -> 回波平台 "
                 "(σ0∝rho, 波形形状不变)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "salinity_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 数据存档 =================
    np.savez_compressed(
        outdir / "coastal_echo_results.npz",
        t_ref=t_ref / 1e9, t_wide=t_wide / 1e9,
        fractions=np.array(fractions), beach_wf=np.stack([beach_wf[f] for f in fractions]),
        reef_pos=np.array(positions_km), reef_wf=np.stack([reef_wf[r] for r in positions_km]),
        island_pos=np.array([0.0, 2.2, 4.4, 6.6]),
        island_wf=np.stack([island_wf[r] for r in [0.0, 2.2, 4.4, 6.6]]),
        s_scan=np.array(s_scan), d_salt=np.array(d_s),
        t_scan=np.array(t_scan), d_temp=np.array(d_t),
    )
    print(f"\n完成: 5 张图 + coastal_echo_results.npz 已存至 {outdir}")
    print(f"总耗时 {time.time()-t_wall:.0f} s")


if __name__ == "__main__":
    main()
