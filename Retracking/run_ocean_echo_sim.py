# -*- coding: utf-8 -*-
"""卫星雷达高度计海面回波仿真主脚本（PM 谱海面 + 逐面元时域法）。

流程:
  1. 二维 PM 方向谱 -> FFT 线性滤波生成随机海面（论文式 2.14-2.17）
  2. 逐面元 GO 散射 + 雷达方程按双程延迟分箱 -> 单次实现回波（论文式 4.1-4.10）
  3. 与点目标响应 sinc^2 卷积 -> 104 门波形; M 次实现系综平均
  4. Brown 数值模型（冲激响应 * 波高PDF * PTR, 论文式 5.1-5.4）作解析对照
  5. 输出: 不同风速平均波形族、散斑示例、Brown 对照验证图、指标表

用法:
  python run_ocean_echo_sim.py            # 完整运行 (2048^2 网格 x 24 实现 x 3 风速, 约 2-3 min)
  python run_ocean_echo_sim.py --quick    # 冒烟测试 (512^2 x 4)
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from altimeter_params import AltimeterParams
from pm_surface import generate_pm_surface, pm_swh, u19_5_from_u10
from echo_kernel import (echo_fine_density, apply_ptr, density_to_gates,
                         slope_variances)
from brown_model import brown_waveform

# ---------------- 仿真配置 ----------------
WINDS_U195 = [8.0, 12.0, 16.0]   # 19.5 m 高度风速 (m/s)
PHI_V = 0.0                      # 风向 (rad), 沿 +x
L_DOMAIN = 32e3                  # 海面域边长 (m): 覆盖 2.2 deg 波束 -3dB 半径的一部分
N_GRID = 2048                    # 每边采样数, dx = 15.625 m
M_ENSEMBLE = 24                  # 系综平均实现数
SEED0 = 2026
VALID_U = 12.0                   # 对照验证所用风速
Q_FINE = 4                       # 每距离门精细分箱数
OUT = Path(__file__).resolve().parent / "output"


def simulate_one(p, eta, X, Y, u19_5, phi_v, tilt=False):
    """单个海面实现的 104 门回波波形。"""
    dens = echo_fine_density(eta, X, Y, L_DOMAIN / eta.shape[0], p,
                             u19_5, phi_v, use_local_tilt=tilt, q=Q_FINE)
    dens = apply_ptr(dens, p, Q_FINE)
    return density_to_gates(dens, p, Q_FINE)


def run_ensemble(p, u19_5, n_grid, m_ens, phi_v=PHI_V, tilt=False, seed0=SEED0):
    """M 个 PM 海面实现的回波系综。返回 (波形[M,104], meta, 海面样例, std列表)。"""
    dx = L_DOMAIN / n_grid
    axis = np.linspace(-L_DOMAIN / 2.0, L_DOMAIN / 2.0, n_grid, endpoint=False)
    X, Y = np.meshgrid(axis, axis)
    wf = np.zeros((m_ens, p.n_gates))
    stds = np.zeros(m_ens)
    sample = None
    meta = None
    for m in range(m_ens):
        eta, meta = generate_pm_surface(u19_5, L_DOMAIN, n_grid, phi_v,
                                        seed=seed0 + m)
        wf[m] = simulate_one(p, eta, X, Y, u19_5, phi_v, tilt)
        stds[m] = np.std(eta)
        if m == 0:
            sample = eta[::8, ::8].copy()
        del eta
    return wf, meta, sample, stds


def edge_metrics(t_rel, wf):
    """(平台功率, 前沿 10%-90% 宽度 ns, 前沿 50% 位置 ns)。"""
    sel = (t_rel >= 120e-9) & (t_rel <= 260e-9)
    plateau = float(wf[sel].mean())
    wfn = wf / plateau
    ns = t_rel * 1e9

    def cross(level):
        i = int(np.argmax(wfn >= level))
        if i == 0:
            return float(ns[0])
        x0, x1 = ns[i - 1], ns[i]
        y0, y1 = wfn[i - 1], wfn[i]
        return float(x0 + (level - y0) / (y1 - y0) * (x1 - x0))

    t10, t90, t50 = cross(0.1), cross(0.9), cross(0.5)
    return plateau, t90 - t10, t50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小网格快速冒烟测试")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    n_grid = 512 if args.quick else N_GRID
    m_ens = 4 if args.quick else M_ENSEMBLE
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = AltimeterParams()
    print("=" * 78)
    print("卫星雷达高度计海面回波仿真 (PM 谱海面 + 逐面元时域法 + Brown 对照)")
    print("=" * 78)
    print(f"  H={p.H/1e3:.0f} km  f={p.f/1e9:.3f} GHz  lambda={p.lam*100:.2f} cm  "
          f"Pt={p.Pt:.0f} W  波束={p.beam_deg:.1f} deg")
    print(f"  门数={p.n_gates}  门间隔={p.gate_dt*1e9:.3f} ns  "
          f"窗=[{p.t_start*1e9:+.0f}, {(p.t_start+(p.n_gates-1)*p.gate_dt)*1e9:+.0f}] ns  "
          f"rho_Fresnel={p.fresnel_rho:.4f}")
    print(f"  海面域 {L_DOMAIN/1e3:.0f} km x {L_DOMAIN/1e3:.0f} km, "
          f"N={n_grid} (dx={L_DOMAIN/n_grid:.2f} m), 系综 M={m_ens}, 风向 {np.degrees(PHI_V):.0f} deg")
    if args.quick:
        print("  [quick 模式: 网格粗, capture 低, 仅用于流程自检, 指标无物理意义]")

    t_wall = time.time()
    results = {}
    for iw, u195 in enumerate(WINDS_U195):
        t0 = time.time()
        wf, meta, sample, stds = run_ensemble(p, u195, n_grid, m_ens)
        wf_mean = wf.mean(axis=0)
        brown = brown_waveform(p, u195, PHI_V, meta["sigma_pm"],
                               L_DOMAIN, n_flat=512 if not args.quick else 256,
                               q=Q_FINE)
        results[u195] = dict(wf=wf, wf_mean=wf_mean, brown=brown,
                             meta=meta, sample=sample, stds=stds)
        print(f"  U19.5={u195:4.0f} m/s: SWH={meta['swh']:.2f} m  "
              f"capture={meta['capture']*100:4.1f}%  "
              f"sigma_h 实现均值={stds.mean():.3f} m (理论 {meta['sigma_pm']:.3f})  "
              f"[{time.time()-t0:.1f} s]")

    # ---- 对照验证: VALID_U 风速下再跑一组带长波倾斜的实现 ----
    t0 = time.time()
    wf_tilt, _, _, _ = run_ensemble(p, VALID_U, n_grid, m_ens, tilt=True,
                                    seed0=SEED0 + 7000)
    results[VALID_U]["wf_tilt"] = wf_tilt.mean(axis=0)
    print(f"  对照组 (长波倾斜调制) U19.5={VALID_U:.0f} m/s "
          f"[{time.time()-t0:.1f} s]")

    t_rel = p.gate_times()
    t_ns = t_rel * 1e9

    # ---------------- 指标表 ----------------
    def u10_from_u19_5(u195):
        lo, hi = 0.1, 60.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if u19_5_from_u10(mid) < u195:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print("\n" + "-" * 78)
    print(f"{'U19.5':>6} {'U10':>6} {'SWH/m':>6} {'sig0(0)/dB':>10} "
          f"{'平台(sim)':>11} {'平台(Brown)':>11} {'前沿10-90ns':>14} {'前沿50%位置':>12}")
    for u195 in WINDS_U195:
        r = results[u195]
        u10 = u10_from_u19_5(u195)
        ps, ws, ms_ = edge_metrics(t_rel, r["wf_mean"])
        pb, wb, mb = edge_metrics(t_rel, r["brown"])
        sig_u2, sig_c2 = slope_variances(u195)
        s0_nadir_db = 10 * np.log10(p.fresnel_rho / (2 * np.sqrt(sig_u2 * sig_c2)))
        print(f"{u195:6.0f} {u10:6.1f} {r['meta']['swh']:6.2f} {s0_nadir_db:10.2f} "
              f"{ps:11.3e} {pb:11.3e} {ws:6.2f}/{wb:6.2f} {ms_:7.2f}/{mb:6.2f}")
    print("-" * 78)
    print("(前沿宽度/50%位置: sim/Brown; 平台功率为任意单位, 量级比即 sig0 风速效应)")

    # ---------------- 图 1: PM 海面样例 ----------------
    fig = plt.figure(figsize=(18, 6))
    for i, u195 in enumerate(WINDS_U195):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        s = results[u195]["sample"]
        xs = np.arange(s.shape[1]) * (L_DOMAIN / n_grid) * 8 / 1e3
        ys = np.arange(s.shape[0]) * (L_DOMAIN / n_grid) * 8 / 1e3
        Xs, Ys = np.meshgrid(xs, ys)
        ax.plot_surface(Xs, Ys, s, cmap="jet", rstride=1, cstride=1,
                        linewidth=0, antialiased=True)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_zlabel("\u03b7 (m)")
        ax.set_title(f"U19.5={u195:.0f} m/s, SWH={results[u195]['meta']['swh']:.2f} m")
        ax.view_init(elev=35, azim=-60)
        ax.ticklabel_format(style="plain")
    fig.suptitle("二维 PM 方向谱随机海面 (FFT 线性滤波, 式 2.14-2.17)", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "pm_surfaces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 2: 不同风速平均波形族 ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(WINDS_U195)))
    for c, u195 in zip(colors, WINDS_U195):
        r = results[u195]
        plateau = edge_metrics(t_rel, r["wf_mean"])[0]
        pb = edge_metrics(t_rel, r["brown"])[0]
        ax1.semilogy(t_ns, r["wf_mean"] / plateau, color=c, lw=2,
                     label=f"U19.5={u195:.0f} m/s (SWH={r['meta']['swh']:.2f} m)")
        ax1.semilogy(t_ns, r["brown"] / pb, color=c, lw=1.2, ls="--", alpha=0.65)
        ax2.plot(t_ns, r["wf_mean"] / plateau, color=c, lw=2)
        ax2.plot(t_ns, r["brown"] / pb, color=c, lw=1.2, ls="--", alpha=0.65)
    ax1.plot([], [], "k--", lw=1.2, label="Brown 模型 (虚线)")
    ax1.set_xlabel("时间延迟 (ns)"); ax1.set_ylabel("归一化回波功率")
    ax1.set_title("平均回波波形 - 不同风速 (全窗, 对数尺度)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("时间延迟 (ns)"); ax2.set_ylabel("归一化回波功率")
    ax2.set_title("上升沿放大 (线性尺度): SWH 越大前沿越缓")
    ax2.grid(True, alpha=0.3); ax2.set_xlim(-15, 80)
    fig.suptitle("高度计海面回波: 系综平均波形 (实线=逐面元仿真, 虚线=Brown 模型)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "echo_wind_family.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 3: 散斑与系综平均 ----------------
    r = results[VALID_U]
    plateau = edge_metrics(t_rel, r["wf_mean"])[0]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in range(min(6, r["wf"].shape[0])):
        ax.plot(t_ns, r["wf"][m] / plateau, color="0.75", lw=0.8, alpha=0.9)
    ax.plot([], [], color="0.75", lw=1, label="单次实现 (散斑)")
    ax.plot(t_ns, r["wf_mean"] / plateau, "b-", lw=2.2,
            label=f"系综平均 (M={r['wf'].shape[0]})")
    ax.plot(t_ns, r["brown"] / edge_metrics(t_rel, r["brown"])[0], "k--", lw=1.5,
            label="Brown 模型")
    ax.set_xlabel("时间延迟 (ns)"); ax.set_ylabel("归一化回波功率")
    ax.set_title(f"单次实现散斑 vs 系综平均 (U19.5={VALID_U:.0f} m/s)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(-55, 285)
    fig.tight_layout()
    fig.savefig(outdir / "echo_speckle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 4: Brown 对照验证 ----------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                   height_ratios=[2.2, 1])
    pb = edge_metrics(t_rel, r["brown"])[0]
    ps = edge_metrics(t_rel, r["wf_mean"])[0]
    pt = edge_metrics(t_rel, r["wf_tilt"])[0]
    ax1.plot(t_ns, r["brown"] / pb, "k--", lw=1.8, label="Brown 数值模型")
    ax1.plot(t_ns, r["wf_mean"] / ps, "b-", lw=2,
             label="逐面元系综平均 (几何入射角)")
    ax1.plot(t_ns, r["wf_tilt"] / pt, "C1-", lw=2,
             label="逐面元系综平均 (含长波倾斜调制)")
    ax1.set_ylabel("归一化回波功率")
    ax1.set_title(f"仿真正确性对照验证 (U19.5={VALID_U:.0f} m/s, SWH={r['meta']['swh']:.2f} m)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(t_ns, (r["wf_mean"] / ps) / np.maximum(r["brown"] / pb, 1e-12), "b-", lw=1.8,
             label="仿真/Brown")
    ax2.plot(t_ns, (r["wf_tilt"] / pt) / np.maximum(r["brown"] / pb, 1e-12), "C1-", lw=1.8,
             label="含倾斜/Brown")
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_xlabel("时间延迟 (ns)"); ax2.set_ylabel("比值")
    ax2.set_ylim(0.6, 1.4); ax2.grid(True, alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig(outdir / "validation_u12.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 数据存档 ----------------
    np.savez_compressed(
        outdir / "ocean_echo_results.npz",
        t_rel=t_rel,
        winds=np.array(WINDS_U195),
        waveforms=np.stack([results[u]["wf"] for u in WINDS_U195]),
        wf_mean=np.stack([results[u]["wf_mean"] for u in WINDS_U195]),
        brown=np.stack([results[u]["brown"] for u in WINDS_U195]),
        wf_tilt=results[VALID_U]["wf_tilt"],
    )

    print(f"\n完成: 4 张图 + ocean_echo_results.npz 已存至 {outdir}")
    print(f"总耗时 {time.time()-t_wall:.0f} s")


if __name__ == "__main__":
    main()
