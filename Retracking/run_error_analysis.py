# -*- coding: utf-8 -*-
"""海洋参数反演与误差分析主脚本（毕设核心环节）。

数据源: Echo_Simulation 两期 npz（纯海面 24 实现 × 3 风速 + 近海复合场景）,
实测对照: Jason-3 SGDR-T c603p147（闽浙沿岸, 可 --jason3 换文件）。

部分:
  [1] 反演器标定与自检: 阈值法 SWH 标定曲线 + Brown 拟合示例
  [2] 第 1 层 散斑随机误差: 24 实现逐条反演 + 平均门数(bootstrap)收敛
  [3] 第 2 层 近海畸变系统误差: 沙滩占比/礁石/岛屿波形反演偏差 + 拟合残差判据
  [4] 第 3 层 模型失配误差: 长波倾斜波形的无倾斜模型反演 + Hayne 等效误指向角
  [5] Hayne 二阶模型验证(ξ=0 vs 数值 Brown)与误指向角扫描（对照论文图 5.1/5.3/5.4）
  [6] 实测对照: c603p147 开阔海 SWH 反演 vs SGDR MLE4 产品 + 近海残差沿轨变化
  [7] σ0 -> 风速反演 + 盐度/温度扰动传播

用法:
  python run_error_analysis.py            # 完整 (约 2 min)
  python run_error_analysis.py --quick    # 冒烟 (bootstrap 20 次, 实测 60 条)
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "Echo_Simulation"
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent / "Data_Reading"))

from altimeter_params import AltimeterParams, C
from pm_surface import generate_pm_surface, pm_swh
from echo_kernel import echo_fine_density, apply_ptr, density_to_gates

from brown_fit_retracker import (BrownFitModel, calibrate_threshold,
                                 threshold_retrieve, unit_sigma0_plateau,
                                 wind_from_waveform, sigma0_nadir)
from hayne_model import HayneModel

N_GRID = 2048
SEED0 = 2026
Q_FINE = 4
OUT = Path(__file__).resolve().parent / "output"


def simulate_gates(p, eta, X, Y, dx, u19_5, sigma0_map=None):
    dens = echo_fine_density(eta, X, Y, dx, p, u19_5, 0.0,
                             sigma0_map=sigma0_map, q=Q_FINE)
    return density_to_gates(apply_ptr(dens, p, Q_FINE), p, Q_FINE)


def fit_guesses(n_t0=3):
    """多初值: t0 ∈ {-5,0,5} ns × σ_h 档。"""
    return [np.array([1.0, dt, sh]) for dt in (-5.0, 0.0, 5.0)
            for sh in (0.3, 0.8, 1.4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jason3", default=r"H:\毕业设计\Jason3_Data_Download\cycle_603\JA3_GPS_2PgP603_147_20250722_025134_20250722_034730.nc")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    n_boot = 20 if args.quick else 100
    n_open = 60 if args.quick else 200
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    p = AltimeterParams()
    p_wide = replace(p, t_start=-150e-9)
    d1 = np.load(SIM / "output" / "ocean_echo_results.npz")
    d2 = np.load(SIM / "output" / "coastal_echo_results.npz")
    winds = d1["winds"]; wf_mean = d1["wf_mean"]; waveforms = d1["waveforms"]
    t_rel = d1["t_rel"]

    print("=" * 78)
    print("海洋参数反演与误差分析 (阈值法 + Brown 拟合 + Hayne 二阶模型)")
    print("=" * 78)
    t_wall = time.time()

    model = {u: BrownFitModel(p, u19_5=float(u)) for u in winds}
    model_wide = BrownFitModel(p_wide, u19_5=8.0)
    hayne = HayneModel(p)
    print("  正向模型缓存完成 (3 风速 + 宽窗)")

    # ============ [1] 标定与自检 ============
    print("\n[1] 反演器标定与自检")
    cal, cal_ssh, t_ref, (W_cal, swh_cal, t50_cal) = calibrate_threshold(model[8.0])
    print(f"    阈值法标定: SWH = {cal[0]:.4f}·W_10-90(ns) + {cal[1]:.4f} m; "
          f"SSH = {cal_ssh[0]:.4f}·t50(ns) + {cal_ssh[1]:.4f} m; "
          f"50%参考 t_ref = {t_ref*1e9:+.2f} ns")

    def smart_guess(wf, p_use):
        """阈值法粗估 -> 拟合初值（实用 retracker 流程: 阈值初始化 + 拟合精化）。"""
        _, swh_th, t50, _ = threshold_retrieve(wf, p_use, (cal, cal_ssh), t_ref)
        return fit_guesses() + [np.array([1.0, t50, max(swh_th / 4.0, 0.05)])]
    rows = {}
    for i, u in enumerate(winds):
        wf = wf_mean[i]
        r_fit = model[float(u)].fit(wf, guess_list=smart_guess(wf, p))
        ssh_fit = r_fit["t0"] * C / 2
        ssh_th, swh_th, _, _ = threshold_retrieve(wf, p, (cal, cal_ssh), t_ref)
        rows[u] = dict(ssh_fit=ssh_fit, swh_fit=r_fit["swh"],
                       resid=r_fit["resid_rms"], ssh_th=ssh_th, swh_th=swh_th)
        print(f"    U19.5={u:.0f} (SWH真值={pm_swh(u):.2f} m): "
              f"拟合 SWH={r_fit['swh']:.3f} SSH={ssh_fit*100:+.2f} cm | "
              f"阈值 SWH={swh_th:.3f} SSH={ssh_th*100:+.2f} cm | 残差={r_fit['resid_rms']*100:.1f}%")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.plot(W_cal, swh_cal, "o", color="C0", ms=5, label="Brown 模型族")
    wl = np.linspace(W_cal.min(), W_cal.max(), 10)
    ax1.plot(wl, cal[0] * wl + cal[1], "r--", lw=1.5,
             label=f"线性标定 SWH={cal[0]:.3f}W{cal[1]:+.2f}")
    ax1.set_xlabel("前沿 10%-90% 宽度 (ns)"); ax1.set_ylabel("SWH (m)")
    ax1.set_title("阈值法 SWH 标定曲线"); ax1.legend(); ax1.grid(alpha=0.3)
    wf_ex = wf_mean[1]
    r_ex = model[12.0].fit(wf_ex, guess_list=fit_guesses())
    wf_mod = model[12.0].waveform(r_ex["sigma_h"], r_ex["t0"], r_ex["A"])
    ax2.plot(t_rel * 1e9, wf_ex / wf_ex.max(), "b-", lw=1.8, label="仿真系综平均 (U19.5=12)")
    ax2.plot(t_rel * 1e9, wf_mod, "r--", lw=1.5, label="Brown 拟合")
    ax2.set_xlabel("时间延迟 (ns)"); ax2.set_ylabel("归一化功率")
    ax2.set_title(f"拟合自检: SWH={r_ex['swh']:.2f} m (真值 {pm_swh(12.0):.2f}), "
                  f"残差 {r_ex['resid_rms']*100:.1f}%")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ [2] 散斑随机误差 ============
    print("\n[2] 第 1 层 散斑随机误差 (24 实现逐条反演 + 平均门数收敛)")
    rng = np.random.default_rng(7)
    speck = {}
    for i, u in enumerate(winds):
        swhs, sshs, swhs_th = [], [], []
        for m in range(waveforms.shape[1]):
            wf = waveforms[i, m]
            r = model[float(u)].fit(wf, guess_list=smart_guess(wf, p))
            swhs.append(r["swh"]); sshs.append(r["t0"] * C / 2)
            swhs_th.append(threshold_retrieve(wf, p, (cal, cal_ssh), t_ref)[1])
        swhs, sshs, swhs_th = map(np.array, (swhs, sshs, swhs_th))
        speck[u] = dict(swh_std=swhs.std(), ssh_std=sshs.std(),
                        swh_th_std=swhs_th.std(),
                        swh_bias=swhs.mean() - pm_swh(float(u)),
                        ssh_bias=sshs.mean())
        print(f"    U19.5={u:.0f}: SWH std={swhs.std()*100:.1f} cm (阈值法 {swhs_th.std()*100:.1f}), "
              f"SSH std={sshs.std()*100:.1f} cm, bias(SWH)={speck[u]['swh_bias']*100:+.1f} cm")
    ks = [1, 2, 4, 8, 16]
    conv = {u: {"swh": [], "ssh": []} for u in winds}
    for i, u in enumerate(winds):
        for k in ks:
            e_swh, e_ssh = [], []
            for _ in range(n_boot):
                idx = rng.choice(waveforms.shape[1], k, replace=False)
                r = model[float(u)].fit(waveforms[i][idx].mean(0),
                                        guess_list=smart_guess(waveforms[i][idx].mean(0), p))
                e_swh.append(r["swh"]); e_ssh.append(r["t0"] * C / 2)
            conv[u]["swh"].append(np.std(e_swh))
            conv[u]["ssh"].append(np.std(e_ssh))
        print(f"    收敛 U19.5={u:.0f}: SWH std {conv[u]['swh'][0]*100:.1f}→{conv[u]['swh'][-1]*100:.1f} cm "
              f"(k=1→16), SSH {conv[u]['ssh'][0]*100:.1f}→{conv[u]['ssh'][-1]*100:.1f} cm")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(winds)))
    for c, u in zip(colors, winds):
        ax1.loglog(ks, np.array(conv[u]["swh"]) * 100, "o-", color=c, lw=1.8,
                   label=f"U19.5={u:.0f} m/s (SWH={pm_swh(float(u)):.1f} m)")
        ax2.loglog(ks, np.array(conv[u]["ssh"]) * 100, "o-", color=c, lw=1.8)
    kk = np.array([1.0, 16.0])
    for ax, lab in ((ax1, "SWH 单实现参考 25 cm"), (ax2, None)):
        ref = conv[winds[0]]["swh" if lab else "ssh"][0] * 100 / np.sqrt(kk / kk[0])
        ax.loglog(kk, ref, "k--", lw=1.2, label="1/√N 参考线" if lab else None)
    ax1.set_xlabel("平均实现数 k (对应 20Hz→1Hz 平均)"); ax1.set_ylabel("SWH 反演 std (cm)")
    ax1.set_title("SWH 散斑误差收敛"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")
    ax2.set_xlabel("平均实现数 k"); ax2.set_ylabel("SSH 反演 std (cm)")
    ax2.set_title("SSH 散斑误差收敛"); ax2.grid(alpha=0.3, which="both")
    fig.suptitle(f"第 1 层: 散斑随机误差 (bootstrap ×{n_boot})", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_speckle_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ [3] 近海畸变系统误差 ============
    print("\n[3] 第 2 层 近海畸变系统误差 (宽窗 baseline + 陆地污染波形反演)")
    m_ens = waveforms.shape[1]
    n_grid = N_GRID if not args.quick else 512
    axis = np.linspace(-32e3 / 2, 32e3 / 2, n_grid, endpoint=False)
    X, Y = np.meshgrid(axis, axis)
    dxg = 32e3 / n_grid
    acc = np.zeros(p.n_gates)
    for k in range(2 if args.quick else 6):
        eta, _ = generate_pm_surface(8.0, 32e3, n_grid, 0.0, seed=SEED0 + k)
        acc += simulate_gates(p_wide, eta, X, Y, dxg, 8.0)
        del eta
    base_wide = acc / (2 if args.quick else 6)
    r_base = model_wide.fit(base_wide, guess_list=smart_guess(base_wide, p_wide))
    ssh_b, swh_b = r_base["t0"] * C / 2, r_base["swh"]
    th_b = threshold_retrieve(base_wide, p_wide, (cal, cal_ssh), t_ref)
    print(f"    宽窗 baseline: SWH={swh_b:.3f} (真值 {pm_swh(8.0):.2f}), "
          f"SSH={ssh_b*100:+.2f} cm, 残差={r_base['resid_rms']*100:.1f}%")
    # beach 波形是标准窗 (-50 ns), 基线用第一期系综平均 wf_mean[0] (同为标准窗)
    r_std = model[8.0].fit(wf_mean[0], guess_list=smart_guess(wf_mean[0], p))
    th_std = threshold_retrieve(wf_mean[0], p, (cal, cal_ssh), t_ref)
    print(f"    标准窗 baseline (wf_mean[0]): SWH={r_std['swh']:.3f}, SSH={r_std['t0']*C/2*100:+.2f} cm")

    near = {"beach": [], "reef": [], "island": []}
    for j, f in enumerate(d2["fractions"]):
        wf = d2["beach_wf"][j]
        r = model[8.0].fit(wf, guess_list=smart_guess(wf, p))
        th = threshold_retrieve(wf, p, (cal, cal_ssh), t_ref)
        near["beach"].append((float(f), (r["t0"] - r_std["t0"]) * C / 2,
                              r["swh"] - r_std["swh"], r["resid_rms"],
                              (th[2] - th_std[2]) * 1e-9 * C / 2, th[1] - th_std[1]))
    for key, pos_key in (("reef", "reef_pos"), ("island", "island_pos")):
        for j, r_km in enumerate(d2[pos_key]):
            wf = d2[key + "_wf"][j]
            r = model_wide.fit(wf, guess_list=smart_guess(wf, p_wide))
            th = threshold_retrieve(wf, p_wide, (cal, cal_ssh), t_ref)
            near[key].append((float(r_km), (r["t0"] - r_base["t0"]) * C / 2,
                              r["swh"] - swh_b, r["resid_rms"],
                              (th[2] * 1e-9 - th_b[2] * 1e-9) * C / 2, th[1] - th_b[1]))
    for key in near:
        for row in near[key]:
            print(f"    {key:6s} {row[0]:5.2f}: ΔSSH={row[1]*100:+7.1f} cm "
                  f"(阈值 {row[4]*100:+7.1f})  ΔSWH={row[2]*100:+7.1f} cm "
                  f"(阈值 {row[5]*100:+7.1f})  残差={row[3]*100:5.1f}%")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax = axes[0, 0]
    fb = [r[0] * 100 for r in near["beach"]]
    ax.plot(fb, [r[1] * 100 for r in near["beach"]], "o-", color="C0", label="ΔSSH 拟合")
    ax.plot(fb, [r[2] * 100 for r in near["beach"]], "s-", color="C3", label="ΔSWH 拟合")
    ax.plot(fb, [r[4] * 100 for r in near["beach"]], "o--", color="C0", alpha=0.5, label="ΔSSH 阈值")
    ax.plot(fb, [r[5] * 100 for r in near["beach"]], "s--", color="C3", alpha=0.5, label="ΔSWH 阈值")
    ax.set_xlabel("沙滩占比 (%)"); ax.set_ylabel("反演偏差 (cm)")
    ax.set_title("沙滩条带 (远岸, 尾部污染)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for ax, key, lab in ((axes[0, 1], "reef", "2 km 礁石 (+2 m)"),
                         (axes[1, 0], "island", "1.5 km 岛屿 (+12 m)")):
        arr = near[key]
        ax.plot([r[0] for r in arr], [r[1] * 100 for r in arr], "o-", color="C0", label="ΔSSH 拟合")
        ax.plot([r[0] for r in arr], [r[2] * 100 for r in arr], "s-", color="C3", label="ΔSWH 拟合")
        ax.plot([r[0] for r in arr], [r[4] * 100 for r in arr], "o--", color="C0", alpha=0.5, label="ΔSSH 阈值")
        ax.set_xlabel("距天底点 (km)"); ax.set_ylabel("反演偏差 (cm)")
        ax.set_title(lab); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    labels = ["开阔海\nbaseline"] + [f"沙滩{f*100:.0f}%" for f in d2["fractions"][1:]] + \
             [f"礁石{r:.1f}km" for r in d2["reef_pos"]] + [f"岛屿{r:.1f}km" for r in d2["island_pos"]]
    vals = [r_base["resid_rms"]] + [r[3] for r in near["beach"][1:]] + \
           [r[3] for r in near["reef"]] + [r[3] for r in near["island"]]
    bars = ax.bar(range(len(vals)), np.array(vals) * 100,
                  color=["C2"] + ["C1"] * (len(vals) - 1))
    bars[0].set_color("C2")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=45, fontsize=7.5)
    ax.set_ylabel("Brown 拟合残差 RMS (%)")
    ax.set_title("拟合残差作为近海波形污染判据")
    ax.axhline(r_base["resid_rms"] * 100 * 2, color="k", ls=":", lw=1)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("第 2 层: 近海畸变引起的参数反演系统偏差 (U19.5=8 m/s)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_nearshore_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ [4] 模型失配 + [5] Hayne 验证 ============
    print("\n[4] 第 3 层 模型失配 (长波倾斜波形用无倾斜模型反演)")
    i12 = int(np.where(winds == 12.0)[0][0])
    r_clean = model[12.0].fit(wf_mean[i12], guess_list=fit_guesses())
    r_tilt = model[12.0].fit(d1["wf_tilt"], guess_list=fit_guesses())
    d_ssh = (r_tilt["t0"] - r_clean["t0"]) * C / 2
    d_swh = r_tilt["swh"] - r_clean["swh"]
    h_tilt = hayne.fit(d1["wf_tilt"], fit_xi=False, t0_init_ns=0.0,
                       guess_list=[np.array([1.0, dt, 6.0, 0.0]) for dt in (-5, 0, 5)])
    # 误指向角与 (A, t0, σ_c) 强耦合, 单波形联合拟合不可辨识（与 SGDR MLE4
    # 不估 ξ 的实践一致）。改为诊断式: 固定 σ_c 与 t0, 扫描 ξ 取残差最小者。
    wfn_t = d1["wf_tilt"] / d1["wf_tilt"].max()
    xi_grid = np.linspace(0.0, 1.0, 21)
    scan = []
    for xid in xi_grid:
        w = hayne.waveform(h_tilt["sigma_c"], h_tilt["t0"], np.deg2rad(xid))
        a_opt = float(np.dot(w, wfn_t) / np.dot(w, w))
        scan.append((float(xid), float(np.sqrt(np.mean((a_opt * w - wfn_t) ** 2)))))
    xi_eq, resid_eq = min(scan, key=lambda t: t[1])
    print(f"    忽略长波倾斜的偏差: ΔSSH={d_ssh*100:+.2f} cm, ΔSWH={d_swh*100:+.2f} cm")
    print(f"    等效误指向角扫描 (固定 σ_c={h_tilt['sigma_c']*1e9:.2f} ns, t0): "
          f"ξ_eq={xi_eq:.2f}°, 残差 {resid_eq*100:.2f}% (ξ=0 时 {scan[0][1]*100:.2f}%)")

    print("\n[5] Hayne 二阶模型验证与误指向角扫描")
    sig_c_ex = np.hypot(hayne.sigma_p, 2.0 * (3.07 / 4) / C)
    wf_an = hayne.waveform(sig_c_ex, r_clean["t0"])
    wf_an /= wf_an.max()
    wf_num = wf_mean[i12] / wf_mean[i12].max()
    rms_xi0 = np.sqrt(np.mean((wf_an - wf_num) ** 2))
    print(f"    ξ=0 二阶模型 vs 数值 Brown (SWH=3.07 m): 全窗 RMS={rms_xi0*100:.2f}%")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.plot(t_rel * 1e9, wf_num, "b-", lw=1.8, label="数值 Brown (仿真同链)")
    ax1.plot(t_rel * 1e9, wf_an, "r--", lw=1.5, label=f"Hayne 二阶 ξ=0 (RMS={rms_xi0*100:.1f}%)")
    ax1.set_xlabel("时间延迟 (ns)"); ax1.set_ylabel("归一化功率")
    ax1.set_title("解析模型 ξ=0 验证 (SWH=3.07 m)")
    ax1.legend(); ax1.grid(alpha=0.3)
    xis = [0.0, 0.1, 0.3, 0.5, 0.8]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.75, len(xis)))
    for c, xid in zip(cmap, xis):
        w = hayne.waveform(np.hypot(hayne.sigma_p, 2.0 * 0.75 / C), 0.0, np.deg2rad(xid))
        ax2.plot(t_rel * 1e9, w / w.max(), color=c, lw=1.6,
                 label=f"ξ={xid:.1f}°")
    ax2.set_xlabel("时间延迟 (ns)"); ax2.set_ylabel("归一化功率")
    ax2.set_title("误指向角扫描 (SWH=3 m, 对照论文图 5.1/5.4)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_hayne_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============ [7] σ0 -> 风速 ============
    print("\n[7] σ0 -> 风速反演 + 盐度/温度扰动")
    p_unit = unit_sigma0_plateau(p)
    u_est, s0_est = [], []
    for i, u in enumerate(winds):
        ue, s0e = wind_from_waveform(wf_mean[i], p, p_unit)
        u_est.append(ue); s0_est.append(s0e)
        s0_true_db = 10 * np.log10(sigma0_nadir(float(u), p.fresnel_rho))
        print(f"    U19.5={u:.0f}: σ0_est={10*np.log10(s0e):.2f} dB (正演 {s0_true_db:.2f}), "
              f"U_est={ue:.2f} m/s (真值 {u:.0f})")
    u_est = np.array(u_est)
    dU_salt = []
    for dd in d2["d_salt"]:
        dsig = 10 ** (dd / 10) - 1
        grad = (sigma0_nadir(9.0, p.fresnel_rho) - sigma0_nadir(7.0, p.fresnel_rho)) / 2.0
        dU_salt.append(s0_est[0] * dsig / abs(grad))
    dU_temp = []
    for dd in d2["d_temp"]:
        dsig = 10 ** (dd / 10) - 1
        grad = (sigma0_nadir(9.0, p.fresnel_rho) - sigma0_nadir(7.0, p.fresnel_rho)) / 2.0
        dU_temp.append(s0_est[0] * dsig / abs(grad))
    print(f"    盐度 0-40 permil 扰动: |ΔU| max = {np.max(np.abs(dU_salt)):.3f} m/s")
    print(f"    温度 -2~30 degC 扰动: |ΔU| max = {np.max(np.abs(dU_temp)):.3f} m/s")

    # ============ [6] 实测对照 ============
    print("\n[6] 实测对照: Jason-3 c603p147 (闽浙沿岸)")
    fig6 = None

    def est_t50_ns(wf):
        """波形 50% 上升点 (ns), 用于实测前缘位置初值（SGDR 前缘在门 ~47）。"""
        w = np.convolve(wf, np.ones(3) / 3.0, "same")
        w = w / w.max()
        ax_ns = (p.t_start + (np.arange(p.n_gates) + 0.5) * p.gate_dt) * 1e9
        i = int(np.argmax(w >= 0.5))
        if i == 0:
            return float(ax_ns[0])
        return float(ax_ns[i - 1] + (0.5 - w[i - 1]) / (w[i] - w[i - 1])
                     * (ax_ns[i] - ax_ns[i - 1]))

    def fit_obs(wf):
        """单条实测波形拟合: 前缘位置自适应初值与窗口。"""
        t50 = est_t50_ns(wf)
        return hayne.fit(wf, fit_xi=False,
                         t0_range_ns=(t50 - 25.0, t50 + 25.0),
                         t0_init_ns=t50)

    try:
        import h5py
        from jason3_reader import decode_variable, SURFACE_FLAG_MEANINGS
        fpath = args.jason3
        with h5py.File(fpath, "r") as f:
            d20 = f["data_20"]
            lat = decode_variable(d20["latitude"])
            surf = np.asarray(d20["surface_classification_flag"][...]).astype(np.int8)
            dtc = decode_variable(d20["distance_to_coast"])
            wf_all = decode_variable(d20["ku/power_waveform"]).astype(np.float64)
            swh_prod = decode_variable(d20["ku/swh_ocean"])
        noise = wf_all[:, :8].mean(1)
        wf_sub = wf_all - noise[:, None]
        open_m = (surf == 0) & (dtc > 40e3) & np.isfinite(swh_prod)
        idx_open = np.where(open_m)[0]
        print(f"    开阔海点 (surface=0, 离岸>40 km): {open_m.sum()} 个")
        if len(idx_open) > n_open:
            sel = np.linspace(idx_open[0], idx_open[-1], n_open).astype(int)
        else:
            sel = idx_open
        swh_est, swh_ref, resid_open = [], [], []
        for i in sel:
            if np.nanmax(wf_sub[i]) <= 0:
                continue
            r = fit_obs(wf_sub[i])
            swh_est.append(r["swh"]); swh_ref.append(swh_prod[i])
            resid_open.append(r["resid_rms"])
        swh_est, swh_ref = np.array(swh_est), np.array(swh_ref)
        good = np.isfinite(swh_est) & np.isfinite(swh_ref) & (swh_ref < 15)
        bias = np.mean(swh_est[good] - swh_ref[good])
        rms = np.sqrt(np.mean((swh_est[good] - swh_ref[good]) ** 2))
        cc = np.corrcoef(swh_est[good], swh_ref[good])[0, 1]
        print(f"    反演 {good.sum()} 条: SWH_est vs SGDR MLE4 产品: "
              f"bias={bias:+.3f} m, RMS={rms:.3f} m, 相关系数={cc:.3f}")

        near_m = (surf == 0) & (dtc > 0) & (dtc < 10e3)
        idx_near = np.where(near_m)[0]
        if len(idx_near) > 40:
            idx_near = idx_near[:: max(1, len(idx_near) // 40)]
        resid_near, dist_near, swh_near = [], [], []
        for i in idx_near:
            if np.nanmax(wf_sub[i]) <= 0:
                continue
            r = fit_obs(wf_sub[i])
            resid_near.append(r["resid_rms"]); dist_near.append(dtc[i] / 1e3)
            swh_near.append(r["swh"])
        resid_near = np.array(resid_near); dist_near = np.array(dist_near)
        print(f"    近海点 (0-10 km): {len(resid_near)} 条, "
              f"残差中位 {np.median(resid_near)*100:.1f}% vs 开阔海 {np.median(resid_open)*100:.1f}%")

        fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
        ax1.plot(swh_ref[good], swh_est[good], ".", color="C0", ms=5, alpha=0.6)
        lim = [0, max(6, np.nanmax(swh_ref[good]) * 1.05)]
        ax1.plot(lim, lim, "k--", lw=1.2, label="1:1")
        ax1.set_xlabel("SGDR MLE4 产品 SWH (m)"); ax1.set_ylabel("本文反演 SWH (m)")
        ax1.set_title(f"开阔海 {good.sum()} 条 20Hz 波形: bias={bias:+.2f} m, "
                      f"RMS={rms:.2f} m, r={cc:.3f}")
        ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xlim(lim); ax1.set_ylim(lim)
        order = np.argsort(dist_near)
        ax2.semilogx(np.maximum(dist_near[order], 0.1), resid_near[order] * 100,
                     "o-", color="C1", ms=4, lw=1.2, label="近海波形拟合残差")
        ax2.axhline(np.median(resid_open) * 100, color="C2", ls="--", lw=1.5,
                    label=f"开阔海基线 {np.median(resid_open)*100:.1f}%")
        ax2.set_xlabel("离岸距离 (km, 对数轴)"); ax2.set_ylabel("Hayne 拟合残差 RMS (%)")
        ax2.set_title("近海波形污染判据沿轨变化 (0-10 km)")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which="both")
        fig6.suptitle("实测对照: Jason-3 cycle 603 pass 147 (闽浙沿岸, 2025-07-22)", fontsize=13)
        fig6.tight_layout()
        fig6.savefig(outdir / "fig5_jason3_validation.png", dpi=150, bbox_inches="tight")
        plt.close(fig6)
        j3_stats = dict(bias=bias, rms=rms, cc=cc, n=int(good.sum()),
                        resid_open=float(np.median(resid_open)),
                        resid_near=float(np.median(resid_near)))
    except Exception as e:
        print(f"    [实测对照跳过: {e}]")
        j3_stats = None

    # ============ 汇总图: 模型失配 + 风速 ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.plot(t_rel * 1e9, d1["wf_tilt"] / d1["wf_tilt"].max(), "b-", lw=1.8,
             label="含长波倾斜系综 (两尺度)")
    ax1.plot(t_rel * 1e9, wf_mod / wf_mod.max(), "r--", lw=1.4, label="Brown 拟合 (无倾斜)")
    ax1.set_xlabel("时间延迟 (ns)"); ax1.set_ylabel("归一化功率")
    ax1.set_title(f"模型失配: ΔSSH={d_ssh*100:+.1f} cm, ΔSWH={d_swh*100:+.1f} cm, "
                  f"等效 ξ={xi_eq:.2f}°")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(winds, u_est, "o-", color="C0", lw=1.8, label="反演 U19.5")
    ax2.plot(winds, winds, "k--", lw=1.2, label="1:1")
    ax2.set_xlabel("U19.5 真值 (m/s)"); ax2.set_ylabel("U19.5 反演 (m/s)")
    ax2.set_title(f"σ0→风速闭环 (盐度扰动 ≤{np.max(np.abs(dU_salt)):.2f}, "
                  f"温度 ≤{np.max(np.abs(dU_temp)):.2f} m/s)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_mismatch_wind.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        outdir / "retracking_error_results.npz",
        winds=winds, cal=np.array(cal), t_ref=t_ref,
        swh_fit_mean=np.array([rows[float(u)]["swh_fit"] for u in winds]),
        ssh_fit_mean=np.array([rows[float(u)]["ssh_fit"] for u in winds]),
        speckle_swh_std=np.array([speck[float(u)]["swh_std"] for u in winds]),
        speckle_ssh_std=np.array([speck[float(u)]["ssh_std"] for u in winds]),
        conv_ks=np.array(ks),
        conv_swh=np.stack([conv[float(u)]["swh"] for u in winds]),
        conv_ssh=np.stack([conv[float(u)]["ssh"] for u in winds]),
        near_beach=np.array(near["beach"]),
        near_reef=np.array(near["reef"]),
        near_island=np.array(near["island"]),
        mismatch_ssh=d_ssh, mismatch_swh=d_swh,
        tilt_xi_eq=xi_eq, hayne_rms_xi0=rms_xi0,
        u_est=u_est, dU_salt=np.array(dU_salt), dU_temp=np.array(dU_temp),
        **({f"j3_{k}": v for k, v in j3_stats.items()} if j3_stats else {}),
    )
    print(f"\n完成: 6 张图 + retracking_error_results.npz -> {outdir}")
    print(f"总耗时 {time.time()-t_wall:.0f} s")


if __name__ == "__main__":
    main()
