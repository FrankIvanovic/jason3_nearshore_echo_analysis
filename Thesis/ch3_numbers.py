# -*- coding: utf-8 -*-
"""为论文第3章提取精确数值"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "Echo_Simulation"))
import numpy as np
from altimeter_params import AltimeterParams
from pm_surface import pm_swh, pm_sigma_h, u19_5_from_u10
from echo_kernel import slope_variances
from coastal import exp_slope_variance, land_spec
from dielectric import seawater_eps, fresnel_rho, LAND_EPS_BEACH, LAND_EPS_ISLAND

p = AltimeterParams()
lam = p.lam
print(f"lambda = {lam*100:.3f} cm, gate_dt = {p.gate_dt*1e9:.3f} ns, "
      f"window = [{p.t_start*1e9:+.0f}, {(p.t_start+(p.n_gates-1)*p.gate_dt)*1e9:+.0f}] ns, "
      f"tau0 = {p.tau0*1e3:.3f} ms")
print(f"rho_sea = {p.fresnel_rho:.4f}")
print(f"sigma_theta = {np.degrees(p.sigma_theta)*60:.2f} arcmin")

print("\n-- 风速组 --")
dx = 32000 / 2048
kmax = np.pi / dx
for u in [8.0, 12.0, 16.0]:
    swh = pm_swh(u)
    sh = pm_sigma_h(u)
    su2, sc2 = slope_variances(u)
    s0 = 10 * np.log10(p.fresnel_rho / (2 * np.sqrt(su2 * sc2)))
    capture = float(np.exp(-0.74 * 9.81**2 / (kmax**2 * u**4)))
    u10 = None
    lo, hi = 0.1, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if u19_5_from_u10(mid) < u: lo = mid
        else: hi = mid
    u10 = 0.5 * (lo + hi)
    print(f"U19.5={u:4.0f}: U10={u10:5.2f}, SWH={swh:5.3f} m, sigma_h={sh:.4f} m, "
          f"sigma_u2={su2:.5f}, sigma_c2={sc2:.5f}, sig0(0)={s0:5.2f} dB, capture={capture*100:.1f}%")

print("\n-- 陆地 --")
for kind in ["beach", "reef", "island"]:
    s = land_spec(kind, lam)
    sb2 = exp_slope_variance(s["sigma"], s["l"], 2 * np.pi / lam)
    rl = fresnel_rho(s["eps"])
    s0 = 10 * np.log10(rl / (2 * sb2))
    print(f"{kind}: eps=({s['eps'].real:.4f},{s['eps'].imag:.4f}) rho_land={rl:.4f} "
          f"sigma={s['sigma']*1e3:.2f} mm l={s['l']*1e2:.2f} cm sig_s2={sb2:.4g} "
          f"sig0(0)={s0:6.2f} dB height=+{s['height']} m")

print("\n-- 回波提前量 --")
for h in [0.5, 2.0, 12.0]:
    print(f"height +{h:4.1f} m -> advance {2*h/1e-9*1e0:.2f} ".replace("e-09","")
          + f"= {2*h/3e8*1e9:.2f} ns")

print("\n-- 介电常数基准与敏感性 --")
eps0 = seawater_eps(35, 20, p.f)
rho0 = fresnel_rho(eps0)
print(f"基准 S=35,T=20: eps=({eps0.real:.4f},{eps0.imag:.4f}) rho={rho0:.4f}")
for s in [0, 40]:
    e = seawater_eps(s, 20, p.f)
    print(f"  S={s}: dP={10*np.log10(fresnel_rho(e)/rho0):+.5f} dB")
for t in [-2, 30]:
    e = seawater_eps(35, t, p.f)
    print(f"  T={t}: dP={10*np.log10(fresnel_rho(e)/rho0):+.4f} dB")

print("\n-- 域几何 --")
print(f"L=32 km, N=2048, dx={dx:.3f} m, k_min={2*np.pi/32000:.2e} rad/m, k_max={kmax:.4f} rad/m")
print(f"-3dB 足印半径 = H*tan(1.1deg) = {p.H*np.tan(np.radians(1.1))/1e3:.2f} km")
print(f"脉冲受限足印半径 c*tau_pulse/2, tau_pulse=1/B={1/p.fs*1e9:.2f} ns -> {3e8/p.fs/2:.2f} m")
