# Retracking — 海洋参数反演与误差分析

毕设核心环节：基于 `Echo_Simulation` 两期仿真波形（纯海面 24 实现 × 3 风速 +
近海复合场景）与 Jason-3 实测数据，反演 SSH / SWH / σ⁰-风速并做三层误差分析。

## 反演器

| 反演器 | 原理 | 参数 |
|---|---|---|
| 阈值法（`brown_fit_retracker.py`） | 50% 点定位 + 前沿 10-90 宽度换算，系数由 Brown 模型族标定（SSH/SWH 双标定消除系统差） | 无（查表） |
| Brown 拟合 | 数值 Brown 正向模型（平坦响应缓存）+ 最小二乘 (A, t0, σ_h)；阈值法粗估初始化 | SSH=t0·c/2, SWH=4σ_h |
| Hayne 二阶（`hayne_model.py`） | 论文式(5.7)-(5.15) 解析模型，ξ=0 时与数值 Brown 全窗 RMS 1.7%、前沿区 <0.5% | (A, t0, σ_c[, ξ]) |
| σ⁰→风速 | 平台窗功率 / σ⁰≡1 响应 → Cox-Munk 正演关系二分解 U | 闭环自洽 |

注意：误指向角 ξ 与 (A, t0, σ_c) 强耦合，单波形联合拟合不可辨识
（ξ=0.3° 的波形可被 σ_c +1.3 ns 完全模仿）——与 SGDR MLE4 不估 ξ 的
工程实践一致；脚本改用固定 σ_c/t0 的诊断式扫描。

## 三层误差分析（主脚本 `run_error_analysis.py`）

| 层 | 内容 | 主要结论（完整运行数字见 stdout） |
|---|---|---|
| 1 散斑随机误差 | 24 实现逐条反演 + bootstrap 平均门数收敛（对应 20Hz→1Hz） | SWH std 1.6-10 cm、SSH std 0.2-1.1 cm，按 ~1/√N 收敛；阈值法 SWH 散斑约为拟合 1.5 倍 |
| 2 近海畸变系统误差 | 沙滩占比/礁石/岛屿波形反演偏差（各自 baseline 相对化） | 礁石@天底点 ΔSSH +40 cm、岛屿@天底点 +78 cm、沙滩占比 ≥60% 时 ΔSSH −53 cm；**拟合残差 RMS 随污染激增 1.2%→7.4%**，可作波形质量自动判据 |
| 3 模型失配 | 长波倾斜波形用无倾斜模型反演；等效误指向角扫描 | ΔSSH +0.25 cm / ΔSWH −0.32 cm（可忽略）；ξ_eq≈0（低于可分辨水平） |

方法学系统差（如实报告）：Brown 形状模型拟合 PM 仿真系综波形时
SWH 低估 10-16%（阈值宽度法仅 −3%）——波形对 σ_h 的似然极平
（SWH 2.4-3.4 m 的全波形 RMS 仅差 0.2%），前缘前"提前回波"结构
比高斯波高 PDF 假设强 ~1% 所致。

## 实测对照（Jason-3 c603p147 闽浙沿岸, 2025-07-22）

- 开阔海（离岸>40 km）200 条 20Hz 波形，Hayne 拟合 vs SGDR MLE4 产品 SWH：
  bias −0.14 m、RMS 0.77 m、相关 0.95；
- 近海（0-10 km）波形拟合残差中位 10.9% vs 开阔海 8.1% —— 实测印证
  第 2 层的"残差判据"。

## 海况参数传播（σ⁰→风速）

盐度 0-40‰：|ΔU| ≤ 0.014 m/s（可忽略）；温度 −2~30 °C：|ΔU| ≤ 1.02 m/s
（需温度修正）。波形形状不受两者影响（GO 模型纯幅度缩放）。

## 使用

```bash
python run_error_analysis.py            # 完整 (约 3-4 min, 含实测 200 条)
python run_error_analysis.py --quick    # 冒烟 (bootstrap 20 次, 实测 60 条)
```

依赖: `Echo_Simulation/` 的两期 npz（`output/ocean_echo_results.npz`、
`output/coastal_echo_results.npz`）+ 实测文件（默认
`H:\毕业设计\Jason3_Data_Download\cycle_603\JA3_GPS_2PgP603_147_*.nc`，
可用 `--jason3` 覆盖）。

输出（`output/`）: `fig1_calibration.png`（标定+拟合自检）·
`fig2_speckle_error.png`（散斑收敛）· `fig3_nearshore_bias.png`（近海偏差+残差判据）·
`fig4_hayne_models.png`（解析模型验证+ξ 扫描）· `fig5_jason3_validation.png`（实测对照）·
`fig6_mismatch_wind.png`（模型失配+风速闭环）· `retracking_error_results.npz`。
