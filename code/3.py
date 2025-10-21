
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

CSV_PATH = r"C:\Users\ompar\Downloads\math IA\Math IA - Sheet1.csv"

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["WPM","Accuracy","Consistency"])
x_min = max(0, float(df["WPM"].min()))
x_max = float(df["WPM"].max())

L_param = 100.0
k_param = 0.0346
x0_param = -30.37
lam = 2
mu = 0.1

def A_of_x(x):
    return L_param / (1.0 + np.exp(-k_param * (x - x0_param)))

def ramp(u):
    return np.maximum(0.0, u)

### loss function
def L_loss(x, C):
    return ramp(100.0 - A_of_x(x)) + lam * ramp(70.0 - C) - mu * x

x_grid = np.linspace(x_min, x_max, 400)
C_grid = np.linspace(50.0, 100.0, 400)
X, C = np.meshgrid(x_grid, C_grid)
L_vals = L_loss(X, C)

plt.figure(figsize=(7,5))
cs = plt.contourf(X, C, L_vals, levels=50)
c0 = plt.contour(X, C, L_vals, levels=[0.0], linewidths=2)
plt.clabel(c0, fmt={0.0:"L=0"}, inline=True, fontsize=9)
plt.xlabel("WPM (x)")
plt.ylabel("Consistency (C)")
plt.title("Acceptable Region (L(x,C) ≤ 0)")
plt.colorbar(cs, label="L(x,C)")
plt.tight_layout()
plt.savefig("figure_3_acceptable_region.png", dpi=300, bbox_inches='tight')
print("Saved Figure 3 as 'figure_3_acceptable_region.png'")

def threshold_x_for_C(C_val):
    def F_case3(x):
        return A_of_x(x) - (100.0 - mu*x)
    def F_case4(x):
        return A_of_x(x) - (100.0 + (70.0 - C_val) - mu*x)

    if C_val >= 70.0:
        F = F_case3
    else:
        F = F_case4

    lo = x_min
    hi = max(x_max, x_min + 150.0)

    f_lo = F(lo)
    f_hi = F(hi)

    if f_lo >= 0:
        return lo
    if f_hi <= 0:
        return np.nan
    try:
        return brentq(F, lo, hi, maxiter=200)
    except ValueError:
        return np.nan

Cs = np.arange(50.0, 100.1, 1.0)
x_thr = np.array([threshold_x_for_C(c) for c in Cs])

plt.figure(figsize=(7,5))
plt.plot(Cs, x_thr, linewidth=2)
mask = np.isnan(x_thr)
if mask.any():
    plt.scatter(Cs[mask], np.full(mask.sum(), np.nanmean(x_thr)), s=1) 
plt.xlabel("Consistency (C)")
plt.ylabel("Threshold WPM  x_thr(C)")
plt.title("Threshold Target Speed vs Consistency")
plt.tight_layout()
plt.savefig("figure_4_threshold_speed.png", dpi=300, bbox_inches='tight')
print("Saved Figure 4 as 'figure_4_threshold_speed.png'")

x_plot = np.linspace(x_min, x_max, 500)
C_values = [60.0, 70.0, 80.0, 90.0]
colors = ['red', 'green', 'orange', 'blue']
linestyles = ['-', '--', '-.', ':']

plt.figure(figsize=(7,5))
for i, C_val in enumerate(C_values):
    L_line = L_loss(x_plot, C_val)
    print(f"Plotting C={C_val}%, min L={L_line.min():.2f}, max L={L_line.max():.2f}")
    plt.plot(x_plot, L_line, linewidth=3, label=f"C={C_val:.0f}%", 
             color=colors[i], linestyle=linestyles[i], alpha=0.8)

    vals = L_line
    sign = np.sign(vals)
    idx = np.where(np.diff(sign) < 0)[0]  
    if len(idx) > 0:
        i_idx = idx[0]
        def F_local(x):
            return L_loss(x, C_val)
        lo = x_plot[i_idx]
        hi = x_plot[i_idx+1]
        try:
            xz = brentq(F_local, lo, hi, maxiter=200)
            plt.scatter([xz], [0.0], s=60, color=colors[i], edgecolor='black', linewidth=1)
            print(f"  Zero crossing at x={xz:.1f}")
        except ValueError:
            print(f"  No zero crossing found for C={C_val}")
            pass
    else:
        print(f"  No sign change found for C={C_val}")

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("WPM (x)")
plt.ylabel("Loss  L(x,C)")
plt.title("Loss Cross-sections at Fixed Consistency")
plt.legend()
plt.tight_layout()
plt.savefig("figure_5_loss_cross_sections.png", dpi=300, bbox_inches='tight')
print("Saved Figure 5 as 'figure_5_loss_cross_sections.png'")

plt.show()
