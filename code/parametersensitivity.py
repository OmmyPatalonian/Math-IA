
import numpy as np
import matplotlib.pyplot as plt


def A(x, k=0.0346, x0=-30.37):
    return 100/(1+np.exp(-k*(x - x0)))

def thr_for_C(C, lam, mu, xmin=0, xmax=160):
    # different equation depending on if consistency >= 70
    rhs = (100 - mu*np.linspace(xmin,xmax,20001)) if C>=70 else (100 + lam*(70-C) - mu*np.linspace(xmin,xmax,20001))
    xs = np.linspace(xmin, xmax, 20001)
    diff = A(xs) - rhs
    s = np.where(np.sign(diff[:-1])<=0, 0, 1) 
    cross = np.where((diff[:-1]<=0) & (diff[1:]>0))[0]
    if len(cross)==0:
        return None
    i = cross[0]
    return xs[i]

Cs = np.linspace(50, 100, 51)

### test how mu affects thresholds
mus = [0.05, 0.10, 0.20]
plt.figure()
for mu in mus:
    xs = [thr_for_C(C, lam=1.0, mu=mu) for C in Cs]
    plt.plot(Cs, [np.nan if v is None else v for v in xs], label=f"mu={mu}")
plt.axvline(70, ls="--", lw=1)
plt.xlabel("Consistency C (%)"); plt.ylabel("Threshold WPM x_thr(C)")
plt.title("Threshold vs Consistency: varying μ (λ=1)")
plt.legend(); plt.tight_layout()
plt.savefig("parameter_sensitivity_varying_mu.png", dpi=300, bbox_inches='tight')
print("Saved 'parameter_sensitivity_varying_mu.png'")

### test how lambda affects thresholds
lams = [0.5, 1.0, 2.0]
plt.figure()
for lam in lams:
    xs = [thr_for_C(C, lam=lam, mu=0.10) for C in Cs]
    plt.plot(Cs, [np.nan if v is None else v for v in xs], label=f"lambda={lam}")
plt.axvline(70, ls="--", lw=1)
plt.xlabel("Consistency C (%)"); plt.ylabel("Threshold WPM x_thr(C)")
plt.title("Threshold vs Consistency: varying λ (μ=0.10)")
plt.legend(); plt.tight_layout()
plt.savefig("parameter_sensitivity_varying_lambda.png", dpi=300, bbox_inches='tight')
print("Saved 'parameter_sensitivity_varying_lambda.png'")

grid = []
for C in [55, 60, 65]:
    row = []
    for lam in lams:
        for mu in mus:
            row.append((lam, mu, thr_for_C(C, lam, mu)))
    grid.append((C, row))

print("No-solution scan (None means boundary has no solution on [0,160] WPM):")
for C, row in grid:
    print(f"C={C}%:")
    for lam, mu, thr in row:
        print(f"  lambda={lam:>3}, mu={mu:>4} -> x_thr={thr}")
plt.show()
