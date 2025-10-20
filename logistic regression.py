#!/usr/bin/env python3
# Plot the logistic accuracy model A(x) = L / (1 + e^{-k(x - x0)})
# and annotate anchors, 95%/98% accuracy speeds, and A'(x)=0.1%/WPM.

import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters (from your derivation) -----
L  = 100.0
k  = 0.0346
x0 = -30.37

# ----- Logistic and derivative -----
def A(x, L=L, k=k, x0=x0):
    return L / (1.0 + np.exp(-k * (x - x0)))

def A_prime(x, L=L, k=k, x0=x0):
    z = np.exp(-k * (x - x0))
    return (L * k * z) / (1.0 + z)**2

# Inverse logistic for target accuracy (in %)
def x_at_target(target, L=L, k=k, x0=x0):
    # target = L / (1 + e^{-k(x-x0)})  =>  x = x0 - (1/k)*ln(L/target - 1)
    return x0 - (1.0 / k) * np.log(L / target - 1.0)

# Solve A'(x) = thresh (thresh in % per WPM) analytically
def x_at_derivative(thresh, L=L, k=k, x0=x0):
    # Let z = e^{-k(x-x0)}. A'(x) = L k z / (1+z)^2 = thresh
    # => thresh*z^2 + (2*thresh - L*k)*z + thresh = 0
    a = thresh
    b = 2.0 * thresh - L * k
    c = thresh
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)
    # z must be > 0; pick the positive root
    z_candidates = [r for r in (r1, r2) if r > 0]
    if not z_candidates:
        return None
    z = min(z_candidates)  # pick smaller positive z (gives larger x)
    return x0 - (1.0 / k) * np.log(z)

# ----- Compute reference points -----
anchors = [(30, 89), (70, 97)]
x95 = x_at_target(95.0)
x98 = x_at_target(98.0)
x_thresh = x_at_derivative(0.1)  # 0.1 %/WPM

# ----- Plot -----
x = np.linspace(0, 150, 1000)
y = A(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$A(x)=\dfrac{L}{1+e^{-k(x-x_0)}}$")
plt.xlabel("Typing speed x (WPM)")
plt.ylabel("Accuracy A(x) (%)")
plt.title("Logistic Accuracy Model vs. Typing Speed")

# Anchors
for xa, ya in anchors:
    plt.scatter([xa], [ya])
    plt.annotate(f"({xa}, {ya}%)", xy=(xa, ya), xytext=(6, 8), textcoords="offset points")

# 95% and 98% lines
plt.axhline(95, linestyle=":", linewidth=1)
plt.axhline(98, linestyle=":", linewidth=1)
plt.annotate(f"95% at x ≈ {x95:.1f}", xy=(x95, 95), xytext=(6, -15), textcoords="offset points")
plt.annotate(f"98% at x ≈ {x98:.1f}", xy=(x98, 98), xytext=(6, -15), textcoords="offset points")

# Derivative threshold
if x_thresh is not None:
    plt.axvline(x_thresh, linestyle="--", linewidth=1)
    plt.annotate(f"A'(x)=0.1 at x ≈ {x_thresh:.1f}",
                 xy=(x_thresh, A(x_thresh)), xytext=(6, -15), textcoords="offset points")

plt.xlim(0, 150)
plt.ylim(80, 100.5)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()#!/usr/bin/env python3
# Plot the logistic accuracy model A(x) = L / (1 + e^{-k(x - x0)})
# and annotate anchors, 95%/98% accuracy speeds, and A'(x)=0.1%/WPM.

import numpy as np
import matplotlib.pyplot as plt

# ----- Parameters (from your derivation) -----
L  = 100.0
k  = 0.0346
x0 = -30.37

# ----- Logistic and derivative -----
def A(x, L=L, k=k, x0=x0):
    return L / (1.0 + np.exp(-k * (x - x0)))

def A_prime(x, L=L, k=k, x0=x0):
    z = np.exp(-k * (x - x0))
    return (L * k * z) / (1.0 + z)**2

# Inverse logistic for target accuracy (in %)
def x_at_target(target, L=L, k=k, x0=x0):
    # target = L / (1 + e^{-k(x-x0)})  =>  x = x0 - (1/k)*ln(L/target - 1)
    return x0 - (1.0 / k) * np.log(L / target - 1.0)

# Solve A'(x) = thresh (thresh in % per WPM) analytically
def x_at_derivative(thresh, L=L, k=k, x0=x0):
    # Let z = e^{-k(x-x0)}. A'(x) = L k z / (1+z)^2 = thresh
    # => thresh*z^2 + (2*thresh - L*k)*z + thresh = 0
    a = thresh
    b = 2.0 * thresh - L * k
    c = thresh
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)
    # z must be > 0; pick the positive root
    z_candidates = [r for r in (r1, r2) if r > 0]
    if not z_candidates:
        return None
    z = min(z_candidates)  # pick smaller positive z (gives larger x)
    return x0 - (1.0 / k) * np.log(z)

# ----- Compute reference points -----
anchors = [(30, 89), (70, 97)]
x95 = x_at_target(95.0)
x98 = x_at_target(98.0)
x_thresh = x_at_derivative(0.1)  # 0.1 %/WPM

# ----- Plot -----
x = np.linspace(0, 150, 1000)
y = A(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$A(x)=\dfrac{L}{1+e^{-k(x-x_0)}}$")
plt.xlabel("Typing speed x (WPM)")
plt.ylabel("Accuracy A(x) (%)")
plt.title("Logistic Accuracy Model vs. Typing Speed")

# Anchors
for xa, ya in anchors:
    plt.scatter([xa], [ya])
    plt.annotate(f"({xa}, {ya}%)", xy=(xa, ya), xytext=(6, 8), textcoords="offset points")

# 95% and 98% lines
plt.axhline(95, linestyle=":", linewidth=1)
plt.axhline(98, linestyle=":", linewidth=1)
plt.annotate(f"95% at x ≈ {x95:.1f}", xy=(x95, 95), xytext=(6, -15), textcoords="offset points")
plt.annotate(f"98% at x ≈ {x98:.1f}", xy=(x98, 98), xytext=(6, -15), textcoords="offset points")

# Derivative threshold
if x_thresh is not None:
    plt.axvline(x_thresh, linestyle="--", linewidth=1)
    plt.annotate(f"A'(x)=0.1 at x ≈ {x_thresh:.1f}",
                 xy=(x_thresh, A(x_thresh)), xytext=(6, -15), textcoords="offset points")

plt.xlim(0, 150)
plt.ylim(80, 100.5)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
