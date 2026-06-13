"""
Generates three figures for the Cognitive Debt paper:
  fig1_minsky.pdf  – Minsky divergence: pi_hat vs pi_true, leverage, AI intensity
  fig2_phase.pdf   – Phase portrait in (K, B) space with Minsky zones
  fig3_hetero.pdf  – Heterogeneous-agent inequality dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
})
BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A582"
GRAY   = "#888888"


# ── Structural parameters ─────────────────────────────────────────────────────
q      = 1.0    # AI quality
s_stress = 0.25 # stress discount on AI quality (G_tilde uses q*s)
phi    = 0.35   # learning-by-doing coefficient  (ell(1-a) = phi*(1-a))
eta    = 0.60   # debt convexity  (d(a) = eta*a^2/2)
delta  = 0.04   # cognitive depreciation
r_b    = 0.10   # cognitive debt compounding rate
beta   = 0.95   # discount factor
mu_k   = 1.0    # shadow value of cognitive capital
mu_b   = 2.5    # shadow value of cognitive debt
lam_b  = 0.10   # belief update rate
lam_pi = 0.55   # true crisis probability scale
gam_pi = 1.60   # crisis probability curvature (>1 → convex)
kappa  = 1.5    # debt-exposure coefficient


def opt_a(pi_hat, k):
    """
    Solve FOC analytically.
    FOC: k*q*(1 - pi_hat*(1-s)) / (1+a) = beta*(mu_k*phi + mu_b*eta*a)
    Rearranging: C*a^2 + (B+C)*a + (B - A) = 0
    where A = k*q*(1 - pi_hat*(1-s)), B = beta*mu_k*phi, C = beta*mu_b*eta
    """
    A = k * q * (1.0 - pi_hat * (1.0 - s_stress))
    B = beta * mu_k * phi
    C = beta * mu_b * eta
    discriminant = (B + C)**2 + 4.0 * C * (A - B)
    if discriminant < 0 or A <= 0:
        return 1e-4
    a = (-(B + C) + np.sqrt(discriminant)) / (2.0 * C)
    return float(np.clip(a, 1e-4, 0.999))


def true_pi(omega):
    return 1.0 - np.exp(-lam_pi * np.maximum(omega, 0)**gam_pi)


def simulate(T, k0, b0, pi_hat0, crisis_schedule=None):
    """Run T-period simulation. crisis_schedule is a set of periods with crises."""
    if crisis_schedule is None:
        crisis_schedule = set()
    k      = np.zeros(T + 1)
    b      = np.zeros(T + 1)
    pi_hat = np.zeros(T + 1)
    pi_tr  = np.zeros(T)
    a_star = np.zeros(T)
    omega  = np.zeros(T)

    k[0], b[0], pi_hat[0] = k0, b0, pi_hat0

    for t in range(T):
        omega[t]  = b[t] / max(k[t], 1e-6)
        pi_tr[t]  = true_pi(omega[t])
        a_star[t] = opt_a(pi_hat[t], k[t])
        crisis_t  = 1 if t in crisis_schedule else 0

        pi_hat[t + 1] = (1.0 - lam_b) * pi_hat[t] + lam_b * crisis_t
        k[t + 1]      = (1.0 - delta) * k[t] + phi * (1.0 - a_star[t])
        b[t + 1]      = (1.0 + r_b)  * b[t] + eta * a_star[t]**2 / 2.0

    return dict(k=k, b=b, pi_hat=pi_hat, pi_tr=pi_tr, a_star=a_star, omega=omega)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Minsky Divergence
# ═══════════════════════════════════════════════════════════════════════════════
T = 80
res = simulate(T, k0=1.0, b0=0.02, pi_hat0=0.30)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
t_ax = np.arange(T)

# Panel A: subjective vs true crisis probability
ax = axes[0]
ax.plot(t_ax, res["pi_hat"][:-1], color=BLUE,  label=r"Subjective risk $\hat{\pi}_t$",  lw=2)
ax.plot(t_ax, res["pi_tr"],       color=RED,   label=r"True fragility $\pi_t$",         lw=2, ls="--")
ax.axhline(0, color="gray", lw=0.6, ls=":")

# find crossing
diff = res["pi_tr"] - res["pi_hat"][:-1]
cross_idx = next((i for i in range(1, T) if diff[i] > 0 and diff[i-1] <= 0), None)
if cross_idx:
    ax.axvline(cross_idx, color=GRAY, lw=1.0, ls=":", alpha=0.7)
    ax.text(cross_idx + 1, 0.55, r"$T^*$", color=GRAY, fontsize=10)

ax.set_xlabel("Period $t$")
ax.set_ylabel("Probability")
ax.set_title("(a) Minsky Divergence", fontweight="bold")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(-0.02, 1.05)

# Panel B: cognitive leverage ratio
ax = axes[1]
ax.plot(t_ax, res["omega"], color=BLUE, lw=2)

# Minsky zone thresholds (illustrative)
om_spec  = 0.5
om_ponzi = 1.2
ax.axhline(om_spec,  color=GREEN,  lw=1.2, ls="--", alpha=0.8)
ax.axhline(om_ponzi, color=RED,    lw=1.2, ls="--", alpha=0.8)
ax.fill_between(t_ax, 0,        om_spec,                   color=GREEN,  alpha=0.07)
ax.fill_between(t_ax, om_spec,  om_ponzi,                  color=ORANGE, alpha=0.10)
ax.fill_between(t_ax, om_ponzi, res["omega"].max() * 1.05, color=RED,    alpha=0.08)

ymax = min(res["omega"].max() * 1.05, 4.0)
ax.text(T * 0.02, om_spec  * 0.4,    "Hedge",       color=GREEN, fontsize=9, alpha=0.9)
ax.text(T * 0.02, (om_spec + om_ponzi)/2, "Speculative", color="darkorange", fontsize=9, alpha=0.9)
ax.text(T * 0.02, om_ponzi * 1.07,   "Ponzi",        color=RED,   fontsize=9, alpha=0.9)

ax.set_xlabel("Period $t$")
ax.set_ylabel(r"Leverage $\Omega_t = B_t/K_t$")
ax.set_title("(b) Cognitive Leverage Ratio", fontweight="bold")
ax.set_ylim(-0.02, ymax)

# Panel C: AI substitution intensity
ax = axes[2]
ax.plot(t_ax, res["a_star"], color=BLUE, lw=2)
ax.set_xlabel("Period $t$")
ax.set_ylabel(r"AI substitution $a_t^*$")
ax.set_title("(c) Equilibrium AI Use", fontweight="bold")
ax.set_ylim(0, 1.0)

plt.tight_layout(pad=1.5)
fig.savefig("fig1_minsky.pdf", bbox_inches="tight")
fig.savefig("fig1_minsky.png", bbox_inches="tight", dpi=200)
plt.close()
print("Figure 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Phase Portrait in (K, B) space
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 5.5))

# ── Background Minsky zones ───────────────────────────────────────────────────
k_range = np.linspace(0.3, 1.6, 300)

b_hedge  = om_spec  * k_range
b_ponzi  = om_ponzi * k_range

ax.fill_between(k_range, 0,       b_hedge,  color=GREEN,  alpha=0.10, label="Hedge zone")
ax.fill_between(k_range, b_hedge, b_ponzi,  color=ORANGE, alpha=0.12, label="Speculative zone")
ax.fill_between(k_range, b_ponzi, b_ponzi * 3, color=RED, alpha=0.08, label="Ponzi zone")

ax.plot(k_range, b_hedge,  color=GREEN,       lw=1.3, ls="--", alpha=0.7,
        label=r"$\Omega = 0.50$ (Hedge $\to$ Spec.)")
ax.plot(k_range, b_ponzi,  color="darkorange", lw=1.3, ls="--", alpha=0.7,
        label=r"$\Omega = 1.20$ (Spec. $\to$ Ponzi)")

# ── Equilibrium trajectory ────────────────────────────────────────────────────
k_path = res["k"][:T]
b_path = res["b"][:T]

# color gradient to show time
n_seg = T - 1
from matplotlib.collections import LineCollection
points = np.array([k_path, b_path]).T.reshape(-1, 1, 2)
segs   = np.concatenate([points[:-1], points[1:]], axis=1)
lc     = LineCollection(segs, cmap="Blues", linewidth=2.4)
lc.set_array(np.linspace(0, 1, n_seg))
ax.add_collection(lc)

# start and end markers
ax.scatter(k_path[0],  b_path[0],  s=70, color=BLUE,  zorder=5, label="$t=0$")
ax.scatter(k_path[-1], b_path[-1], s=70, marker="s", color=BLUE, zorder=5, label=f"$t={T}$")

# Arrows to show direction
for frac in [0.15, 0.4, 0.65, 0.85]:
    idx = int(frac * (T - 2))
    dx  = k_path[idx+1] - k_path[idx]
    dy  = b_path[idx+1] - b_path[idx]
    ax.annotate("", xy=(k_path[idx]+dx*5, b_path[idx]+dy*5),
                xytext=(k_path[idx], b_path[idx]),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

ax.set_xlim(0.28, 1.65)
ax.set_ylim(-0.02, min(b_path.max() * 1.1, 3.5))
ax.set_xlabel(r"Cognitive Capital $K_t$")
ax.set_ylabel(r"Cognitive Debt $B_t$")
ax.set_title("Phase Portrait: Equilibrium Path Through Minsky Zones",
             fontweight="bold")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper right", frameon=True,
          framealpha=0.9, fontsize=9)

plt.tight_layout()
fig.savefig("fig2_phase.pdf", bbox_inches="tight")
fig.savefig("fig2_phase.png", bbox_inches="tight", dpi=200)
plt.close()
print("Figure 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Heterogeneous Agents: Cognitive Capital Inequality
# ═══════════════════════════════════════════════════════════════════════════════
T_h = 100

# Two types: H = high initial cognitive capital, L = low
kH0, kL0 = 1.6, 0.6
b0       = 0.01
pi0      = 0.25

res_H = simulate(T_h, k0=kH0, b0=b0, pi_hat0=pi0)
res_L = simulate(T_h, k0=kL0, b0=b0, pi_hat0=pi0)

t_h = np.arange(T_h)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# Panel A: Cognitive capital paths
ax = axes[0]
ax.plot(t_h, res_H["k"][:T_h], color=BLUE, lw=2,    label="High-$k$ type (H)")
ax.plot(t_h, res_L["k"][:T_h], color=RED,  lw=2, ls="--", label="Low-$k$ type (L)")
ax.set_xlabel("Period $t$")
ax.set_ylabel(r"Cognitive capital $k_t$")
ax.set_title("(a) Cognitive Capital Trajectories", fontweight="bold")
ax.legend(frameon=False)

# Panel B: Gap in cognitive capital (Delta k)
ax = axes[1]
delta_k = res_H["k"][:T_h] - res_L["k"][:T_h]
ax.plot(t_h, delta_k, color=BLUE, lw=2)
ax.axhline(0, color="gray", lw=1.0, ls="--")
# shade reversal region
reversal_idx = np.where(delta_k < 0)[0]
if len(reversal_idx) > 0:
    first_rev = reversal_idx[0]
    ax.axvline(first_rev, color=GRAY, lw=1.0, ls=":", alpha=0.8)
    ax.text(first_rev + 1, delta_k.min() * 0.5,
            r"Reversal $T^{**}$", color=GRAY, fontsize=9)
    ax.fill_between(t_h, delta_k, 0, where=(delta_k < 0),
                    color=RED, alpha=0.15, label="H type below L type")
ax.set_xlabel("Period $t$")
ax.set_ylabel(r"$k_{H,t} - k_{L,t}$")
ax.set_title("(b) Cognitive Capital Gap (Reversal of Fortune)", fontweight="bold")
ax.legend(frameon=False, fontsize=9)

# Panel C: AI substitution intensity comparison
ax = axes[2]
ax.plot(t_h, res_H["a_star"], color=BLUE, lw=2,    label="High-$k$ type (H)")
ax.plot(t_h, res_L["a_star"], color=RED,  lw=2, ls="--", label="Low-$k$ type (L)")
ax.set_xlabel("Period $t$")
ax.set_ylabel(r"AI substitution $a_t^*$")
ax.set_title("(c) Equilibrium AI Adoption by Type", fontweight="bold")
ax.legend(frameon=False)
ax.set_ylim(0, 1.0)

plt.tight_layout(pad=1.5)
fig.savefig("fig3_hetero.pdf", bbox_inches="tight")
fig.savefig("fig3_hetero.png", bbox_inches="tight", dpi=200)
plt.close()
print("Figure 3 saved.")

print("\nAll figures generated successfully.")
