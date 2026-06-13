# Cognitive Debt: AI as Intellectual Leverage and the Dynamics of Systemic Fragility

**Working Paper** | New York University | June 2026

---

## Overview

This paper develops a formal economic theory of **cognitive debt** — the stock of unverified reasoning obligations that accumulates when individuals use AI as a substitute for first-principles cognition. The framework maps the Minsky financial instability hypothesis onto the cognitive domain, with cognitive capital serving as collateral and AI substitution intensity as leverage.

The central finding: rational agents optimally incur cognitive debt because costs are deferred and partially external. Tranquil periods of AI-driven productivity systematically build the conditions for cognitive crises — a **cognitive Minsky moment** in which perceived risk falls while true systemic fragility rises.

---

## Key Results

| Proposition | Content |
|---|---|
| **P1** | Rational agents incur positive cognitive debt; AI adoption rises with AI quality, cognitive capital, and output pressure |
| **P2** | Cognitive Minsky moment: tranquil periods drive $\hat\pi_t \downarrow$ while $\pi_t \uparrow$ |
| **P3** | Crisis losses are convex in leverage — high-leverage economies are disproportionately fragile |
| **P4** | False-correction loop: post-crisis output pressure leads agents to patch AI failures with more AI |
| **P5** | Decentralised equilibrium over-adopts AI; optimal policy is a Pigouvian tax indexed to aggregate leverage |
| **P6** | Reversal of fortune: high-capital agents adopt AI more aggressively, erode capital faster, and eventually fall behind |

---

## Model Structure

**State variables per agent:**
- $k_{it}$ — cognitive capital (unaided reasoning, verification, transfer ability)
- $b_{it}$ — cognitive debt (unserviced reasoning obligations)

**Production technology:**
$$y_{it}^N = k_{it} \cdot G(a_{it};\, q_t)$$

Cognitive capital is the *collateral*: the marginal product of AI is $k \cdot G_a$, strictly proportional to $k$. When $k = 0$, AI yields zero output.

**Capital dynamics:**
$$k_{i,t+1} = (1-\delta)\,k_{it} + \ell(1-a_{it}) + \nu x_{it}$$
$$b_{i,t+1} = (1+r_b)\,b_{it} + d(a_{it}) - \rho x_{it}$$

**Minsky taxonomy:**
- *Hedge cognition*: $k \geq b +$ task demand
- *Speculative cognition*: viable only by rolling over AI reliance
- *Ponzi cognition*: new debt + compounding exceeds repayment capacity

---

## Figures

| Figure | Description |
|---|---|
| `fig1_minsky` | Minsky divergence ($\hat\pi_t$ vs $\pi_t$), leverage ratio through Hedge/Speculative/Ponzi zones, equilibrium AI intensity |
| `fig2_phase` | Phase portrait in $(K, B)$ space — equilibrium trajectory with Minsky zone boundaries |
| `fig3_hetero` | Two-type heterogeneous economy: capital trajectories, gap reversal at $T^{**}$, AI adoption by type |

---

## Repository Structure

```
.
├── main.tex          # Full paper source (LaTeX)
├── refs.bib          # Bibliography (30 entries)
├── figures.py        # Figure generation script (Python/matplotlib)
├── main.pdf          # Compiled manuscript (41 pages)
├── fig1_minsky.pdf   # Figure 1
├── fig2_phase.pdf    # Figure 2
├── fig3_hetero.pdf   # Figure 3
└── README.md
```

---

## Reproducing Figures

Requires Python 3 with `numpy`, `scipy`, and `matplotlib`.

```bash
python3 figures.py
```

Outputs `fig1_minsky.pdf`, `fig2_phase.pdf`, `fig3_hetero.pdf` (and `.png` versions).

## Compiling the Paper

Requires a standard LaTeX distribution (TeX Live / MacTeX) with `natbib`.

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Citation

```bibtex
@unpublished{CognitivDebt2026,
  author = {[Author]},
  title  = {Cognitive Debt: {AI} as Intellectual Leverage and the Dynamics of Systemic Fragility},
  year   = {2026},
  note   = {Working Paper, New York University}
}
```

---

## Related References

- Noy & Zhang (2023) — *Experimental Evidence on the Productivity Effects of Generative AI*, Science
- Dell'Acqua et al. (2023) — *Navigating the Jagged Technological Frontier*, HBS Working Paper
- Shumailov et al. (2024) — *The Curse of Recursion: Training on Generated Data Makes Models Forget*, Nature
- Minsky (1986) — *Stabilizing an Unstable Economy*
- Acemoglu & Restrepo (2018, 2019) — automation and labor
