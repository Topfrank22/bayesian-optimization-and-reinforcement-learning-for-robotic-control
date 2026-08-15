"""
MAX_BO.py — Bayesian Optimization per la massima elongazione (2 parametri).

Variante di bayes_opt_chirp.py con le seguenti differenze:
  - AMP_FRACTION rimosso dallo spazio di ricerca: fisicamente non può superare 1.0,
    quindi viene fissato a 1.0 costante. Lo spazio diventa 2D (F0_CHIRP, CHIRP_RATE).
  - CHIRP_RATE_RANGE ristretto a (0.005, 0.15): il valore 0 è degenere (sinusoide fissa),
    e valori > 0.15 fanno uno sweep troppo veloce per eccitare la risonanza con zeta_r=0.30.
  - N_ITER aumentato a 300 e PATIENCE abbassato a 40: più budget utile, early stopping
    più reattivo.
  - LOSS_FN fisso a "max_elongation": questo file ha un unico obiettivo.
  - Tutto il resto è identico a bayes_opt_chirp.py: stessa struttura cartelle, stesso
    formato CSV, stesso checkpoint/resume, stessi plot.

Features ereditate da bayes_opt_chirp.py:
- one output folder per run, named run_YYYYMMDD_HHMMSS/
- explicit resume config at the top (RESUME_FROM_CHECKPOINT, CHECKPOINT_PATH_RESUME)
- checkpoint contains full run configuration snapshot
- hard stop if search space changed when resuming
- warning + user confirmation if other config changed on resume
- LHS warm start
- configurable GP kernel and acquisition function via scikit-optimize
- CSV logging of all evaluations
- checkpoint / resume with pickle
- run_config.json saved in each run folder
- verbose logging every N iterations
- convergence plot
- GP heatmap 2D (F0_CHIRP vs CHIRP_RATE)
- 2D scatter of explored points colored by objective value
- best_params.json with top-5 configurations saved at end of run
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import qmc
from skopt import Optimizer
from skopt.space import Real

from msd_simulator_core import run_simulation, DEFAULT_PARAMS


# ═══════════════════════════════════════════════════════════════
# SEZIONE 0 — RUN / RESUME CONFIG (editabile)
# ═══════════════════════════════════════════════════════════════
RESUME_FROM_CHECKPOINT = False
CHECKPOINT_PATH_RESUME = ""   # es. "ROBOT/bo_outputs_max/run_20260513_142000/bo_checkpoint.pkl"


# ═══════════════════════════════════════════════════════════════
# SEZIONE 1 — SEARCH SPACE (editabile)
# ═══════════════════════════════════════════════════════════════
# AMP_FRACTION è fissa a 1.0 — limite fisico dell'attuatore.
# Lo spazio è 2D: solo F0_CHIRP e CHIRP_RATE.

F0_RANGE         = (0.01, 25.0)
CHIRP_RATE_RANGE = (-5.0, 5.0)   # 0 escluso (degenere), max 0.15 (sweep fisicamente utile)

AMP_FRACTION_FIXED = 1.0           # COSTANTE — non ottimizzabile

SPACE = [
    Real(*F0_RANGE,         name="F0_CHIRP"),
    Real(*CHIRP_RATE_RANGE, name="CHIRP_RATE"),
]


# ═══════════════════════════════════════════════════════════════
# SEZIONE 2 — OBJECTIVE CONFIG
# ═══════════════════════════════════════════════════════════════
# Questo file ottimizza SOLO la massima elongazione.
# LOSS_FN è fissa a "max_elongation" e non è modificabile qui.
LOSS_FN = "max_elongation"

T_MAX = DEFAULT_PARAMS["T_MAX"]


# ═══════════════════════════════════════════════════════════════
# SEZIONE 3 — GP CONFIG
# ═══════════════════════════════════════════════════════════════
KERNEL_NAME = "Matern"
ACQ_FUNC    = "EI"
XI          = 0.01
KAPPA       = 2.576
NORMALIZE_Y = True
NOISE       = 1e-10


# ═══════════════════════════════════════════════════════════════
# SEZIONE 4 — LOOP CONFIG
# ═══════════════════════════════════════════════════════════════
N_INIT          = 75     # punti iniziali latin hypercube (meno ne servono in 2D)
N_ITER          = 1000   # iterazioni totali (più budget rispetto a bayes_opt_chirp.py)
PATIENCE        = 100    # early stopping più reattivo
MIN_DELTA       = 0.005
PATIENCE_DELTA  = 100
CHECKPOINT_EVERY = 20
TOP_N           = 5

VERBOSE       = True
VERBOSE_EVERY = 10

BASE_OUTPUT_DIR = "ROBOT/bo_outputs_max"   # cartella separata per non mescolare le run
RUN_PREFIX      = "run"

# Variabili popolate a runtime da initialize_run()
OUTPUT_DIR            = None
CHECKPOINT_PATH       = None
CSV_PATH              = None
CONV_PATH             = None
HEATMAP_PATH          = None
SCATTER2D_PATH        = None
RUN_CONFIG_JSON_PATH  = None
BEST_PARAMS_PATH      = None


# ═══════════════════════════════════════════════════════════════
# SEZIONE 5 — UTILITIES
# ═══════════════════════════════════════════════════════════════
def build_optimizer():
    return Optimizer(
        dimensions=SPACE,
        base_estimator="GP",
        acq_func=ACQ_FUNC,
        acq_func_kwargs={"xi": XI, "kappa": KAPPA},
        random_state=42,
    )


def current_config_snapshot():
    return {
        "search_space": {
            "F0_RANGE":           list(F0_RANGE),
            "CHIRP_RATE_RANGE":   list(CHIRP_RATE_RANGE),
            "AMP_FRACTION_FIXED": AMP_FRACTION_FIXED,
        },
        "objective": {
            "LOSS_FN": LOSS_FN,
            "T_MAX":   T_MAX,
        },
        "gp": {
            "KERNEL_NAME": KERNEL_NAME,
            "ACQ_FUNC":    ACQ_FUNC,
            "XI":          XI,
            "KAPPA":       KAPPA,
            "NORMALIZE_Y": NORMALIZE_Y,
            "NOISE":       NOISE,
        },
        "loop": {
            "N_INIT":           N_INIT,
            "N_ITER":           N_ITER,
            "PATIENCE":         PATIENCE,
            "MIN_DELTA":        MIN_DELTA,
            "PATIENCE_DELTA":   PATIENCE_DELTA,
            "CHECKPOINT_EVERY": CHECKPOINT_EVERY,
        },
    }


def create_new_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(BASE_OUTPUT_DIR) / f"{RUN_PREFIX}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def setup_output_paths(output_dir):
    global OUTPUT_DIR, CHECKPOINT_PATH, CSV_PATH, CONV_PATH, HEATMAP_PATH, SCATTER2D_PATH, RUN_CONFIG_JSON_PATH, BEST_PARAMS_PATH
    OUTPUT_DIR           = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_PATH      = os.path.join(OUTPUT_DIR, "bo_checkpoint.pkl")
    CSV_PATH             = os.path.join(OUTPUT_DIR, "bo_evaluations.csv")
    CONV_PATH            = os.path.join(OUTPUT_DIR, "bo_convergence.png")
    HEATMAP_PATH         = os.path.join(OUTPUT_DIR, "bo_gp_heatmap.png")
    SCATTER2D_PATH       = os.path.join(OUTPUT_DIR, "bo_scatter2d.png")
    RUN_CONFIG_JSON_PATH = os.path.join(OUTPUT_DIR, "run_config.json")
    BEST_PARAMS_PATH     = os.path.join(OUTPUT_DIR, "best_params.json")


def save_run_config_json(config_snapshot):
    payload = {
        "saved_at":        datetime.now().isoformat(timespec="seconds"),
        "output_dir":      OUTPUT_DIR,
        "checkpoint_path": CHECKPOINT_PATH,
        "config":          config_snapshot,
    }
    with open(RUN_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_best_params(df, top_n=5):
    top_df = df.nsmallest(top_n, "loss").reset_index(drop=True)
    entries = []
    for rank, row in top_df.iterrows():
        entries.append({
            "rank":         int(rank + 1),
            "iter":         int(row["iter"]),
            "phase":        row["phase"],
            "F0_CHIRP":     float(row["F0_CHIRP"]),
            "CHIRP_RATE":   float(row["CHIRP_RATE"]),
            "AMP_FRACTION": AMP_FRACTION_FIXED,
            "loss":         float(row["loss"]),
            "xee_max":      float(row["xee_max"]),
            "stop_reason":  row["stop_reason"],
        })
    payload = {
        "saved_at":           datetime.now().isoformat(timespec="seconds"),
        "top_n":              top_n,
        "total_evaluations":  len(df),
        "loss_fn":            LOSS_FN,
        "AMP_FRACTION_fixed": AMP_FRACTION_FIXED,
        "configurations":     entries,
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _flat_diff(old, new, prefix=""):
    diffs = []
    keys = sorted(set(old.keys()) | set(new.keys()))
    for key in keys:
        full_key = f"{prefix}.{key}" if prefix else key
        in_old, in_new = key in old, key in new
        if not in_old:
            diffs.append((full_key, None, new[key]))
        elif not in_new:
            diffs.append((full_key, old[key], None))
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            diffs.extend(_flat_diff(old[key], new[key], full_key))
        elif old[key] != new[key]:
            diffs.append((full_key, old[key], new[key]))
    return diffs


def validate_resume_config(saved_config, current_config):
    saved_space   = saved_config.get("search_space", {})
    current_space = current_config.get("search_space", {})

    if saved_space != current_space:
        print("\n" + "=" * 60)
        print("[ERRORE] Resume bloccato: lo spazio di esplorazione è cambiato.")
        print("=" * 60)
        for key, old, new in _flat_diff(saved_space, current_space, prefix="search_space"):
            print(f"  {key}:  checkpoint={old}  |  current={new}")
        print("=" * 60)
        raise RuntimeError(
            "Search space changed between checkpoint and current config. "
            "Cannot resume this run. Use RESUME_FROM_CHECKPOINT=False to start a new run."
        )

    other_saved   = {k: v for k, v in saved_config.items() if k != "search_space"}
    other_current = {k: v for k, v in current_config.items() if k != "search_space"}
    other_diffs   = _flat_diff(other_saved, other_current)

    if other_diffs:
        print("\n" + "=" * 60)
        print("[WARNING] Il checkpoint è compatibile (search space invariato),")
        print("          ma alcuni altri parametri sono cambiati:")
        print("=" * 60)
        for key, old, new in other_diffs:
            print(f"  {key}:  checkpoint={old}  |  current={new}")
        print("=" * 60)
        answer = input("Vuoi continuare comunque da questo checkpoint? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            raise RuntimeError("Resume annullato dall'utente.")
        print("[INFO] Continuazione confermata.")


def objective(params, sim_params=None):
    """Loss function fissa: max_elongation. AMP_FRACTION sempre 1.0."""
    F0, rate = params
    res  = run_simulation(F0, rate, AMP_FRACTION_FIXED, sim_params=sim_params)
    loss = -res["xee_max"]   # minimizziamo -xee_max = massimizziamo xee_max
    return loss, res


def lhs_points(n_samples):
    sampler  = qmc.LatinHypercube(d=2, seed=42)
    sample   = sampler.random(n=n_samples)
    l_bounds = [F0_RANGE[0], CHIRP_RATE_RANGE[0]]
    u_bounds = [F0_RANGE[1], CHIRP_RATE_RANGE[1]]
    return qmc.scale(sample, l_bounds, u_bounds).tolist()


def save_checkpoint(payload):
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint_from_path(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato: '{checkpoint_path}'. "
            "Controlla il valore di CHECKPOINT_PATH_RESUME."
        )
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def save_csv(df):
    df.to_csv(CSV_PATH, index=False)


def stopping_by_convergence(best_history):
    if len(best_history) >= PATIENCE + 1:
        recent = best_history[-(PATIENCE + 1):]
        no_improve_count = 0
        current_best = recent[0]
        for val in recent[1:]:
            if val < current_best:
                current_best = val
                no_improve_count = 0
            else:
                no_improve_count += 1
        if no_improve_count >= PATIENCE:
            return True, f"no_improvement_{PATIENCE}"

    if len(best_history) >= PATIENCE_DELTA + 1:
        improvement = best_history[-(PATIENCE_DELTA + 1)] - best_history[-1]
        if improvement < MIN_DELTA:
            return True, f"small_improvement_{improvement:.6f}"

    return False, None


def vprint(msg):
    if VERBOSE:
        print(msg)


# ═══════════════════════════════════════════════════════════════
# SEZIONE 6 — PLOT
# ═══════════════════════════════════════════════════════════════
def plot_convergence(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["iter"], df["loss"],            marker="o", lw=1.0, label="loss (−xee_max)")
    ax.plot(df["iter"], df["best_loss_so_far"], lw=2.0,            label="best so far")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective  (−xee_max)")
    ax.set_title("MAX_BO convergence  ↓ = maggiore elongazione")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CONV_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gp_heatmap(optimizer, df):
    """Heatmap 2D del modello GP: F0_CHIRP vs CHIRP_RATE."""
    if len(optimizer.yi) < 3:
        return

    model = optimizer.models[-1]
    f0_vals   = np.linspace(F0_RANGE[0],        F0_RANGE[1],        80)
    rate_vals = np.linspace(CHIRP_RATE_RANGE[0], CHIRP_RATE_RANGE[1], 80)
    F0_grid, RATE_grid = np.meshgrid(f0_vals, rate_vals)

    X  = np.column_stack([F0_grid.ravel(), RATE_grid.ravel()])
    mu = model.predict(X).reshape(F0_grid.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.contourf(F0_grid, RATE_grid, mu, levels=40, cmap="viridis")
    ax.scatter(df["F0_CHIRP"], df["CHIRP_RATE"], c=df["loss"], cmap="coolwarm",
               s=30, edgecolor="k", linewidths=0.5, label="valutazioni")

    best_row = df.iloc[df["loss"].idxmin()]
    ax.scatter(best_row["F0_CHIRP"], best_row["CHIRP_RATE"],
               c="yellow", edgecolor="black", s=200, marker="*", zorder=5, label="best")

    ax.set_xlabel("F0_CHIRP (Hz)")
    ax.set_ylabel("CHIRP_RATE")
    ax.set_title("GP mean — F0_CHIRP vs CHIRP_RATE  (AMP_FRACTION=1.0 fisso)")
    fig.colorbar(im, ax=ax, label="−xee_max")
    ax.legend()
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter2d(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["F0_CHIRP"], df["CHIRP_RATE"],
                    c=df["loss"], cmap=cm.viridis, s=60, edgecolor="k", linewidths=0.4)
    best_row = df.iloc[df["loss"].idxmin()]
    ax.scatter(best_row["F0_CHIRP"], best_row["CHIRP_RATE"],
               c="red", s=200, marker="*", zorder=5, label=f"best  xee={best_row['xee_max']:.4f} m")
    ax.set_xlabel("F0_CHIRP (Hz)")
    ax.set_ylabel("CHIRP_RATE")
    ax.set_title("Spazio esplorato dalla BO  (AMP_FRACTION=1.0 fisso)")
    fig.colorbar(sc, ax=ax, label="loss (−xee_max)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SCATTER2D_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# SEZIONE 7 — INIT RUN
# ═══════════════════════════════════════════════════════════════
def initialize_run():
    config_snapshot = current_config_snapshot()

    if RESUME_FROM_CHECKPOINT:
        checkpoint = load_checkpoint_from_path(CHECKPOINT_PATH_RESUME)
        resume_dir = checkpoint.get("output_dir") or str(
            Path(CHECKPOINT_PATH_RESUME).resolve().parent
        )
        setup_output_paths(resume_dir)
        saved_config = checkpoint.get("config_snapshot", {})
        validate_resume_config(saved_config, config_snapshot)
        save_run_config_json(config_snapshot)
        print(f"[INFO] Resume dalla cartella: {OUTPUT_DIR}")
        print(f"[INFO] Iterazione di partenza: {checkpoint.get('iter_start', '?')}/{N_ITER}")
        return checkpoint, config_snapshot

    output_dir = create_new_run_dir()
    setup_output_paths(output_dir)
    save_run_config_json(config_snapshot)
    print(f"[INFO] Nuova run creata in: {OUTPUT_DIR}")
    print(f"[INFO] Iterazioni totali: {N_ITER}  |  LHS warm-start: {N_INIT}  |  Checkpoint ogni: {CHECKPOINT_EVERY}")
    print(f"[INFO] Spazio 2D: F0_CHIRP {F0_RANGE}  CHIRP_RATE {CHIRP_RATE_RANGE}  AMP_FRACTION={AMP_FRACTION_FIXED} (fisso)")
    return None, config_snapshot


# ═══════════════════════════════════════════════════════════════
# SEZIONE 8 — MAIN LOOP
# ═══════════════════════════════════════════════════════════════
def main(sim_params=None):
    checkpoint, config_snapshot = initialize_run()

    if checkpoint is None:
        optimizer      = build_optimizer()
        records        = []
        best_history   = []
        lhs_init       = lhs_points(N_INIT)
        iter_start     = 0
        phase          = "lhs"
        run_started_at = datetime.now().isoformat(timespec="seconds")
    else:
        optimizer      = checkpoint["optimizer"]
        records        = checkpoint["records"]
        best_history   = checkpoint["best_history"]
        lhs_init       = checkpoint["lhs_init"]
        iter_start     = checkpoint["iter_start"]
        phase          = checkpoint["phase"]
        run_started_at = checkpoint.get("run_started_at", datetime.now().isoformat(timespec="seconds"))

    stop_reason = "max_iterations"

    for i in range(iter_start, N_ITER):
        if phase == "lhs" and i < len(lhs_init):
            x = lhs_init[i]
            next_phase = "bo" if i == len(lhs_init) - 1 else "lhs"
        else:
            phase = "bo"
            x = optimizer.ask()
            next_phase = "bo"

        loss, res = objective(x, sim_params=sim_params)
        optimizer.tell(x, loss)

        best_loss = loss if len(best_history) == 0 else min(best_history[-1], loss)
        best_history.append(best_loss)

        records.append({
            "iter":             i,
            "phase":            phase,
            "F0_CHIRP":         x[0],
            "CHIRP_RATE":       x[1],
            "AMP_FRACTION":     AMP_FRACTION_FIXED,
            "xee_max":          res["xee_max"],
            "loss":             loss,
            "best_loss_so_far": best_loss,
            "stop_reason":      res["stop_reason"],
        })

        df = pd.DataFrame(records)
        save_csv(df)

        checkpoint_saved = False
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint({
                "optimizer":       optimizer,
                "records":         records,
                "best_history":    best_history,
                "lhs_init":        lhs_init,
                "iter_start":      i + 1,
                "phase":           next_phase,
                "output_dir":      OUTPUT_DIR,
                "run_started_at":  run_started_at,
                "last_saved_at":   datetime.now().isoformat(timespec="seconds"),
                "config_snapshot": config_snapshot,
            })
            checkpoint_saved = True

        if VERBOSE and ((i + 1) % VERBOSE_EVERY == 0 or i == iter_start):
            ckpt_tag = "  [checkpoint saved]" if checkpoint_saved else ""
            print(
                f"[iter {i+1:>4}/{N_ITER} | {phase:>3}] "
                f"loss={loss:.4f}  best={best_loss:.4f}  "
                f"F0={x[0]:.3f}  rate={x[1]:.4f}  "
                f"xee_max={res['xee_max']:.4f} m"
                f"{ckpt_tag}"
            )

        should_stop, conv_reason = stopping_by_convergence(best_history)
        if should_stop:
            stop_reason = conv_reason
            vprint(f"\n[INFO] Early stopping at iter {i+1}: {stop_reason}")
            break

        phase = next_phase

    df = pd.DataFrame(records)
    save_csv(df)
    plot_convergence(df)
    plot_gp_heatmap(optimizer, df)
    plot_scatter2d(df)
    save_best_params(df, top_n=TOP_N)

    save_checkpoint({
        "optimizer":         optimizer,
        "records":           records,
        "best_history":      best_history,
        "lhs_init":          lhs_init,
        "iter_start":        len(records),
        "phase":             phase,
        "final_stop_reason": stop_reason,
        "output_dir":        OUTPUT_DIR,
        "run_started_at":    run_started_at,
        "last_saved_at":     datetime.now().isoformat(timespec="seconds"),
        "config_snapshot":   config_snapshot,
    })

    best_row = df.iloc[df["loss"].idxmin()]
    print(f"\nMAX_BO completed. Stop reason: {stop_reason}")
    print(f"Best xee_max:      {best_row['xee_max']:.4f} m  (loss={best_row['loss']:.4f})")
    print(f"Best F0_CHIRP:     {best_row['F0_CHIRP']:.4f} Hz")
    print(f"Best CHIRP_RATE:   {best_row['CHIRP_RATE']:.6f}")
    print(f"AMP_FRACTION:      {AMP_FRACTION_FIXED} (fisso)")
    print(f"Output directory:  {OUTPUT_DIR}")
    print(f"Checkpoint:        {CHECKPOINT_PATH}")
    print(f"CSV:               {CSV_PATH}")
    print(f"Convergence plot:  {CONV_PATH}")
    print(f"GP heatmap:        {HEATMAP_PATH}")
    print(f"2D scatter:        {SCATTER2D_PATH}")
    print(f"Run config JSON:   {RUN_CONFIG_JSON_PATH}")
    print(f"Best params JSON:  {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
