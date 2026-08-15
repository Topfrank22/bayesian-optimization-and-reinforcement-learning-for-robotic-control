"""
Bayesian Optimization for chirp tuning.


Features:
- one output folder per run, named run_YYYYMMDD_HHMMSS/
- explicit resume config at the top (RESUME_FROM_CHECKPOINT, CHECKPOINT_PATH_RESUME)
- checkpoint contains full run configuration snapshot
- hard stop if search space changed when resuming
- warning + user confirmation if other config changed on resume
- LHS warm start (instead of naive random init)
- configurable GP kernel and acquisition function via scikit-optimize
- CSV logging of all evaluations
- checkpoint / resume with pickle
- run_config.json saved in each run folder
- verbose logging every N iterations (configurable)
- convergence plot
- 3 GP pairwise heatmaps around the explored space
- 3D scatter of explored points colored by objective value
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


from scipy.stats import qmc
from skopt import Optimizer
from skopt.space import Real


from msd_simulator_core import run_simulation, DEFAULT_PARAMS



# ═══════════════════════════════════════════════════════════════
# SEZIONE 0 — RUN / RESUME CONFIG (editabile)
# ═══════════════════════════════════════════════════════════════
RESUME_FROM_CHECKPOINT = False
CHECKPOINT_PATH_RESUME = ""   # es. "ROBOT/bo_outputs/run_20260513_142000/bo_checkpoint.pkl"



# ═══════════════════════════════════════════════════════════════
# SEZIONE 1 — SEARCH SPACE (editabile)
# ═══════════════════════════════════════════════════════════════
F0_RANGE = (0.05, 6.0)
CHIRP_RATE_RANGE = (-10.00, 10.00)
AMP_FRACTION_RANGE = (0.10, 1.0)


SPACE = [
    Real(*F0_RANGE, name="F0_CHIRP"),
    Real(*CHIRP_RATE_RANGE, name="CHIRP_RATE"),
    Real(*AMP_FRACTION_RANGE, name="AMP_FRACTION"),
]



# ═══════════════════════════════════════════════════════════════
# SEZIONE 2 — OBJECTIVE CONFIG
# ═══════════════════════════════════════════════════════════════
#
# Scegli la loss function impostando LOSS_FN su uno dei valori seguenti:
#
#   "accuracy_time"   (default)
#       Minimizza sia l'errore di accuratezza sia il tempo per raggiungere il target.
#       L = W1 * |accuracy| / SIGMA_A + W2 * t_eff / SIGMA_T + PENALTY_NOTGT
#       Usa questo quando vuoi trovare il chirp che arriva al target il più velocemente
#       e con la minima overshooting.
#
#   "accuracy_only"
#       Minimizza solo l'errore di accuratezza (|xee_max - X_TARGET|).
#       L = |accuracy| / SIGMA_A + PENALTY_NOTGT
#       Usa questo quando ti interessa solo raggiungere il target, indipendentemente
#       da quanto tempo ci vuole.
#
#   "time_only"
#       Minimizza solo il tempo per raggiungere il target.
#       L = t_eff / SIGMA_T + PENALTY_NOTGT
#       Usa questo quando vuoi il chirp più veloce possibile, senza penalizzare l'overshoot.
#
#   "max_elongation"
#       Massimizza l'elongazione massima della punta del robot (xee_max).
#       L = -xee_max
#       La BO minimizza -xee_max, equivalente a massimizzare xee_max.
#       Nessun altro termine: l'unico obiettivo è spingere la punta il più lontano possibile.
#       NOTA: in questo modo il plot di convergenza mostra valori negativi —
#             più il valore scende, più l'elongazione cresce.
#
LOSS_FN = "accuracy_only"   # <-- cambia qui: "accuracy_time" | "accuracy_only" | "time_only" | "max_elongation"


# Target di elongazione (m) che la simulazione deve raggiungere.
# Se questa variabile non esiste ancora nel tuo progetto, è definita qui come
# nuovo parametro editabile: viene passata a run_simulation tramite sim_params,
# sovrascrivendo un eventuale valore presente in DEFAULT_PARAMS.
X_TARGET = 1.4   # <-- modifica qui il target desiderato (m)


# Parametri condivisi tra le loss functions
W1 = 1.0          # peso accuratezza  (usato da: accuracy_time)
W2 = 0.25         # peso tempo        (usato da: accuracy_time)
SIGMA_A = 0.25    # scala accuratezza (usato da: accuracy_time, accuracy_only)
SIGMA_T = 30.0    # scala tempo       (usato da: accuracy_time, time_only)
PENALTY_NOTGT = 5.0  # penality se il target non viene raggiunto (tutte le loss tranne max_elongation)
T_MAX = DEFAULT_PARAMS["T_MAX"]


_LOSS_FN_VALID = {"accuracy_time", "accuracy_only", "time_only", "max_elongation"}
if LOSS_FN not in _LOSS_FN_VALID:
    raise ValueError(f"LOSS_FN='{LOSS_FN}' non valido. Scegli tra: {_LOSS_FN_VALID}")



# ═══════════════════════════════════════════════════════════════
# SEZIONE 3 — GP CONFIG
# ═══════════════════════════════════════════════════════════════
# Usa il GP interno di skopt ("GP") per compatibilità con sklearn recente.
# Evita il bug: GaussianProcessRegressor.predict() unexpected keyword 'return_mean_grad'.
# KERNEL_NAME è usato solo come tag nel config_snapshot.
KERNEL_NAME = "Matern"      # informativo — skopt usa Matern 5/2 internamente
ACQ_FUNC = "EI"             # "EI", "PI", "LCB"
XI = 0.01
KAPPA = 2.576
NORMALIZE_Y = True
NOISE = 1e-10



# ═══════════════════════════════════════════════════════════════
# SEZIONE 4 — LOOP CONFIG
# ═══════════════════════════════════════════════════════════════
N_INIT = 30 #punti iniziali  latin square
N_ITER = 230 #iterazioni totali (inclusi i punti iniziali)
PATIENCE = 100
MIN_DELTA = 0.005
PATIENCE_DELTA = 100
CHECKPOINT_EVERY = 10
TOP_N = 5  # numero di configurazioni top da salvare in best_params.json


# Verbose: stampa avanzamento ogni VERBOSE_EVERY iterazioni.
# Metti VERBOSE = False per silenziare tutto (tranne info di avvio e fine run).
VERBOSE = True
VERBOSE_EVERY = 10


BASE_OUTPUT_DIR = "ROBOT/bo_outputs"
RUN_PREFIX = "run"


# Queste variabili vengono popolate a runtime da initialize_run()
OUTPUT_DIR = None
CHECKPOINT_PATH = None
CSV_PATH = None
CONV_PATH = None
HEATMAP_PATH = None
SCATTER3D_PATH = None
RUN_CONFIG_JSON_PATH = None
BEST_PARAMS_PATH = None



# ═══════════════════════════════════════════════════════════════
# SEZIONE 5 — UTILITIES
# ═══════════════════════════════════════════════════════════════
def build_optimizer():
    """
    Usa il GP interno di skopt (base_estimator="GP") per evitare incompatibilità
    con versioni recenti di scikit-learn che non espongono return_mean_grad.
    """
    return Optimizer(
        dimensions=SPACE,
        base_estimator="GP",
        acq_func=ACQ_FUNC,
        acq_func_kwargs={"xi": XI, "kappa": KAPPA},
        random_state=42,
    )



def current_config_snapshot():
    """Snapshot completo di tutti i parametri della run corrente."""
    return {
        "search_space": {
            "F0_RANGE": list(F0_RANGE),
            "CHIRP_RATE_RANGE": list(CHIRP_RATE_RANGE),
            "AMP_FRACTION_RANGE": list(AMP_FRACTION_RANGE),
        },
        "objective": {
            "LOSS_FN": LOSS_FN,
            "X_TARGET": X_TARGET,
            "W1": W1,
            "W2": W2,
            "SIGMA_A": SIGMA_A,
            "SIGMA_T": SIGMA_T,
            "PENALTY_NOTGT": PENALTY_NOTGT,
            "T_MAX": T_MAX,
        },
        "gp": {
            "KERNEL_NAME": KERNEL_NAME,
            "ACQ_FUNC": ACQ_FUNC,
            "XI": XI,
            "KAPPA": KAPPA,
            "NORMALIZE_Y": NORMALIZE_Y,
            "NOISE": NOISE,
        },
        "loop": {
            "N_INIT": N_INIT,
            "N_ITER": N_ITER,
            "PATIENCE": PATIENCE,
            "MIN_DELTA": MIN_DELTA,
            "PATIENCE_DELTA": PATIENCE_DELTA,
            "CHECKPOINT_EVERY": CHECKPOINT_EVERY,
        },
    }



def create_new_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(BASE_OUTPUT_DIR) / f"{RUN_PREFIX}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)



def setup_output_paths(output_dir):
    global OUTPUT_DIR, CHECKPOINT_PATH, CSV_PATH, CONV_PATH, HEATMAP_PATH, SCATTER3D_PATH, RUN_CONFIG_JSON_PATH, BEST_PARAMS_PATH
    OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "bo_checkpoint.pkl")
    CSV_PATH = os.path.join(OUTPUT_DIR, "bo_evaluations.csv")
    CONV_PATH = os.path.join(OUTPUT_DIR, "bo_convergence.png")
    HEATMAP_PATH = os.path.join(OUTPUT_DIR, "bo_gp_pairwise.png")
    SCATTER3D_PATH = os.path.join(OUTPUT_DIR, "bo_scatter3d.png")
    RUN_CONFIG_JSON_PATH = os.path.join(OUTPUT_DIR, "run_config.json")
    BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params.json")



def save_run_config_json(config_snapshot):
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": OUTPUT_DIR,
        "checkpoint_path": CHECKPOINT_PATH,
        "config": config_snapshot,
    }
    with open(RUN_CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def save_best_params(df, top_n=5):
    """Salva le top_n configurazioni con loss minore in best_params.json."""
    top_df = df.nsmallest(top_n, "loss").reset_index(drop=True)
    entries = []
    for rank, row in top_df.iterrows():
        entries.append({
            "rank": int(rank + 1),
            "iter": int(row["iter"]),
            "phase": row["phase"],
            "F0_CHIRP": float(row["F0_CHIRP"]),
            "CHIRP_RATE": float(row["CHIRP_RATE"]),
            "AMP_FRACTION": float(row["AMP_FRACTION"]),
            "loss": float(row["loss"]),
            "xee_max": float(row["xee_max"]),
            "accuracy": float(row["accuracy"]),
            "time_to_target": float(row["time_to_target"]) if pd.notna(row["time_to_target"]) else None,
            "stop_reason": row["stop_reason"],
        })
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "top_n": top_n,
        "total_evaluations": len(df),
        "loss_fn": LOSS_FN,
        "x_target": X_TARGET,
        "configurations": entries,
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def _flat_diff(old, new, prefix=""):
    """Ritorna lista di (chiave, valore_vecchio, valore_nuovo) per ogni differenza."""
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
    """
    Confronta la configurazione salvata nel checkpoint con quella corrente.
    - Se lo search space è cambiato → blocco hard (RuntimeError).
    - Se altri parametri sono cambiati (incluso X_TARGET) → warning + conferma utente.
    """
    saved_space = saved_config.get("search_space", {})
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


    other_saved = {k: v for k, v in saved_config.items() if k != "search_space"}
    other_current = {k: v for k, v in current_config.items() if k != "search_space"}
    other_diffs = _flat_diff(other_saved, other_current)


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
    F0, rate, amp_frac = params

    # Merge dei parametri di simulazione: DEFAULT_PARAMS < sim_params passati < X_TARGET globale.
    # X_TARGET è il parametro "editabile" della Sezione 2 e ha sempre la precedenza,
    # cosi il target si può modificare in cima al file senza toccare msd_simulator_core.
    merged_sim_params = dict(DEFAULT_PARAMS)
    if sim_params:
        merged_sim_params.update(sim_params)
    merged_sim_params["X_TARGET"] = X_TARGET

    res = run_simulation(F0, rate, amp_frac, sim_params=merged_sim_params)
    acc = res["accuracy"]
    t_tgt = res["time_to_target"]
    t_eff = t_tgt if t_tgt is not None else T_MAX
    penalty = 0.0 if t_tgt is not None else PENALTY_NOTGT


    if LOSS_FN == "accuracy_time":
        # Minimizza sia errore di accuratezza sia tempo al target
        loss = W1 * abs(acc) / SIGMA_A + W2 * t_eff / SIGMA_T + penalty
    elif LOSS_FN == "accuracy_only":
        # Minimizza solo l'errore di accuratezza
        loss = abs(acc) / SIGMA_A + penalty
    elif LOSS_FN == "time_only":
        # Minimizza solo il tempo per raggiungere il target
        loss = t_eff / SIGMA_T + penalty
    elif LOSS_FN == "max_elongation":
        # Massimizza la posizione massima raggiunta dalla punta (xee_max).
        # La BO minimizza per convenzione, quindi usiamo loss = -xee_max.
        # Nessuna penalty: non importa se il target non viene raggiunto.
        loss = -res["xee_max"]
    else:
        raise ValueError(f"LOSS_FN='{LOSS_FN}' non riconosciuta.")


    return loss, res



def lhs_points(n_samples):
    sampler = qmc.LatinHypercube(d=3, seed=42)
    sample = sampler.random(n=n_samples)
    l_bounds = [F0_RANGE[0], CHIRP_RATE_RANGE[0], AMP_FRACTION_RANGE[0]]
    u_bounds = [F0_RANGE[1], CHIRP_RATE_RANGE[1], AMP_FRACTION_RANGE[1]]
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



def stopping_by_convergence(loss_history, best_history):
    """
    loss_history: lista delle loss grezze (una per iterazione)
    best_history: lista del best_loss_so_far (minimo cumulativo non crescente)

    Criteri:
    1. PATIENCE: se la best loss non migliora per PATIENCE iterazioni consecutive → stop
    2. PATIENCE_DELTA + MIN_DELTA: se il miglioramento su PATIENCE_DELTA iterazioni
       è inferiore a MIN_DELTA → stop (usa la loss grezza, più sensibile)
    """
    # Criterio 1: best loss non migliora per PATIENCE iterazioni consecutive
    if len(best_history) >= PATIENCE + 1:
        last_patience = best_history[-PATIENCE:]   # ultime PATIENCE best loss
        if len(set(last_patience)) == 1:            # tutte uguali → nessun miglioramento
            return True, f"no_improvement_{PATIENCE}"

    # Criterio 2: miglioramento troppo piccolo su PATIENCE_DELTA iterazioni
    # Usa best_history (minimo cumulativo, non decrescente) per evitare falsi
    # positivi causati dalla loss grezza che la BO fa oscillare deliberatamente.
    if len(best_history) >= PATIENCE_DELTA + 1:
        improvement = best_history[-(PATIENCE_DELTA + 1)] - best_history[-1]
        if improvement < MIN_DELTA:
            return True, f"small_improvement_{improvement:.6f}"

    return False, None



def vprint(msg):
    """Stampa solo se VERBOSE è True."""
    if VERBOSE:
        print(msg)



# ═══════════════════════════════════════════════════════════════
# SEZIONE 6 — PLOT
# ═══════════════════════════════════════════════════════════════
def plot_convergence(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["iter"], df["loss"], marker="o", lw=1.0, label="loss")
    ax.plot(df["iter"], df["best_loss_so_far"], lw=2.0, label="best so far")
    ax.set_xlabel("Iteration")
    if LOSS_FN == "max_elongation":
        ax.set_ylabel("Objective  (-xee_max)")
        ax.set_title(f"BO convergence  [loss_fn={LOSS_FN}]  ↓ = maggiore elongazione")
    else:
        ax.set_ylabel("Objective")
        ax.set_title(f"BO convergence  [loss_fn={LOSS_FN}, target={X_TARGET}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CONV_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_pairwise_gp(optimizer, df):
    if len(optimizer.yi) < 3:
        return


    best_idx = int(np.argmin(optimizer.yi))
    best_x = optimizer.Xi[best_idx]
    model = optimizer.models[-1]


    pairs = [(0, 1), (0, 2), (1, 2)]
    names = ["F0_CHIRP", "CHIRP_RATE", "AMP_FRACTION"]
    bounds = [F0_RANGE, CHIRP_RATE_RANGE, AMP_FRACTION_RANGE]


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))


    for ax, (i, j) in zip(axes, pairs):
        xi = np.linspace(bounds[i][0], bounds[i][1], 60)
        xj = np.linspace(bounds[j][0], bounds[j][1], 60)
        XI_, XJ_ = np.meshgrid(xi, xj)
        X = np.tile(best_x, (XI_.size, 1))
        X[:, i] = XI_.ravel()
        X[:, j] = XJ_.ravel()
        mu = model.predict(X).reshape(XI_.shape)


        im = ax.contourf(XI_, XJ_, mu, levels=30, cmap="viridis")
        ax.scatter(df[names[i]], df[names[j]], c=df["loss"], cmap="coolwarm", s=25, edgecolor="k")
        ax.scatter(best_x[i], best_x[j], c="yellow", edgecolor="black", s=120, marker="*")
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_title(f"GP mean: {names[i]} vs {names[j]}")
        fig.colorbar(im, ax=ax)


    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_scatter3d(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        df["F0_CHIRP"],
        df["CHIRP_RATE"],
        df["AMP_FRACTION"],
        c=df["loss"],
        cmap=cm.viridis,
        s=60,
    )
    best_row = df.iloc[df["loss"].idxmin()]
    ax.scatter(
        [best_row["F0_CHIRP"]],
        [best_row["CHIRP_RATE"]],
        [best_row["AMP_FRACTION"]],
        c="red",
        s=180,
        marker="*",
    )
    ax.set_xlabel("F0_CHIRP")
    ax.set_ylabel("CHIRP_RATE")
    ax.set_zlabel("AMP_FRACTION")
    ax.set_title(f"Explored BO space  [loss_fn={LOSS_FN}, target={X_TARGET}]")
    fig.colorbar(p, ax=ax, shrink=0.7, label="loss")
    plt.tight_layout()
    plt.savefig(SCATTER3D_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ═══════════════════════════════════════════════════════════════
# SEZIONE 7 — INIT RUN
# ═══════════════════════════════════════════════════════════════
def initialize_run():
    """
    Decide se creare una nuova run o riprendere da checkpoint.
    Ritorna (checkpoint_data | None, config_snapshot).
    """
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
    print(f"[INFO] Loss function: {LOSS_FN}  |  X_TARGET: {X_TARGET}")
    return None, config_snapshot



# ═══════════════════════════════════════════════════════════════
# SEZIONE 8 — MAIN LOOP
# ═══════════════════════════════════════════════════════════════
def main(sim_params=None):
    checkpoint, config_snapshot = initialize_run()


    if checkpoint is None:
        optimizer = build_optimizer()
        records = []
        best_history = []
        lhs_init = lhs_points(N_INIT)
        iter_start = 0
        phase = "lhs"
        run_started_at = datetime.now().isoformat(timespec="seconds")
    else:
        optimizer = checkpoint["optimizer"]
        records = checkpoint["records"]
        best_history = checkpoint["best_history"]
        lhs_init = checkpoint["lhs_init"]
        iter_start = checkpoint["iter_start"]
        phase = checkpoint["phase"]
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
            "iter": i,
            "phase": phase,
            "F0_CHIRP": x[0],
            "CHIRP_RATE": x[1],
            "AMP_FRACTION": x[2],
            "accuracy": res["accuracy"],
            "time_to_target": res["time_to_target"],
            "stop_reason": res["stop_reason"],
            "xee_max": res["xee_max"],
            "loss": loss,
            "best_loss_so_far": best_loss,
        })


        df = pd.DataFrame(records)
        save_csv(df)


        checkpoint_saved = False
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint({
                "optimizer": optimizer,
                "records": records,
                "best_history": best_history,
                "lhs_init": lhs_init,
                "iter_start": i + 1,
                "phase": next_phase,
                "output_dir": OUTPUT_DIR,
                "run_started_at": run_started_at,
                "last_saved_at": datetime.now().isoformat(timespec="seconds"),
                "config_snapshot": config_snapshot,
            })
            checkpoint_saved = True


        # Verbose log ogni VERBOSE_EVERY iterazioni
        if VERBOSE and ((i + 1) % VERBOSE_EVERY == 0 or i == iter_start):
            t_tgt = res["time_to_target"]
            t_str = f"{t_tgt:.1f}s" if t_tgt is not None else "  N/A "
            ckpt_tag = "  [checkpoint saved]" if checkpoint_saved else ""
            xee_tag = f"  xee_max={res['xee_max']:.4f}" if LOSS_FN == "max_elongation" else f"  acc={res['accuracy']:.4f} (target={X_TARGET})"
            print(
                f"[iter {i+1:>4}/{N_ITER} | {phase:>3}] "
                f"loss={loss:.4f}  best={best_loss:.4f}  "
                f"F0={x[0]:.3f}  rate={x[1]:.4f}  amp={x[2]:.2f}  "
                f"t_tgt={t_str}"
                f"{xee_tag}"
                f"{ckpt_tag}"
            )


        loss_history = [r["loss"] for r in records]
        should_stop, conv_reason = stopping_by_convergence(loss_history, best_history)
        if should_stop:
            stop_reason = conv_reason
            vprint(f"\n[INFO] Early stopping at iter {i+1}: {stop_reason}")
            break


        phase = next_phase


    df = pd.DataFrame(records)
    save_csv(df)
    plot_convergence(df)
    plot_pairwise_gp(optimizer, df)
    plot_scatter3d(df)
    save_best_params(df, top_n=TOP_N)


    save_checkpoint({
        "optimizer": optimizer,
        "records": records,
        "best_history": best_history,
        "lhs_init": lhs_init,
        "iter_start": len(records),
        "phase": phase,
        "final_stop_reason": stop_reason,
        "output_dir": OUTPUT_DIR,
        "run_started_at": run_started_at,
        "last_saved_at": datetime.now().isoformat(timespec="seconds"),
        "config_snapshot": config_snapshot,
    })


    best_row = df.iloc[df["loss"].idxmin()]
    print(f"\nBO completed. Stop reason: {stop_reason}")
    print(f"Loss function:     {LOSS_FN}")
    print(f"Target (X_TARGET): {X_TARGET}")
    if LOSS_FN == "max_elongation":
        print(f"Best xee_max:      {best_row['xee_max']:.4f} m  (loss={best_row['loss']:.4f})")
    else:
        print(f"Best xee_max:      {best_row['xee_max']:.4f} m")
        print(f"Best accuracy:     {best_row['accuracy']:.4f}")
        t_best = best_row["time_to_target"]
        print(f"Best time_to_tgt:  {t_best:.2f}s" if pd.notna(t_best) else "Best time_to_tgt:  N/A")
    print(f"Output directory:  {OUTPUT_DIR}")
    print(f"Checkpoint:        {CHECKPOINT_PATH}")
    print(f"CSV:               {CSV_PATH}")
    print(f"Convergence plot:  {CONV_PATH}")
    print(f"GP pairwise plot:  {HEATMAP_PATH}")
    print(f"3D scatter plot:   {SCATTER3D_PATH}")
    print(f"Run config JSON:   {RUN_CONFIG_JSON_PATH}")
    print(f"Best params JSON:  {BEST_PARAMS_PATH}")



if __name__ == "__main__":
    main()