"""
visualize_bo_chirp.py
=====================
Visualizer animato per la Bayesian Optimization (chirp) del sistema MSD.

A differenza del visualizer RL, non ha bisogno di caricare dati salvati:
imposta i parametri ottimali del chirp all'inizio, esegue la simulazione
e mostra l'animazione in tempo reale.

GRAFICA identica a visualize_rl_run.py (tema chiaro, stessi colori).

MODALITÀ DI VISUALIZZAZIONE  →  impostare VISUALIZER_MODE all'inizio dello script:
  "target"         : comportamento originale. La simulazione si ferma
                     T_POST_TARGET secondi dopo aver raggiunto il target.
                     Il campo visivo arriva fino a x_target + 0.50 m.
  "max_elongation" : la simulazione continua finché xee non ha raggiunto
                     il suo massimo assoluto; poi si ferma dopo
                     MAX_ELONG_EXTRA secondi. Il campo visivo arriva fino a
                     max(xee_max * ELONG_VIEW_PAD, ELONG_VIEW_MIN_MAX) + margine.

CONTROLLI INTERATTIVI:
  - Slider "Frame" : scrubbing avanti/indietro su qualsiasi fotogramma
  - [Play/Pause]   : avvia o mette in pausa l'animazione
  - [⏮ Rewind]     : riporta la riproduzione all'inizio
  - [◀ -10]        : salta 10 frame indietro
  - [+10 ▶]        : salta 10 frame avanti

USO:
    python ROBOT/visualize_bo_chirp.py

Oppure passando parametri via CLI:
    python ROBOT/visualize_bo_chirp.py --f0 0.8 --rate 0.05 --amp 0.6
    python ROBOT/visualize_bo_chirp.py --speed 5
    python ROBOT/visualize_bo_chirp.py --gif
    python ROBOT/visualize_bo_chirp.py --mode max_elongation
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp


# ═══════════════════════════════════════════════════════════════
# SEZIONE 0  —  MODALITÀ VISUALIZZATORE  ← CAMBIA QUI
# ═══════════════════════════════════════════════════════════════
#   "target"         →  comportamento originale
#   "max_elongation" →  vista larga, si ferma dopo il picco di xee
VISUALIZER_MODE = "target"

# Parametri della modalità max_elongation
MAX_ELONG_EXTRA    = 3.0   # secondi extra dopo il picco di xee
ELONG_VIEW_PAD     = 1.25  # fattore su xee_max per il campo visivo
ELONG_VIEW_MIN_MAX = 4.0   # campo visivo minimo garantito (m)


# ═══════════════════════════════════════════════════════════════
# SEZIONE 1  —  PARAMETRI OTTIMALI DEL CHIRP  (editabili qui)
# ═══════════════════════════════════════════════════════════════
F0_CHIRP_OPT     = 0.3279280997832674
CHIRP_RATE_OPT   = 2.7590076996640462
AMP_FRACTION_OPT = 0.9933581097255908


# ═══════════════════════════════════════════════════════════════
# SEZIONE 2  —  PARAMETRI FISICI DEL SISTEMA MSD  (editabili)
# ═══════════════════════════════════════════════════════════════
SIM_PARAMS = {
    "Mb": 70.0,
    "Kb": 4000.0,
    "zeta_b": 0.25,
    "Mr": 2.5,
    "Kr": 1500.0,
    "zeta_r": 0.30,
    "xb0": 0.0,
    "X_MAX_ROBOT": 2.0,
    "GAINED_DISTANCE": 0.3,
    "T_MAX": 200.0,
    "V_ZERO_THRESH": 0.01,
    "T_ZERO_VEL": 10.0,
    "T_STAGNANT": 50.0,
    "T_POST_TARGET": 10.0,
    "DT": 0.01,
    "STEP_SIZE": 0.25,
    "RTOL": 1e-5,
    "ATOL": 1e-7,
}


# ═══════════════════════════════════════════════════════════════
# SEZIONE 3  —  GRAFICA (costanti layout, stessi del visualizer RL)
# ═══════════════════════════════════════════════════════════════
BLOCK_W  = 0.25
BLOCK_H  = 0.08
SPRING_W = 0.04
N_COILS  = 6

# Tema chiaro
FIG_BG     = 'white'
AX_BG      = '#f5f7fa'
SPINE_CLR  = '#cccccc'
TEXT_CLR   = '#222222'
TEXT_MUTED = '#555555'
GRID_CLR   = '#dddddd'

COL_KB     = '#0a8a7e'
COL_KR     = '#c97d00'
COL_BASE   = '#2c6fcd'
COL_ROBOT  = '#c0195e'
COL_TARGET = '#e03030'
COL_CTRL   = '#b36000'
COL_MAXE   = '#7700cc'   # colore indicatori max elongation

GHOST_STEPS     = 4
GHOST_ALPHA_MAX = 0.30


# ═══════════════════════════════════════════════════════════════
# SEZIONE 4  —  SIMULATORE CORE  (ricalca msd_simulator_core.py)
# ═══════════════════════════════════════════════════════════════
def _build_params(overrides=None):
    p = SIM_PARAMS.copy()
    if overrides:
        p.update(overrides)
    p["Db"] = 2 * np.sqrt(p["Kb"] * p["Mb"]) * p["zeta_b"]
    p["Dr"] = 2 * np.sqrt(p["Kr"] * p["Mr"]) * p["zeta_r"]
    p["X_TARGET"] = p["X_MAX_ROBOT"] / 2 + p["GAINED_DISTANCE"]
    return p


def _chirp_value(t, f0, chirp_rate, amp_chirp):
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    return 0.5 * amp_chirp * np.sin(phase)


def _msd_rhs(t, z, xr0_val, p):
    xb, xb_dot, xr, xr_dot = z
    M_mat = np.array([
        [p["Mr"] + p["Mb"], p["Mr"]],
        [p["Mr"],           p["Mr"]],
    ])
    rhs = np.array([
        -p["Db"] * xb_dot - p["Kb"] * (xb - p["xb0"]),
        -p["Dr"] * xr_dot - p["Kr"] * (xr - xr0_val),
    ])
    xb_ddot, xr_ddot = np.linalg.solve(M_mat, rhs)
    return [xb_dot, xb_ddot, xr_dot, xr_ddot]


def run_simulation(f0, chirp_rate, amp_fraction, p=None, mode="target"):
    """
    Esegue la simulazione MSD con chirp.

    mode="target"         → si ferma T_POST_TARGET s dopo il target (comportamento originale).
    mode="max_elongation" → si ferma MAX_ELONG_EXTRA s dopo il picco assoluto di xee.
    """
    if p is None:
        p = _build_params()

    amp_chirp        = amp_fraction * p["X_MAX_ROBOT"]
    z_current        = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    t_current        = 0.0
    target_reached   = False
    t_target         = None
    xr0_hold         = None
    xee_max_ever     = -np.inf
    t_xee_max        = 0.0       # istante del picco massimo di xee
    t_after_peak     = None      # primo istante in cui xee scende dopo il picco
    t_stagnant_start = 0.0
    t_vel_zero_start = None
    stop_reason      = "T_MAX reached"

    t_vals, xb_vals, xr_vals, xee_vals, xee_dot_vals, u_vals = [], [], [], [], [], []
    t_vals.append(0.0)
    xb_vals.append(0.0)
    xr_vals.append(0.0)
    xee_vals.append(0.0)
    xee_dot_vals.append(0.0)
    u_vals.append(_chirp_value(0.0, f0, chirp_rate, amp_chirp))

    while t_current < p["T_MAX"]:
        t_next = min(t_current + p["STEP_SIZE"], p["T_MAX"])
        t_eval = np.arange(t_current, t_next + p["DT"], p["DT"])
        t_eval = t_eval[t_eval <= t_next]
        if t_eval.size == 0 or t_eval[-1] < t_next:
            t_eval = np.append(t_eval, t_next)

        def _rhs_closure(t, z, _tr=target_reached, _hold=xr0_hold):
            xr0 = _hold if _tr else _chirp_value(t, f0, chirp_rate, amp_chirp)
            return _msd_rhs(t, z, xr0, p)

        sol = solve_ivp(
            fun=_rhs_closure,
            t_span=(t_current, t_next),
            y0=z_current,
            t_eval=t_eval,
            method="RK45",
            rtol=p["RTOL"],
            atol=p["ATOL"],
        )

        inner_break = False
        for k in range(1, len(sol.t)):
            ti      = sol.t[k]
            xb      = sol.y[0, k]
            xb_dot  = sol.y[1, k]
            xr      = sol.y[2, k]
            xr_dot  = sol.y[3, k]
            xee     = xb + xr
            xee_dot = xb_dot + xr_dot
            u_cur = xr0_hold if target_reached else _chirp_value(ti, f0, chirp_rate, amp_chirp)

            t_vals.append(ti)
            xb_vals.append(xb)
            xr_vals.append(xr)
            xee_vals.append(xee)
            xee_dot_vals.append(xee_dot)
            u_vals.append(u_cur)

            # Traccia il massimo assoluto di xee (usato da entrambe le modalità)
            if xee > xee_max_ever:
                xee_max_ever = xee
                t_xee_max    = ti
                t_after_peak = None   # reset: nuovo picco trovato
            elif t_after_peak is None and xee < xee_max_ever:
                t_after_peak = ti     # xee ha iniziato a scendere dal picco

            # ── Logica di stop: modalità TARGET ──────────────────────
            if mode == "target":
                if (not target_reached) and (xee >= p["X_TARGET"]):
                    target_reached = True
                    t_target       = ti
                    xr0_hold       = _chirp_value(ti, f0, chirp_rate, amp_chirp)

                if target_reached:
                    if ti >= t_target + p["T_POST_TARGET"]:
                        stop_reason = f"post_target_{p['T_POST_TARGET']}s"
                        z_current   = sol.y[:, k]
                        t_current   = ti
                        inner_break = True
                        break
                else:
                    if abs(xee_dot) < p["V_ZERO_THRESH"]:
                        if t_vel_zero_start is None:
                            t_vel_zero_start = ti
                        elif ti - t_vel_zero_start >= p["T_ZERO_VEL"]:
                            stop_reason = f"near_zero_velocity_{p['T_ZERO_VEL']}s"
                            z_current   = sol.y[:, k]
                            t_current   = ti
                            inner_break = True
                            break
                    else:
                        t_vel_zero_start = None

                    if xee > xee_max_ever - 1e-9:  # aggiornato sopra
                        t_stagnant_start = ti
                    elif ti - t_stagnant_start >= p["T_STAGNANT"]:
                        stop_reason = f"stagnation_{p['T_STAGNANT']}s"
                        z_current   = sol.y[:, k]
                        t_current   = ti
                        inner_break = True
                        break

            # ── Logica di stop: modalità MAX_ELONGATION ───────────────
            elif mode == "max_elongation":
                # Aspettiamo che il picco sia stato superato (xee già scesa)
                # poi aspettiamo MAX_ELONG_EXTRA secondi
                if t_after_peak is not None and ti >= t_after_peak + MAX_ELONG_EXTRA:
                    stop_reason = f"max_elongation_extra_{MAX_ELONG_EXTRA}s"
                    z_current   = sol.y[:, k]
                    t_current   = ti
                    inner_break = True
                    break

        else:
            z_current = sol.y[:, -1]
            t_current = sol.t[-1]
            continue

        if inner_break:
            break

    return {
        "t":              np.array(t_vals,       dtype=np.float32),
        "xb":             np.array(xb_vals,      dtype=np.float32),
        "xr":             np.array(xr_vals,      dtype=np.float32),
        "xee":            np.array(xee_vals,     dtype=np.float32),
        "xee_dot":        np.array(xee_dot_vals, dtype=np.float32),
        "u":              np.array(u_vals,        dtype=np.float32),
        "target_reached": target_reached,
        "t_target":       t_target,
        "stop_reason":    stop_reason,
        "x_target":       p["X_TARGET"],
        "xee_max":        float(xee_max_ever) if xee_max_ever != -np.inf else 0.0,
        "t_xee_max":      float(t_xee_max),
    }


# ═══════════════════════════════════════════════════════════════
# SEZIONE 5  —  HELPER GRAFICI
# ═══════════════════════════════════════════════════════════════
def _spring_path_h(x_start, x_end, y_center=0.5):
    """Molla orizzontale tra x_start e x_end."""
    n_pts = N_COILS * 4 + 2
    xs    = np.linspace(x_start, x_end, n_pts)
    ys    = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        phase = (i - 1) / (n_pts - 2) * N_COILS * 2 * np.pi
        ys[i] = SPRING_W * np.sin(phase)
    ys += y_center
    return xs, ys


def _make_block_h(ax, x_center, y_center, color):
    bw = BLOCK_H * 1.5
    bh = BLOCK_W * 0.45
    return mpatches.FancyBboxPatch(
        (x_center - bw / 2, y_center - bh / 2), bw, bh,
        boxstyle="round,pad=0.005",
        linewidth=1.2,
        edgecolor='#444444',
        facecolor=color,
        zorder=4,
        transform=ax.transData,
    )


# ═══════════════════════════════════════════════════════════════
# SEZIONE 6  —  ANIMAZIONE PRINCIPALE
# ═══════════════════════════════════════════════════════════════
def run_animation(f0, chirp_rate, amp_fraction, speed=1, save_gif=False, mode=None):
    # Risolvi modalità: argomento CLI ha priorità sulla variabile globale
    if mode is None:
        mode = VISUALIZER_MODE
    if mode not in ("target", "max_elongation"):
        raise ValueError(f"Modalità non valida: '{mode}'. Usa 'target' o 'max_elongation'.")

    p = _build_params()
    print(f"[visualizer-BO] Modalità: {mode}")
    print("[visualizer-BO] Eseguo simulazione con:")
    print(f"   F0_CHIRP     = {f0}")
    print(f"   CHIRP_RATE   = {chirp_rate}")
    print(f"   AMP_FRACTION = {amp_fraction}")
    print(f"   X_TARGET     = {p['X_TARGET']} m")

    data      = run_simulation(f0, chirp_rate, amp_fraction, p=p, mode=mode)
    ts        = data["t"]
    xees      = data["xee"]
    xbs       = data["xb"]
    xrs       = data["xr"]
    us        = data["u"]
    x_target  = data["x_target"]
    xee_max   = data["xee_max"]
    t_xee_max = data["t_xee_max"]
    n_frames  = len(ts)
    dt_sim    = float(ts[1] - ts[0]) if len(ts) > 1 else p["DT"]

    if data["target_reached"]:
        print(f"[visualizer-BO] Target raggiunto a t = {data['t_target']:.2f} s")
    else:
        print(f"[visualizer-BO] Target NON raggiunto. Stop: {data['stop_reason']}")
    print(f"[visualizer-BO] Max elongation: {xee_max:.3f} m @ t = {t_xee_max:.2f} s")

    LANE_Y = 0.5

    # ── Campo visivo MSD ──────────────────────────────────
    X_MIN = -0.15
    if mode == "target":
        X_MAX = x_target + 0.50
    else:  # max_elongation: mostra almeno ELONG_VIEW_MIN_MAX m con un margine
        X_MAX = max(xee_max * ELONG_VIEW_PAD, ELONG_VIEW_MIN_MAX) + 0.15

    # ── figura ────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9.5), facecolor=FIG_BG)
    mode_label = "TARGET" if mode == "target" else "MAX ELONGATION"
    fig.suptitle(
        f"BO Visualizer  —  chirp  |  [{mode_label}]  |  "
        f"F0={f0:.3f} Hz  rate={chirp_rate:.4f} Hz/s  amp={amp_fraction:.2f}  |  "
        f"target: {x_target} m",
        color=TEXT_CLR, fontsize=10, fontweight='bold'
    )

    gs = gridspec.GridSpec(
        2, 2,
        left=0.06, right=0.97,
        top=0.92, bottom=0.22,
        wspace=0.30, hspace=0.42,
        height_ratios=[1.6, 1.0],
    )

    ax_msd  = fig.add_subplot(gs[0, :])
    ax_pos  = fig.add_subplot(gs[1, 0])
    ax_ctrl = fig.add_subplot(gs[1, 1])

    for ax in [ax_msd, ax_pos, ax_ctrl]:
        ax.set_facecolor(AX_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_CLR)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.yaxis.label.set_color(TEXT_MUTED)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.title.set_color(TEXT_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle='--', alpha=0.7)

    # ── pannello MSD orizzontale ──────────────────────────
    ax_msd.set_xlim(X_MIN, X_MAX)
    ax_msd.set_ylim(0.0, 1.0)
    ax_msd.set_yticks([])
    ax_msd.set_xlabel('Posizione (m)', color=TEXT_MUTED, fontsize=9)
    ax_msd.set_title(
        f'Sistema MSD  —  Chirp input  [{mode_label}]',
        color=TEXT_CLR, fontsize=10, fontweight='bold'
    )

    ax_msd.axvline(0.0, color='#888888', linewidth=2.0, zorder=1)
    ax_msd.fill_betweenx([0, 1], [X_MIN, X_MIN], [0.0, 0.0], color='#e8e8e8', zorder=0)
    ax_msd.text(X_MIN + 0.02, 0.08, 'SUOLO', color='#999999', fontsize=8, va='bottom')

    # Linea target (sempre visibile in entrambe le modalità)
    ax_msd.axvline(x_target, color=COL_TARGET, linewidth=1.8, linestyle='--', zorder=2, alpha=0.9)
    ax_msd.text(x_target + 0.02, 0.88, f'TARGET\n{x_target} m',
                color=COL_TARGET, fontsize=8, va='top', fontweight='bold')

    # In modalità max_elongation: linea tratteggiata del picco raggiunto
    if mode == "max_elongation" and xee_max > 0.0:
        ax_msd.axvline(xee_max, color=COL_MAXE, linewidth=1.5,
                       linestyle=':', zorder=2, alpha=0.85)
        ax_msd.text(xee_max + 0.04, 0.70,
                    f'MAX\n{xee_max:.2f} m',
                    color=COL_MAXE, fontsize=8, va='top', fontweight='bold')

    ax_msd.axhline(LANE_Y, color='#cccccc', linewidth=1.0, linestyle='-', zorder=1)

    spring_base_line,  = ax_msd.plot([], [], color=COL_KB, linewidth=2.2, zorder=3)
    spring_robot_line, = ax_msd.plot([], [], color=COL_KR, linewidth=2.2, zorder=3)

    ghost_base_patches  = []
    ghost_robot_patches = []
    base_patch  = [None]
    robot_patch = [None]

    state_text = ax_msd.text(
        X_MIN + 0.04, 0.97, '',
        color=TEXT_CLR, fontsize=8, va='top',
        fontfamily='monospace', zorder=10
    )

    # ── pannello posizione ────────────────────────────────
    ax_pos.set_xlim(ts[0], ts[-1])
    y_margin = max(0.05, 0.1 * (np.max(xees) - np.min(xees) + 1e-6))
    ax_pos.set_ylim(min(xees) - y_margin, max(xees) + y_margin)
    ax_pos.axhline(x_target, color=COL_TARGET, linewidth=1.2,
                   linestyle='--', alpha=0.8, label=f'Target {x_target} m')
    if mode == "max_elongation":
        ax_pos.axhline(xee_max, color=COL_MAXE, linewidth=1.2,
                       linestyle=':', alpha=0.8, label=f'Max {xee_max:.2f} m')
        ax_pos.axvline(t_xee_max, color=COL_MAXE, linewidth=0.9,
                       linestyle=':', alpha=0.55)
    ax_pos.set_xlabel('Tempo (s)', fontsize=8)
    ax_pos.set_ylabel('xb + xr  (m)', fontsize=8)
    ax_pos.set_title('Posizione punta (xee)', fontsize=9, fontweight='bold')
    ax_pos.legend(fontsize=7, facecolor='white', framealpha=0.8)

    pos_line,  = ax_pos.plot([], [], color=COL_KB, linewidth=1.8)
    pos_cursor = ax_pos.axvline(0, color='#888888', linewidth=0.8, alpha=0.6)

    # ── pannello chirp ────────────────────────────────────
    u_range = max(us) - min(us) + 1e-6
    ax_ctrl.set_xlim(ts[0], ts[-1])
    ax_ctrl.set_ylim(min(us) - 0.05 * u_range, max(us) + 0.05 * u_range)
    ax_ctrl.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax_ctrl.set_xlabel('Tempo (s)', fontsize=8)
    ax_ctrl.set_ylabel('u = chirp  (m)', fontsize=8)
    ax_ctrl.set_title('Setpoint chirp  (u)', fontsize=9, fontweight='bold')

    ctrl_line,  = ax_ctrl.plot([], [], color=COL_CTRL, linewidth=1.8)
    ctrl_cursor = ax_ctrl.axvline(0, color='#888888', linewidth=0.8, alpha=0.6)

    # ── legenda MSD ───────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=COL_KB,    linewidth=2, label='Molla base (Kb)'),
        Line2D([0], [0], color=COL_KR,    linewidth=2, label='Molla robot (Kr)'),
        mpatches.Patch(facecolor=COL_BASE,  label='Massa base (Mb)'),
        mpatches.Patch(facecolor=COL_ROBOT, label='Massa robot (Mr)'),
    ]
    ax_msd.legend(handles=legend_elements, loc='lower right',
                  fontsize=7, facecolor='white', framealpha=0.8)

    arrival_line_drawn   = [False]
    max_elong_line_drawn = [False]

    history_base_full  = list(xbs)
    history_robot_full = list(xrs)

    # ── STATO RIPRODUZIONE ────────────────────────────────
    state = {
        "frame":   0,
        "playing": True,
        "speed":   max(1, speed),
    }

    n_anim_frames = (n_frames + state["speed"] - 1) // state["speed"]

    # ─────────────────────────────────────────────────────
    # FUNZIONE DI DISEGNO
    # ─────────────────────────────────────────────────────
    def draw_frame(anim_frame_idx):
        i = min(anim_frame_idx * state["speed"], n_frames - 1)

        xb      = float(xbs[i])
        xr      = float(xrs[i])
        x_base  = xb
        x_robot = xb + xr

        # molle
        bw_half = BLOCK_H * 0.75
        sx,  sy  = _spring_path_h(0.0,              x_base  - bw_half, LANE_Y)
        sx2, sy2 = _spring_path_h(x_base + bw_half, x_robot - bw_half, LANE_Y)
        spring_base_line.set_data(sx, sy)
        spring_robot_line.set_data(sx2, sy2)

        # ghost trail
        for gp in ghost_base_patches:
            try: gp.remove()
            except Exception: pass
        for gp in ghost_robot_patches:
            try: gp.remove()
            except Exception: pass
        ghost_base_patches.clear()
        ghost_robot_patches.clear()

        for g in range(1, GHOST_STEPS + 1):
            idx_h = i - g
            if idx_h < 0:
                break
            alpha = GHOST_ALPHA_MAX * (1.0 - g / (GHOST_STEPS + 1))
            gp_b = _make_block_h(ax_msd, history_base_full[idx_h],  LANE_Y, COL_BASE)
            gp_r = _make_block_h(ax_msd, history_robot_full[idx_h], LANE_Y, COL_ROBOT)
            gp_b.set_alpha(alpha); gp_r.set_alpha(alpha)
            gp_b.set_zorder(3);    gp_r.set_zorder(3)
            ax_msd.add_patch(gp_b); ax_msd.add_patch(gp_r)
            ghost_base_patches.append(gp_b)
            ghost_robot_patches.append(gp_r)

        # blocchi correnti
        if base_patch[0] is not None:
            try: base_patch[0].remove()
            except Exception: pass
        if robot_patch[0] is not None:
            try: robot_patch[0].remove()
            except Exception: pass

        bp = _make_block_h(ax_msd, x_base,  LANE_Y, COL_BASE)
        rp = _make_block_h(ax_msd, x_robot, LANE_Y, COL_ROBOT)
        bp.set_alpha(1.0); rp.set_alpha(1.0)
        ax_msd.add_patch(bp); ax_msd.add_patch(rp)
        base_patch[0]  = bp
        robot_patch[0] = rp

        # testo stato
        t_cur = float(ts[i])
        u_cur = float(us[i])
        dist  = abs(x_target - float(xees[i]))
        reached_str = (
            "SI" if (data["target_reached"] and data["t_target"] is not None
                     and t_cur >= data["t_target"])
            else "NO"
        )
        state_text.set_text(
            f"t={t_cur:.2f}s   xb={xb:.3f}m   xr={xr:.3f}m   "
            f"xee={float(xees[i]):.3f}m   u(t)={u_cur:.3f}m   "
            f"err={dist:.3f}m   target={reached_str}"
        )

        # linea arrivo target (modalità target)
        if (not arrival_line_drawn[0]
                and data["target_reached"]
                and data["t_target"] is not None
                and t_cur >= data["t_target"]):
            ax_pos.axvline(data["t_target"], color='#009900', linewidth=1.2,
                           linestyle=':', alpha=0.8,
                           label=f"Arrivo {data['t_target']:.1f}s")
            ax_pos.legend(fontsize=7, facecolor='white', framealpha=0.8)
            arrival_line_drawn[0] = True

        # linea picco massimo (modalità max_elongation)
        if (mode == "max_elongation"
                and not max_elong_line_drawn[0]
                and t_cur >= t_xee_max):
            ax_pos.axvline(t_xee_max, color=COL_MAXE, linewidth=1.2,
                           linestyle=':', alpha=0.8,
                           label=f"Picco {t_xee_max:.1f}s")
            ax_pos.legend(fontsize=7, facecolor='white', framealpha=0.8)
            max_elong_line_drawn[0] = True

        pos_line.set_data(ts[:i + 1], xees[:i + 1])
        pos_cursor.set_xdata([t_cur, t_cur])
        ctrl_line.set_data(ts[:i + 1], us[:i + 1])
        ctrl_cursor.set_xdata([t_cur, t_cur])

        # aggiorna slider senza triggerare callback
        slider_frame.eventson = False
        slider_frame.set_val(anim_frame_idx)
        slider_frame.eventson = True

        return (spring_base_line, spring_robot_line,
                pos_line, pos_cursor, ctrl_line, ctrl_cursor, state_text)

    # ─────────────────────────────────────────────────────
    # WIDGETS  (slider + bottoni)
    # ─────────────────────────────────────────────────────
    BTN_CLR  = '#e8edf2'
    BTN_HOV  = '#cfd8e3'

    ax_slider = fig.add_axes([0.10, 0.12, 0.80, 0.03], facecolor='#f0f0f0')
    ax_pp     = fig.add_axes([0.35, 0.05, 0.12, 0.045])
    ax_rw     = fig.add_axes([0.21, 0.05, 0.12, 0.045])
    ax_back   = fig.add_axes([0.28, 0.05, 0.06, 0.045])
    ax_fwd    = fig.add_axes([0.48, 0.05, 0.06, 0.045])

    slider_frame = Slider(
        ax_slider, 'Frame', 0, n_anim_frames - 1,
        valinit=0, valstep=1, color=COL_KB,
    )
    slider_frame.label.set_color(TEXT_CLR)
    slider_frame.valtext.set_color(TEXT_CLR)

    btn_pp   = Button(ax_pp,   '⏸ Pause', color=BTN_CLR, hovercolor=BTN_HOV)
    btn_rw   = Button(ax_rw,   '⏮ Rewind', color=BTN_CLR, hovercolor=BTN_HOV)
    btn_back = Button(ax_back, '◀ -10',   color=BTN_CLR, hovercolor=BTN_HOV)
    btn_fwd  = Button(ax_fwd,  '+10 ▶',   color=BTN_CLR, hovercolor=BTN_HOV)

    for btn in [btn_pp, btn_rw, btn_back, btn_fwd]:
        btn.label.set_fontsize(9)
        btn.label.set_color(TEXT_CLR)

    def on_slider(val):
        state["frame"] = int(slider_frame.val)
        draw_frame(state["frame"])
        fig.canvas.draw_idle()

    slider_frame.on_changed(on_slider)

    def on_play_pause(event):
        state["playing"] = not state["playing"]
        btn_pp.label.set_text('⏸ Pause' if state["playing"] else '▶ Play')
        fig.canvas.draw_idle()

    btn_pp.on_clicked(on_play_pause)

    def on_rewind(event):
        state["frame"] = 0
        arrival_line_drawn[0]   = False
        max_elong_line_drawn[0] = False
        draw_frame(0)
        fig.canvas.draw_idle()

    btn_rw.on_clicked(on_rewind)

    def on_back(event):
        state["frame"] = max(0, state["frame"] - 10)
        draw_frame(state["frame"])
        fig.canvas.draw_idle()

    btn_back.on_clicked(on_back)

    def on_fwd(event):
        state["frame"] = min(n_anim_frames - 1, state["frame"] + 10)
        draw_frame(state["frame"])
        fig.canvas.draw_idle()

    btn_fwd.on_clicked(on_fwd)

    # ─────────────────────────────────────────────────────
    # LOOP DI ANIMAZIONE  (via FuncAnimation)
    # ─────────────────────────────────────────────────────
    def update(tick):
        if not state["playing"]:
            return draw_frame(state["frame"])
        if state["frame"] >= n_anim_frames - 1:
            state["playing"] = False
            btn_pp.label.set_text('▶ Play')
            return draw_frame(state["frame"])
        state["frame"] += 1
        return draw_frame(state["frame"])

    interval_ms = max(1, int(dt_sim * 1000 * state["speed"]))

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(
        fig, update,
        frames=n_anim_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_gif:
        gif_path = "ROBOT/bo_visualizer.gif"
        print(f"[visualizer-BO] Salvo GIF in: {gif_path}...")
        anim.save(gif_path, writer='pillow', fps=max(1, int(1.0 / (dt_sim * speed))))
        print("[visualizer-BO] GIF salvata.")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════
# SEZIONE 7  —  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualizer animato per la Bayesian Optimization (chirp) — sistema MSD"
    )
    parser.add_argument(
        "--f0", type=float, default=F0_CHIRP_OPT,
        help=f"Frequenza iniziale del chirp in Hz (default={F0_CHIRP_OPT})"
    )
    parser.add_argument(
        "--rate", type=float, default=CHIRP_RATE_OPT,
        help=f"Chirp rate in Hz/s (default={CHIRP_RATE_OPT})"
    )
    parser.add_argument(
        "--amp", type=float, default=AMP_FRACTION_OPT,
        help=f"Ampiezza come frazione di X_MAX_ROBOT (default={AMP_FRACTION_OPT})"
    )
    parser.add_argument(
        "--speed", "-s", type=int, default=1,
        help="Velocita di riproduzione (default=1, usa 5 o 10 per accelerare)"
    )
    parser.add_argument(
        "--gif", action="store_true",
        help="Salva l'animazione come ROBOT/bo_visualizer.gif"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["target", "max_elongation"],
        help=(
            "Modalità visualizzatore: 'target' (default) oppure "
            "'max_elongation' (campo visivo ~4 m, ferma dopo il picco)."
        )
    )

    args = parser.parse_args()
    run_animation(
        f0=args.f0,
        chirp_rate=args.rate,
        amp_fraction=args.amp,
        speed=args.speed,
        save_gif=args.gif,
        mode=args.mode,
    )
