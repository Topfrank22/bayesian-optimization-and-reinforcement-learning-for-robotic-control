"""
visualize_rl_run.py
====================
Visualizer animato per le run di RL_tommy e RL_tommy_curriculum.

USO:
    python visualize_rl_run.py <cartella_run>

Esempio:
    python visualize_rl_run.py ROBOT/RL_results/run_750ep_reward7_arrivo_morbido_20260101_1200

La cartella deve contenere:
    - simulation_config.json   (parametri fisici + rete)
    - agent_actor.pth          (pesi dell'actor)
    - test_rollout.npz         (traiettoria pre-registrata, usata come fallback)

Se agent_actor.pth e' presente viene eseguita una simulazione live;
altrimenti viene riprodotto il rollout salvato in test_rollout.npz.

MODALITA' DI VISUALIZZAZIONE
─────────────────────────────
  VIS_MODE = "target"   → la simulazione termina non appena il tip raggiunge
                          y_des (target), poi prosegue per T_HOLD secondi
                          con l'ultima azione dell'agente tenuta fissa.

  VIS_MODE = "generico" → nessun early-stop: la simulazione gira per
                          T_GENERIC secondi esatti, indipendentemente
                          dal target. Adatta per policy di massimizzazione
                          dell'elongazione (reward 8) o qualsiasi policy
                          che non converge su un target fisso.

  --target               → imposta un target personalizzato (sovrascrive y_des
                          del config). Valido sia in modalità "target" (come
                          condizione di early-stop) sia come riferimento visivo
                          in modalità "generico".

CONTROLLI INTERATTIVI:
  - Slider "Frame" : scrubbing avanti/indietro su qualsiasi fotogramma
  - [Play/Pause]   : avvia o mette in pausa l'animazione
  - [⏮ Rewind]     : riporta la riproduzione all'inizio
  - [◀ -10]        : salta 10 frame indietro
  - [+10 ▶]        : salta 10 frame avanti

VISUALIZZAZIONE PRECOMPUTATA:
  La simulazione viene eseguita al completo (con dt = 0.01 s), dopodiché
  i frame di visualizzazione vengono sottocampionati a DISPLAY_FPS fissi.
  L'animazione scorre quindi a velocità reale indipendentemente dalla
  risoluzione temporale della simulazione.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn

# ============================================================
# ★  CONFIGURAZIONE UTENTE — modifica solo questi parametri  ★
# ============================================================

RUN_FOLDER = r"C:\Users\tomga\Desktop\Progetti ML\ROBOT\RL_results\Random_target_Gasu_vel_longer_training_with_higher_noise"
AGENT_PATH = r"C:\Users\tomga\Desktop\Progetti ML\ROBOT\RL_results\Random_target_Gasu_vel_longer_training_with_higher_noise\training_snapshots\snapshot_ep001500.json"
USE_RUN_FOLDER = True   #false - run folder, true - agent path

VIS_MODE  = "target"
T_GENERIC = 20.0
T_HOLD = 2.0
DISPLAY_FPS = 30
TARGET_OVERRIDE = 1.70

ACTUATOR_PHYSICAL_LIMIT = False 

V_MAX_ATTUATORE = 13.33

MSD_X_MIN = -2.0
MSD_X_MAX = None

POS_Y_MIN = None
POS_Y_MAX = None
POS_Y_MARGIN = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. RETE ACTOR
# ============================================================

def _build_mlp(input_dim, output_dim, hidden_size, hidden_layers, output_activation=None):
    layers, in_dim = [], input_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(in_dim, hidden_size), nn.ELU()]
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, u_min, u_max,
                 hidden_size=256, hidden_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer('u_min_t', torch.tensor(u_min, dtype=torch.float32))
        self.register_buffer('u_max_t', torch.tensor(u_max, dtype=torch.float32))
        self.u_min = u_min
        self.u_max = u_max
        self.network = _build_mlp(state_dim, action_dim, hidden_size,
                                   hidden_layers, nn.Tanh())

    def forward(self, state):
        t = self.network(state)
        scaled = 0.5 * (self.u_max - self.u_min) * t + 0.5 * (self.u_max + self.u_min)
        return torch.clamp(scaled, self.u_min_t, self.u_max_t)

    def act(self, state_np):
        s = torch.tensor(state_np, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            return self.forward(s).cpu().numpy().item()


# ============================================================
# 2. SIMULATORE MSD
# ============================================================

class MSDSimulator:
    def __init__(self, cfg, actuator_physical_limit=False, v_max_attuatore=13.33):
        self.dt = cfg["dt"]
        self.Mr = cfg["Mr"]
        self.Mb = cfg["Mb"]
        self.Kr = cfg["Kr"]
        self.Kb = cfg["Kb"]
        self.Dr = 2 * cfg["hr"] * np.sqrt(cfg["Kr"] * cfg["Mr"])
        self.Db = 2 * cfg["hb"] * np.sqrt(cfg["Kb"] * cfg["Mb"])
        self.xb0 = 0.0
        self.M_mat = np.array(
            [[self.Mr + self.Mb, self.Mr],
             [self.Mr, self.Mr]], dtype=float
        )
        self.x = np.zeros(4)
        self.actuator_physical_limit = actuator_physical_limit
        self.v_max_attuatore = v_max_attuatore
        self.u_prev = 0.0

    def reset(self):
        self.x = np.zeros(4)
        self.u_prev = 0.0
        return self.x.copy()

    def _rhs(self, z, u):
        xb, xb_dot, xr, xr_dot = z
        rhs = np.array([
            -self.Db * xb_dot - self.Kb * (xb - self.xb0),
            -self.Dr * xr_dot - self.Kr * (xr - u),
        ], dtype=float)
        xb_ddot, xr_ddot = np.linalg.solve(self.M_mat, rhs)
        return np.array([xb_dot, xb_ddot, xr_dot, xr_ddot], dtype=float)

    def _rk4(self, z, u):
        dt = self.dt
        k1 = self._rhs(z, u)
        k2 = self._rhs(z + 0.5 * dt * k1, u)
        k3 = self._rhs(z + 0.5 * dt * k2, u)
        k4 = self._rhs(z + dt * k3, u)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, u):
        u_target = float(np.clip(u, -1.0, 1.0))
        if self.actuator_physical_limit:
            max_delta_u = self.v_max_attuatore * self.dt
            delta_u = np.clip(u_target - self.u_prev, -max_delta_u, max_delta_u)
            u_real = self.u_prev + delta_u
            self.u_prev = u_real
        else:
            u_real = u_target
        self.x = self._rk4(self.x, u_real)
        return self.x.copy(), u_real


# ============================================================
# 3. CONFIG E HELPERS
# ============================================================

_CFG_DEFAULTS = {
    "state_dim": 5,
    "action_dim": 1,
    "actor_hidden_size": 64,
    "actor_hidden_layers": 2,
    "u_min": -1.0,
    "u_max": 1.0,
    "y_des": 1.8,
    "T_max": 20.0,
    "reward_name": "unknown",
    "dt": 0.01,
    "Mr": 2.5,
    "Mb": 70.0,
    "Kr": 1500.0,
    "Kb": 4000.0,
    "hr": 0.30,
    "hb": 0.25,
}

_STATE_SCALE_DEFAULT = np.array([0.15, 1.5, 2.0, 40.0, 2.0], dtype=np.float32)
_STATE_SCALE = _STATE_SCALE_DEFAULT


def normalize_states(s):
    s = np.asarray(s, dtype=np.float32)
    scale = _STATE_SCALE[: s.shape[0]] if s.shape[0] <= len(_STATE_SCALE) else _STATE_SCALE
    return s / scale



def _infer_actor_architecture(checkpoint, fallback_state_dim=5, fallback_hidden_size=64, fallback_hidden_layers=2):
    if not isinstance(checkpoint, dict):
        return fallback_state_dim, fallback_hidden_size, fallback_hidden_layers

    weight_keys = sorted(
        k for k in checkpoint.keys()
        if k.startswith("network.") and k.endswith(".weight")
    )
    if not weight_keys:
        return fallback_state_dim, fallback_hidden_size, fallback_hidden_layers

    first_w = checkpoint[weight_keys[0]]
    if first_w.dim() != 2:
        return fallback_state_dim, fallback_hidden_size, fallback_hidden_layers

    state_dim = int(first_w.shape[1])
    hidden_size = int(first_w.shape[0])
    hidden_layers = len(weight_keys) - 1
    return state_dim, hidden_size, hidden_layers


def _resolve_target_value(cfg):
    for key in ("y_des", "target_position", "target", "TARGET_POSITION"):
        value = cfg.get(key)
        if value is not None:
            return float(value)
    return 1.8


def _prepare_actor_state(state, cfg, actor):
    state = np.asarray(state, dtype=np.float32)
    actor_state_dim = getattr(actor, "state_dim", state.shape[0])

    if state.shape[0] == actor_state_dim:
        if actor_state_dim == 5:
            return normalize_states(state)
        return state

    if state.shape[0] + 1 == actor_state_dim:
        target_value = _resolve_target_value(cfg)
        state = np.concatenate([state, np.array([target_value], dtype=np.float32)])
        return normalize_states(state)

    raise ValueError(
        f"Stato simulatore di dimensione {state.shape[0]} incompatibile con actor_state_dim={actor_state_dim}."
    )


def _read_actuator_config(raw_cfg, user_actuator_limit, user_v_max):
    limit = user_actuator_limit
    v_max = user_v_max
    actuator_cfg = raw_cfg.get("attuatore", {})
    if limit is None:
        limit = actuator_cfg.get("actuator_physical_limit", False)
        if v_max is None:
            v_max = actuator_cfg.get("v_max_attuatore", 13.33)
    elif v_max is None:
        v_max = actuator_cfg.get("v_max_attuatore", 13.33)
    if limit is None:
        limit = False
    if v_max is None:
        v_max = 13.33
    return bool(limit), float(v_max)


def load_run(folder, agent_path=None, target_override=None, actuator_physical_limit=None, v_max_attuatore=None):
    global _STATE_SCALE

    cfg_path = os.path.join(folder, "simulation_config.json")
    actor_path = agent_path if agent_path is not None else os.path.join(folder, "agent_actor.pth")
    rollout_path = os.path.join(folder, "test_rollout.npz")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"simulation_config.json non trovato in: {folder}")

    with open(cfg_path, "r") as f:
        raw_cfg = json.load(f)

    cfg = {**_CFG_DEFAULTS, **raw_cfg}

    # I parametri fisici del sistema sono salvati dal training annidati sotto
    # "sistema": {Mr, Mb, Kr, Kb, hr, hb}. Se presenti, sovrascrivono i default.
    sistema_cfg = raw_cfg.get("sistema", {})
    for key in ("Mr", "Mb", "Kr", "Kb", "hr", "hb"):
        if key in sistema_cfg:
            cfg[key] = float(sistema_cfg[key])

    # Il vettore di normalizzazione dello stato è salvato dal training sotto
    # "normalizzazione": {"state_scale": [...], "reward_scale": ...}.
    norm_cfg = raw_cfg.get("normalizzazione", {})
    state_scale_cfg = norm_cfg.get("state_scale")
    if state_scale_cfg:
        _STATE_SCALE = np.array(state_scale_cfg, dtype=np.float32)
    else:
        _STATE_SCALE = _STATE_SCALE_DEFAULT

    # target_threshold/vel_threshold salvati dal training, usati come fallback
    # per coerenza con i criteri di successo usati in fase di training.
    cfg["target_threshold"] = float(raw_cfg.get("target_threshold", 0.05))
    cfg["vel_threshold"] = float(raw_cfg.get("vel_threshold", 10.0))

    cfg["y_des"] = _resolve_target_value(cfg)


    if target_override is not None:
        cfg["y_des"] = float(target_override)
        print(f"[visualizer] Target sovrascritto dall'utente: y_des = {cfg['y_des']}")

    cfg["reward_name"] = raw_cfg.get("reward_name", raw_cfg.get("training_mode", "unknown"))

    act_limit, act_vmax = _read_actuator_config(raw_cfg, actuator_physical_limit, v_max_attuatore)
    cfg["actuator_physical_limit"] = act_limit
    cfg["v_max_attuatore"] = act_vmax

    actor = None
    if os.path.exists(actor_path):
        checkpoint = torch.load(actor_path, map_location=device)
        inferred_sd, inferred_hs, inferred_hl = _infer_actor_architecture(
            checkpoint,
            fallback_state_dim=cfg["state_dim"],
            fallback_hidden_size=cfg["actor_hidden_size"],
            fallback_hidden_layers=cfg["actor_hidden_layers"],
        )
        cfg["state_dim"] = inferred_sd
        cfg["actor_hidden_size"] = inferred_hs
        cfg["actor_hidden_layers"] = inferred_hl

        actor = Actor(
            state_dim=inferred_sd,
            action_dim=cfg["action_dim"],
            u_min=cfg["u_min"],
            u_max=cfg["u_max"],
            hidden_size=inferred_hs,
            hidden_layers=inferred_hl,
        ).to(device)
        actor.load_state_dict(checkpoint)
        actor.eval()
        print(f"[visualizer] Actor caricato da: {actor_path}")
        print(f"[visualizer] Architettura inferita: state_dim={inferred_sd}, hidden_size={inferred_hs}, hidden_layers={inferred_hl}")

    rollout = None
    if os.path.exists(rollout_path):
        rollout = np.load(rollout_path)
        print(f"[visualizer] Rollout caricato da: {rollout_path}")

    if actor is None and rollout is None:
        raise FileNotFoundError("Nessun artifact valido trovato. Servono almeno agent_actor.pth o test_rollout.npz.")

    return cfg, actor, rollout


# ============================================================
# 4. SIMULAZIONE LIVE
# ============================================================

def simulate_live_target(actor, cfg, t_hold=2.0):
    sim = MSDSimulator(
        cfg,
        actuator_physical_limit=cfg.get("actuator_physical_limit", False),
        v_max_attuatore=cfg.get("v_max_attuatore", 13.33),
    )
    state = sim.reset()
    dt = cfg["dt"]
    T_max = cfg["T_max"]
    y_des = cfg["y_des"]
    n_steps = int(T_max / dt)

    ts, ys, dys, us, xs = [], [], [], [], []
    t = 0.0
    target_reached = False
    t_target = None
    u_last = 0.0

    for _ in range(n_steps):
        if not target_reached:
            u = actor.act(_prepare_actor_state(state, cfg, actor))
            u_last = u
        else:
            u = u_last

        ts.append(t)
        ys.append(state[0] + state[2])
        dys.append(state[1] + state[3])
        us.append(u)
        xs.append(state.copy())

        state, u_real = sim.step(u)
        us[-1] = u_real
        t += dt

        if not target_reached and (state[0] + state[2] >= y_des or ys[-1] >= y_des):
            target_reached = True
            t_target = t

        if target_reached and (t - t_target >= t_hold):
            ts.append(t)
            ys.append(state[0] + state[2])
            dys.append(state[1] + state[3])
            us.append(u_real)
            xs.append(state.copy())
            break

    return (
        np.array(ts, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        np.array(dys, dtype=np.float32),
        np.array(us, dtype=np.float32),
        np.array(xs, dtype=np.float32),
    )


def simulate_live_generico(actor, cfg, t_generic):
    sim = MSDSimulator(
        cfg,
        actuator_physical_limit=cfg.get("actuator_physical_limit", False),
        v_max_attuatore=cfg.get("v_max_attuatore", 13.33),
    )
    state = sim.reset()
    dt = cfg["dt"]
    n_steps = int(t_generic / dt)

    ts, ys, dys, us, xs = [], [], [], [], []
    t = 0.0
    for _ in range(n_steps):
        u = actor.act(_prepare_actor_state(state, cfg, actor))
        ts.append(t)
        ys.append(state[0] + state[2])
        dys.append(state[1] + state[3])
        us.append(u)
        xs.append(state.copy())
        state, u_real = sim.step(u)
        us[-1] = u_real
        t += dt

    return (
        np.array(ts, dtype=np.float32),
        np.array(ys, dtype=np.float32),
        np.array(dys, dtype=np.float32),
        np.array(us, dtype=np.float32),
        np.array(xs, dtype=np.float32),
    )


def build_visual_frames(ts, ys, us, xs, display_fps=30):
    dt_vis = 1.0 / display_fps
    t_end = float(ts[-1]) if len(ts) > 0 else 0.0
    t_visual = np.arange(0.0, t_end + 0.5 * dt_vis, dt_vis)
    sample_idx = np.searchsorted(ts, t_visual, side='left')
    sample_idx = np.clip(sample_idx, 0, len(ts) - 1)

    return {
        "t_visual": t_visual,
        "sample_idx": sample_idx,
        "y": ys[sample_idx],
        "u": us[sample_idx],
        "xb": xs[sample_idx, 0],
        "xr": xs[sample_idx, 2],
        "tip": xs[sample_idx, 0] + xs[sample_idx, 2],
    }


# ============================================================
# 5. GRAFICA
# ============================================================

BLOCK_W  = 0.25
BLOCK_H  = 0.08
SPRING_W = 0.04
N_COILS  = 6

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
COL_PEAK   = '#8B00CC'

GHOST_STEPS = 4
GHOST_ALPHA_MAX = 0.30


def _spring_path_h(x_start, x_end, y_center=0.5):
    n_pts = N_COILS * 4 + 2
    xsp = np.linspace(x_start, x_end, n_pts)
    ysp = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        phase = (i - 1) / (n_pts - 2) * N_COILS * 2 * np.pi
        ysp[i] = SPRING_W * np.sin(phase)
    ysp += y_center
    return xsp, ysp


def run_animation(folder, agent_path=None, speed=1, save_gif=False, vis_mode=None, t_generic=None,
                  msd_x_min=None, msd_x_max=None,
                  pos_y_min=None, pos_y_max=None, pos_y_margin=None,
                  target=None, display_fps=None, t_hold=None,
                  actuator_physical_limit=None, v_max_attuatore=None):
    if vis_mode is None: vis_mode = VIS_MODE
    if t_generic is None: t_generic = T_GENERIC
    if msd_x_min is None: msd_x_min = MSD_X_MIN
    if msd_x_max is None: msd_x_max = MSD_X_MAX
    if pos_y_min is None: pos_y_min = POS_Y_MIN
    if pos_y_max is None: pos_y_max = POS_Y_MAX
    if pos_y_margin is None: pos_y_margin = POS_Y_MARGIN
    if target is None: target = TARGET_OVERRIDE
    if display_fps is None: display_fps = DISPLAY_FPS
    if t_hold is None: t_hold = T_HOLD
    if actuator_physical_limit is None: actuator_physical_limit = ACTUATOR_PHYSICAL_LIMIT
    if v_max_attuatore is None: v_max_attuatore = V_MAX_ATTUATORE

    cfg, actor, rollout = load_run(
        folder,
        agent_path=agent_path,
        target_override=target,
        actuator_physical_limit=actuator_physical_limit,
        v_max_attuatore=v_max_attuatore
    )

    if actor is not None:
        if vis_mode == "generico":
            print(f"[visualizer] Modalità GENERICO — simulazione per {t_generic}s...")
            ts, ys, dys, us, xs = simulate_live_generico(actor, cfg, t_generic)
        else:
            print(f"[visualizer] Modalità TARGET — simulazione live (hold={t_hold}s)...")
            ts, ys, dys, us, xs = simulate_live_target(actor, cfg, t_hold=t_hold)
    else:
        print("[visualizer] Riproduco rollout pre-registrato...")
        ts = rollout["t"]; ys = rollout["y"]; dys = rollout["dy"]
        us = rollout["u"]; xs = rollout["x"]

    y_des = cfg["y_des"]
    dt = cfg["dt"]

    visual = build_visual_frames(ts, ys, us, xs, display_fps=display_fps)
    sample_idx = visual["sample_idx"]
    n_anim_frames = len(sample_idx)
    interval_ms = int(round(1000 / display_fps))

    t_sim = ts
    y_sim = ys
    u_sim = us

    y_vis = visual["y"]
    u_vis = visual["u"]
    xb_vis = visual["xb"]
    xr_vis = visual["xr"]
    tip_vis = visual["tip"]

    y_peak = float(np.max(ys))
    y_valley = float(np.min(ys))
    t_peak_idx = int(np.argmax(ys))
    t_peak = float(ts[t_peak_idx])

    X_MIN = msd_x_min
    if msd_x_max is not None:
        X_MAX = msd_x_max
    else:
        X_MAX = max(y_peak + 0.20, y_des + 0.50)
    X_MIN = min(X_MIN, y_valley - 0.10)

    if pos_y_min is not None and pos_y_max is not None:
        PY_MIN = pos_y_min
        PY_MAX = pos_y_max
    else:
        PY_MIN = (pos_y_min if pos_y_min is not None else y_valley - pos_y_margin)
        PY_MAX = (pos_y_max if pos_y_max is not None else y_peak + pos_y_margin)

    LANE_Y = 0.5

    if vis_mode == "target":
        target_info = f"target: {y_des} m  |  hold: {t_hold}s"
    else:
        target_info = f"durata: {t_generic}s"

    title_str = (
        f"RL Visualizer  [{vis_mode.upper()}]  —  reward: {cfg.get('reward_name','?')}  |  "
        f"{target_info}  |  peak tip: {y_peak:.4f} m @ t={t_peak:.2f}s  |  "
        f"dt={dt}s  |  {display_fps} FPS"
    )

    fig = plt.figure(figsize=(14, 9.5), facecolor=FIG_BG)
    fig.suptitle(title_str, color=TEXT_CLR, fontsize=11, fontweight='bold')

    gs = gridspec.GridSpec(
        2, 2,
        left=0.06, right=0.97,
        top=0.92, bottom=0.22,
        wspace=0.30, hspace=0.42,
        height_ratios=[1.6, 1.0],
    )
    ax_msd = fig.add_subplot(gs[0, :])
    ax_pos = fig.add_subplot(gs[1, 0])
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

    ax_msd.set_xlim(X_MIN, X_MAX)
    ax_msd.set_ylim(0.0, 1.0)
    ax_msd.set_yticks([])
    ax_msd.set_xlabel('Posizione (m)', color=TEXT_MUTED, fontsize=9)
    ax_msd.set_title('Sistema MSD', color=TEXT_CLR, fontsize=10, fontweight='bold')

    ax_msd.axvline(0.0, color='#888888', linewidth=2.0, zorder=1)
    if X_MIN < 0:
        ax_msd.fill_betweenx([0, 1], [X_MIN, X_MIN], [0.0, 0.0],
                              color='#ffe8e8', alpha=0.5, zorder=0)
    ax_msd.axvline(y_des, color=COL_TARGET, linewidth=1.8, linestyle='--', zorder=2, alpha=0.9)
    ax_msd.axvline(y_peak, color=COL_PEAK, linewidth=1.6, linestyle=':', zorder=2, alpha=0.85)
    ax_msd.axhline(LANE_Y, color='#cccccc', linewidth=1.0, linestyle='-', zorder=1)

    spring_base_line, = ax_msd.plot([], [], color=COL_KB, linewidth=2.2, zorder=3)
    spring_robot_line, = ax_msd.plot([], [], color=COL_KR, linewidth=2.2, zorder=3)

    ghost_base_patches = []
    ghost_robot_patches = []
    for _ in range(GHOST_STEPS):
        gp_b = mpatches.FancyBboxPatch((0, 0), BLOCK_H * 1.5, BLOCK_W * 0.45,
                                       boxstyle="round,pad=0.005",
                                       linewidth=1.2, edgecolor='#444444', facecolor=COL_BASE,
                                       zorder=3, transform=ax_msd.transData)
        gp_r = mpatches.FancyBboxPatch((0, 0), BLOCK_H * 1.5, BLOCK_W * 0.45,
                                       boxstyle="round,pad=0.005",
                                       linewidth=1.2, edgecolor='#444444', facecolor=COL_ROBOT,
                                       zorder=3, transform=ax_msd.transData)
        gp_b.set_alpha(0.0)
        gp_r.set_alpha(0.0)
        ax_msd.add_patch(gp_b)
        ax_msd.add_patch(gp_r)
        ghost_base_patches.append(gp_b)
        ghost_robot_patches.append(gp_r)

    def _make_block_h():
        bw = BLOCK_H * 1.5
        bh = BLOCK_W * 0.45
        return mpatches.FancyBboxPatch(
            (0, 0), bw, bh,
            boxstyle="round,pad=0.005",
            linewidth=1.2, edgecolor='#444444',
            zorder=4, transform=ax_msd.transData,
        )

    base_patch = _make_block_h()
    robot_patch = _make_block_h()
    base_patch.set_facecolor(COL_BASE)
    robot_patch.set_facecolor(COL_ROBOT)
    ax_msd.add_patch(base_patch)
    ax_msd.add_patch(robot_patch)

    def _update_patch_pos(patch, x_center, y_center):
        bw = BLOCK_H * 1.5
        bh = BLOCK_W * 0.45
        patch.set_width(bw)
        patch.set_height(bh)
        patch.set_x(x_center - bw / 2)
        patch.set_y(y_center - bh / 2)

    state_text = ax_msd.text(X_MIN + abs(X_MAX - X_MIN) * 0.02, 0.97, '',
                             color=TEXT_CLR, fontsize=8, va='top',
                             fontfamily='monospace', zorder=10)

    ax_pos.set_xlim(t_sim[0], t_sim[-1])
    ax_pos.set_ylim(PY_MIN, PY_MAX)
    ax_pos.axhline(0.0, color='#aaaaaa', linewidth=0.8, linestyle=':')
    ax_pos.axhline(y_des, color=COL_TARGET, linewidth=1.2,
                   linestyle='--', alpha=0.8, label=f'Target {y_des} m')
    ax_pos.axhline(y_peak, color=COL_PEAK, linewidth=1.0,
                   linestyle=':', alpha=0.7, label=f'max dist reached {y_peak:.4f} m')
    ax_pos.axvline(t_peak, color=COL_PEAK, linewidth=0.8,
                   linestyle=':', alpha=0.5)
    ax_pos.set_xlabel('Tempo (s)', fontsize=8)
    ax_pos.set_ylabel('xb + xr  (m)', fontsize=8)
    ax_pos.set_title('Posizione punta (tip)', fontsize=9, fontweight='bold')
    ax_pos.legend(fontsize=7, facecolor='white', framealpha=0.8)

    pos_line, = ax_pos.plot([], [], color=COL_KB, linewidth=1.8)
    pos_cursor = ax_pos.axvline(0, color='#888888', linewidth=0.8, alpha=0.6)

    ax_ctrl.set_xlim(t_sim[0], t_sim[-1])
    ax_ctrl.set_ylim(-1.05, 1.05)
    ax_ctrl.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax_ctrl.axhline(cfg["u_max"], color=COL_TARGET, linewidth=0.8,
                    linestyle=':', alpha=0.6, label=f'+{cfg["u_max"]}')
    ax_ctrl.axhline(cfg["u_min"], color=COL_TARGET, linewidth=0.8,
                    linestyle=':', alpha=0.6, label=f'{cfg["u_min"]}')
    ax_ctrl.set_xlabel('Tempo (s)', fontsize=8)
    ax_ctrl.set_ylabel('u (azione)', fontsize=8)
    ax_ctrl.set_title('Controllo', fontsize=9, fontweight='bold')
    ax_ctrl.legend(fontsize=7, facecolor='white', framealpha=0.8)

    ctrl_line, = ax_ctrl.plot([], [], color=COL_CTRL, linewidth=1.8)
    ctrl_cursor = ax_ctrl.axvline(0, color='#888888', linewidth=0.8, alpha=0.6)

    legend_elements = [
        Line2D([0],[0], color=COL_KB, linewidth=2, label='Molla base (Kb)'),
        Line2D([0],[0], color=COL_KR, linewidth=2, label='Molla robot (Kr)'),
        mpatches.Patch(facecolor=COL_BASE, label='Massa base (Mb)'),
        mpatches.Patch(facecolor=COL_ROBOT, label='Massa robot (Mr)'),
    ]
    ax_msd.legend(handles=legend_elements, loc='lower right',
                  fontsize=7, facecolor='white', framealpha=0.8)

    BTN_CLR = '#e8edf2'
    BTN_HOV = '#cfd8e3'

    ax_slider = fig.add_axes([0.10, 0.12, 0.80, 0.03], facecolor='#f0f0f0')
    ax_pp = fig.add_axes([0.35, 0.05, 0.12, 0.045])
    ax_rw = fig.add_axes([0.21, 0.05, 0.12, 0.045])
    ax_back = fig.add_axes([0.28, 0.05, 0.06, 0.045])
    ax_fwd = fig.add_axes([0.48, 0.05, 0.06, 0.045])

    slider_frame = Slider(ax_slider, 'Frame', 0, n_anim_frames - 1,
                          valinit=0, valstep=1, color=COL_KB)
    btn_pp = Button(ax_pp, '⏸ Pause', color=BTN_CLR, hovercolor=BTN_HOV)
    btn_rw = Button(ax_rw, '⏮ Rewind', color=BTN_CLR, hovercolor=BTN_HOV)
    btn_back = Button(ax_back, '◀ -10', color=BTN_CLR, hovercolor=BTN_HOV)
    btn_fwd = Button(ax_fwd, '+10 ▶', color=BTN_CLR, hovercolor=BTN_HOV)

    play_state = {"frame": 0, "playing": True}

    def draw_frame(anim_frame_idx):
        anim_frame_idx = int(np.clip(anim_frame_idx, 0, n_anim_frames - 1))
        i = int(sample_idx[anim_frame_idx])

        xb = float(xb_vis[anim_frame_idx])
        xr = float(xr_vis[anim_frame_idx])
        x_base = xb
        x_robot = float(tip_vis[anim_frame_idx])

        sx, sy = _spring_path_h(0.0, x_base - 0.04, LANE_Y)
        sx2, sy2 = _spring_path_h(x_base + 0.04, x_robot - 0.04, LANE_Y)
        spring_base_line.set_data(sx, sy)
        spring_robot_line.set_data(sx2, sy2)

        for g in range(GHOST_STEPS):
            idx_vis = anim_frame_idx - (g + 1)
            if idx_vis >= 0:
                alpha = GHOST_ALPHA_MAX * (1.0 - (g + 1) / (GHOST_STEPS + 1))
                _update_patch_pos(ghost_base_patches[g], float(xb_vis[idx_vis]), LANE_Y)
                _update_patch_pos(ghost_robot_patches[g], float(tip_vis[idx_vis]), LANE_Y)
                ghost_base_patches[g].set_alpha(alpha)
                ghost_robot_patches[g].set_alpha(alpha)
            else:
                ghost_base_patches[g].set_alpha(0.0)
                ghost_robot_patches[g].set_alpha(0.0)

        _update_patch_pos(base_patch, x_base, LANE_Y)
        _update_patch_pos(robot_patch, x_robot, LANE_Y)

        t_cur = float(t_sim[i])
        u_cur = float(u_vis[anim_frame_idx])
        tip = float(y_vis[anim_frame_idx])

        state_text.set_text(
            f"t={t_cur:.2f}s   xb={xb:.3f}m   xr={xr:.3f}m   "
            f"tip={tip:.4f}m   max_dist={y_peak:.4f}m   u={u_cur:.3f}"
        )

        pos_line.set_data(t_sim[:i + 1], y_sim[:i + 1])
        pos_cursor.set_xdata([t_cur, t_cur])
        ctrl_line.set_data(t_sim[:i + 1], u_sim[:i + 1])
        ctrl_cursor.set_xdata([t_cur, t_cur])

        slider_frame.eventson = False
        slider_frame.set_val(anim_frame_idx)
        slider_frame.eventson = True

        return (spring_base_line, spring_robot_line,
                pos_line, pos_cursor, ctrl_line, ctrl_cursor, state_text)

    def on_slider(val):
        play_state["frame"] = int(slider_frame.val)
        draw_frame(play_state["frame"])
        fig.canvas.draw_idle()

    slider_frame.on_changed(on_slider)

    def on_play_pause(event):
        play_state["playing"] = not play_state["playing"]
        btn_pp.label.set_text('⏸ Pause' if play_state["playing"] else '▶ Play')
        fig.canvas.draw_idle()

    btn_pp.on_clicked(on_play_pause)

    def on_rewind(event):
        play_state["frame"] = 0
        play_state["playing"] = False
        btn_pp.label.set_text('▶ Play')
        draw_frame(0)
        fig.canvas.draw_idle()

    btn_rw.on_clicked(on_rewind)

    def on_back(event):
        play_state["frame"] = max(0, play_state["frame"] - 10)
        play_state["playing"] = False
        btn_pp.label.set_text('▶ Play')
        draw_frame(play_state["frame"])
        fig.canvas.draw_idle()

    btn_back.on_clicked(on_back)

    def on_fwd(event):
        play_state["frame"] = min(n_anim_frames - 1, play_state["frame"] + 10)
        play_state["playing"] = False
        btn_pp.label.set_text('▶ Play')
        draw_frame(play_state["frame"])
        fig.canvas.draw_idle()

    btn_fwd.on_clicked(on_fwd)

    def update(_):
        if not play_state["playing"]:
            return draw_frame(play_state["frame"])
        if play_state["frame"] >= n_anim_frames - 1:
            play_state["playing"] = False
            btn_pp.label.set_text('▶ Play')
            return draw_frame(play_state["frame"])
        play_state["frame"] += 1
        return draw_frame(play_state["frame"])

    anim = FuncAnimation(
        fig, update,
        frames=n_anim_frames,
        interval=int(round(1000 / display_fps)),
        blit=False,
        repeat=False,
    )

    if save_gif:
        gif_path = os.path.join(folder, "simulation.gif")
        anim.save(gif_path, writer='pillow', fps=display_fps)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizer animato per run RL_tommy / RL_tommy_curriculum")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Cartella della run (contiene simulation_config.json)")
    parser.add_argument("--speed", "-s", type=int, default=1,
                        help="Velocita' di riproduzione (default=1, usato solo per GIF)")
    parser.add_argument("--gif", action="store_true",
                        help="Salva simulation.gif nella cartella run")
    parser.add_argument("--mode", "-m", type=str, default=None,
                        choices=["target", "generico"],
                        help="Modalita': 'target' o 'generico'")
    parser.add_argument("--t-generic", type=float, default=None,
                        help=f"Durata simulazione in modalita' generico (default={T_GENERIC}s)")
    parser.add_argument("--t-hold", type=float, default=None,
                        help=f"Durata dell'hold dopo il target in modalita' target (default={T_HOLD}s)")
    parser.add_argument("--target", "--y-des", type=float, default=None,
                        help="Target personalizzato (sovrascrive y_des del config)")
    parser.add_argument("--fps", type=int, default=None,
                        help=f"Frame rate di visualizzazione (default={DISPLAY_FPS})")
    parser.add_argument("--xmin", type=float, default=None,
                        help="Limite sinistro asse MSD (m), es. -2.0")
    parser.add_argument("--xmax", type=float, default=None,
                        help="Limite destro asse MSD (m), es. 2.0")
    parser.add_argument("--ymin", type=float, default=None,
                        help="Limite inferiore grafico posizione (m), es. -2.0")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Limite superiore grafico posizione (m), es. 2.0")
    parser.add_argument("--actuator-limit", action="store_true", default=None,
                        help="Attiva il rate limiter fisico dell'attuatore")
    parser.add_argument("--no-actuator-limit", action="store_false", dest="actuator_limit",
                        help="Disabilita il rate limiter fisico dell'attuatore")
    parser.add_argument("--v-max", type=float, default=None, dest="v_max",
                        help=f"Velocità massima attuatore [m/s] (default={V_MAX_ATTUATORE})")
    parser.add_argument("--agent-path", type=str, default=None,
                        help="Percorso diretto al file agent_actor.pth (ignora RUN_FOLDER)")

    args = parser.parse_args()

    if args.folder is not None:
        folder = args.folder
        agent_path = None
    elif args.agent_path is not None:
        folder = os.path.dirname(args.agent_path)
        agent_path = args.agent_path
    elif RUN_FOLDER is not None and AGENT_PATH is not None:
        if USE_RUN_FOLDER:
            folder = RUN_FOLDER
            agent_path = None
        else:
            folder = os.path.dirname(AGENT_PATH)
            agent_path = AGENT_PATH
    elif RUN_FOLDER is not None:
        folder = RUN_FOLDER
        agent_path = None
    elif AGENT_PATH is not None:
        folder = os.path.dirname(AGENT_PATH)
        agent_path = AGENT_PATH
    else:
        print("Nessuna run specificata.")
        sys.exit(1)

    if args.actuator_limit is not None:
        actuator_limit_val = args.actuator_limit
    else:
        actuator_limit_val = None

    run_animation(
        folder,
        agent_path=agent_path,
        speed=args.speed,
        save_gif=args.gif,
        vis_mode=args.mode,
        t_generic=args.t_generic,
        t_hold=args.t_hold,
        msd_x_min=args.xmin,
        msd_x_max=args.xmax,
        pos_y_min=args.ymin,
        pos_y_max=args.ymax,
        target=args.target,
        display_fps=args.fps,
        actuator_physical_limit=actuator_limit_val,
        v_max_attuatore=args.v_max,
    )