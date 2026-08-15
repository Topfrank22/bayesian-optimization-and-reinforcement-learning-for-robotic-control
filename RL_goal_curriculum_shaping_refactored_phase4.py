import os
import csv
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

import matplotlib
import numexpr as ne

matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗
# ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║
# ██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   ██████╔╝██║
# ██╔═══╝ ██╔══██╗██╔══██╗██╔══██╗██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║
# ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║
# ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝
#
#  TUTTI I PARAMETRI MODIFICABILI SONO QUI — divisi per sezione
#  Il resto del codice NON va toccato (salvo bug fix).
# ============================================================


# ------------------------------------------------------------
# [1] SISTEMA FISICO
# ------------------------------------------------------------
PARAMETRI_SISTEMA = [2.5, 70.0, 1500, 4000, 0.30, 0.25]

U_MIN  = -1.0
U_MAX  =  1.0
DT     =  0.01
T_MAX  = 5.0

TARGET_THRESHOLD = 0.05   # [m] distanza sotto cui il target è considerato raggiunto
VEL_THRESHOLD    = 10.0   # [m/s] velocità sotto cui il target è considerato stabilizzato

# ------------------------------------------------------------
# [2] REWARD
# ------------------------------------------------------------
C_TIME        = 0.05    # penalità temporale per step
USE_SHAPING   = True    # attiva reward di shaping (usato come default globale)
W_SHAPING     = 1.0     # peso della reward di shaping
R_GOAL        = 200.0   # bonus fisso al raggiungimento del target
R_VEL_MAX     = 225.0    # bonus massimo per velocità nulla al target
VEL_SIGMA     = 8     # tasso di decrescita del bonus velocità
LAMBDA_REG    = 0.005     # peso regolarizzazione azione (0 = disabilitato)

# Reward shaping configurabile
# ──────────────────────────────────────────────────────────────────────────────
# SHAPING_FORMULA (str | None)
#   Se impostato, la reward di shaping viene calcolata valutando questa formula
#   con numexpr. Se None, si usa il comportamento legacy (alpha_energy/beta_velocity).
#
#   Variabili disponibili nella formula:
#     Posizioni  : x1a (xb corrente),  x1b (xb precedente)
#                  x2a (xr corrente),  x2b (xr precedente)
#     Velocità   : v1a (xb_dot corrente),  v1b (xb_dot precedente)
#                  v2a (xr_dot corrente),  v2b (xr_dot precedente)
#     Parametri  : kb, kr, mb, mr, xb0
#     Azione     : u   (azione reale corrente)
#     Target     : target
#
#   Esempi:
#     Formula energia potenziale (equivalente al legacy delta_v):
#       "(x1b**2*kb*0.5 - x1a**2*kb*0.5) / ((x1b**2*kb*0.5 - x1a**2*kb*0.5) + 1)"
#
#     Formula con prodotto delle velocità:
#       "v1a * v2a"
#
#     Formula composita:
#       "((x1b**2*kb*0.5 - x1a**2*kb*0.5) / ((abs(x1b**2*kb*0.5 - x1a**2*kb*0.5)) + 1)) + 5.0 * (v1a * v2a) / (abs(v1a * v2a) + 8)"
#
#   NOTA: numexpr NON supporta math.tanh; usa where(...) o espressioni algebriche.
#         Per tanh approssimato: usa la forma razionale come sopra.
# ──────────────────────────────────────────────────────────────────────────────
SHAPING_FORMULA = "(1.0*(x1a**2*kb*0.5 - x1b**2*kb*0.5 + x2a**2*kr*0.5 - x2b**2*kr*0.5)  " \
    " + 1.0*(v1a**2*mb*0.5 - v1b**2*mb*0.5 + v2a**2*mr*0.5 - v2b**2*mr*0.5)) " \
    "/ 100"   # <-- imposta la tua formula qui (stringa) oppure None per legacy


SHAPING_CONFIG = {
    "shaping_type": "rational",  # opzioni: "linear", "tanh", "rational"  [solo in modalità legacy]
    "alpha_energy": 1.0,
    "beta_velocity": 5.0,
    "tanh_gain_energy": 1.5,
    "tanh_gain_velocity": 0.5,
    "rational_constant_energy": 1500,
    "rational_constant_velocity": 8,
}
# ------------------------------------------------------------
# [3] ARCHITETTURA RETI NEURALI
# ------------------------------------------------------------
ACTOR_HIDDEN_SIZE    = 64
ACTOR_HIDDEN_LAYERS  = 2

CRITIC_HIDDEN_SIZE   = 64
CRITIC_HIDDEN_LAYERS = 2

ACTOR_HIDDEN_ACT  = "elu"
ACTOR_OUTPUT_ACT  = "tanh"
CRITIC_HIDDEN_ACT = "elu"
CRITIC_OUTPUT_ACT = "none"

ACTOR_HIDDEN_INIT  = "he"
CRITIC_HIDDEN_INIT = "he"

ACTOR_OUTPUT_INIT  = "tiny_uniform"
CRITIC_OUTPUT_INIT = "default"

#    "relu":       nn.ReLU,
#    "tanh":       nn.Tanh,
#    "elu":        nn.ELU,
#    "leaky_relu": nn.LeakyReLU,
#    "none":       None,

# STATE_SCALE per stato a 5 elementi: [xb, xb_dot, xr, xr_dot, target]
STATE_SCALE = [0.15, 1.5, 2.0, 40.0, 2.0]

REWARD_SCALE = 50.0

# ------------------------------------------------------------
# [4] TRAINING DDPG
# ------------------------------------------------------------
LR_ACTOR        = 1e-3 #1e-4 
LR_CRITIC       = 5e-3 #1e-3 
GAMMA           = 0.99
TAU             = 0.003
CLIP_GRAD_NORM  = 1.0

BATCH_SIZE      = 128
BUFFER_CAPACITY = 50000
WARMUP_STEPS    = 5000


# Unica posizione target usata nel training e nei test
TARGET_POSITION = 1.40   # [m]

# --- Curriculum learning a 3 fasi basato sui pesi della reward ---
CURRICULUM_PHASES = [
    {
        "name": "shaping_only",
        "episodes": 80,
        "use_shaping": True,
        "w_shaping": 1.0,
        "final_reward_scale": 0.0,
        "terminate_on_target": False,   # fase 1: non terminare al target, impara lo shaping
        "time_penalty_scale": 0.0,
        "min_phase_episodes": 80,          # episodi minimi prima di poter avanzare
        "success_rate_threshold": 0.75,     # soglia SR per avanzare (override globale)
        "phase_eval_window": 50,            # finestra episodi per valutare SR (override globale)
        "target_mode": "fixed",
        "target_value": TARGET_POSITION,
    },
    {
        "name": "mixed",
        "episodes": 300,
        "use_shaping": True,
        "w_shaping": 0.5,
        "final_reward_scale": 0.75,
        "terminate_on_target": True,
        "time_penalty_scale": 0.5,
        "min_phase_episodes": 200,
        "success_rate_threshold": 0.95,
        "phase_eval_window": 50,
        "target_mode": "random_range",      #random_range or fixed
        "target_value": TARGET_POSITION,
        "target_min": 1.5,
        "target_max": 2.0,
    },
    {
        "name": "final_only",
        "episodes": 400,
        "use_shaping": False,
        "w_shaping": 0.0,
        "final_reward_scale": 1.0,
        "terminate_on_target": True,
        "time_penalty_scale": 1.0,
        "min_phase_episodes": 400,
        "success_rate_threshold": 0.95,
        "phase_eval_window": 50,
        "target_mode": "random_range",
        "target_value": TARGET_POSITION,
        "target_min": 1.5,
        "target_max": 2.0,
    },
    {
        "name": "fine_tuning",
        "episodes": 800,
        "use_shaping": False,
        "w_shaping": 0.0,
        "final_reward_scale": 1.0,
        "terminate_on_target": True,
        "time_penalty_scale": 1.0,
        "min_phase_episodes": 800,
        "success_rate_threshold": 0.95,
        "phase_eval_window": 50,
        "target_mode": "random_range",
        "target_value": TARGET_POSITION,
        "target_min": 1.5,
        "target_max": 2.0,
    },
]

PHASE_EVAL_WINDOW      = 50
SUCCESS_RATE_THRESHOLD = 0.95

NOISE_RESET_PER_PHASE = [1.0, 0.6, 0.8, 0.5]
NOISE_DECAY_PER_PHASE  = [0.989, 0.991, 0.998, 0.997]

# Rescue noise parametrico (compatibile con il curriculum a 3 fasi)
NOISE_RESCUE_ENABLE          = True
NOISE_RESCUE_TRIGGER_SIGMA   = 0.20
NOISE_RESCUE_TRIGGER_SR      = 0.20
NOISE_RESCUE_WINDOW          = 50
NOISE_RESCUE_SR_LOW          = 0.10
NOISE_RESCUE_SR_VERY_LOW     = 0.05
NOISE_RESCUE_VALUE_LOW       = 0.60
NOISE_RESCUE_VALUE_VERY_LOW  = 0.80
NOISE_RESCUE_HOLD_EPISODES   = 75

# ------------------------------------------------------------
# [5] ESPLORAZIONE
# ------------------------------------------------------------
# NOISE_TYPE : tipo di rumore aggiunto all'azione dell'actor.
#   "gaussian"  -> rumore i.i.d. N(0, sigma) per step
#   "ou"        -> Ornstein-Uhlenbeck
#   "coloured"  -> Coloured Noise (1/f^beta) generato a inizio episodio
NOISE_TYPE = os.environ.get("RL_NOISE_TYPE", "ou").lower()   # "gaussian" | "ou" | "coloured"

OU_THETA = 4
OU_MU    = 0.0
OU_SIGMA = 1.2
COLOURED_BETA = 1.3

NOISE_START  = 1.0
NOISE_MIN    = 0.05
NOISE_DECAY  = 0.993

# ------------------------------------------------------------
# [6] LOGGING E OUTPUT
# ------------------------------------------------------------
N_BEST_RUNS            = 5
N_TEST_RUNS            = 5      # episodi deterministici per valutare ogni candidato nella selezione finale
LOG_EVERY              = 10
SEED                   = 42
SUCCESS_THRESHOLD_DIST = TARGET_THRESHOLD   # alias per chiarezza nei log
DASHBOARD_UPDATE_EVERY = 50
SNAPSHOT_EVERY         = 500
CSV_FIELDNAMES = [
    "episode", "phase", "target",
    "reward_total", "r_time", "r_shaping", "r_goal", "r_vel", "r_reg",
    "success", "time_to_target", "episode_length",
    "final_dist", "final_velocity", "peak_elongation", "peak_velocity",
    "mean_action", "std_action", "mean_abs_delta_u",
    "target_crossings", "overshoot", "peak_mechanical_energy",
    "sigma", "critic_loss", "mean_q", "actor_loss",
]

# ============================================================
# [7] LIMITI FISICI ATTUATORE
# ============================================================
ACTUATOR_PHYSICAL_LIMIT = False 
V_MAX_ATTUATORE = 13.33


# ============================================================
# NORMALIZZAZIONE STATI (stato a 5 elementi)
# ============================================================
_STATE_SCALE_NP = np.array(STATE_SCALE, dtype=np.float32)

def normalize_state(s):
    """Divide lo stato grezzo (5 elem) per STATE_SCALE → valori approx in [-1, +1]."""
    return s / _STATE_SCALE_NP


def potential_energy_total(xb, xr, u, kb, kr, xb0=0.0):
    """Energia potenziale totale delle due molle."""
    return 0.5 * kb * (xb - xb0) ** 2 + 0.5 * kr * (xr - u) ** 2


def _apply_single_shaping(value, shaping_type, tanh_gain=None, rational_constant=None):
    """Normalizzazione scalare per la modalità legacy (senza SHAPING_FORMULA)."""
    z = float(value)
    st = str(shaping_type).lower()
    if st == "linear":
        return z
    if st == "tanh":
        gain = 1.0 if tanh_gain is None else float(tanh_gain)
        return math.tanh(gain * z)
    if st == "rational":
        c = 1.0 if rational_constant is None else float(rational_constant)
        if c <= 0:
            raise ValueError("La costante rational deve essere > 0")
        return z / (c + abs(z))
    raise ValueError(f"Shaping type '{shaping_type}' non riconosciuto")


# Cache della formula compilata da numexpr (evita il parsing ad ogni step)
_shaping_formula_cache: dict = {"expr": None, "compiled": None}


def _compile_shaping_formula(formula: str):
    """Compila e cachea la formula di shaping con numexpr."""
    if _shaping_formula_cache["expr"] != formula:
        # numexpr.evaluate() compila internamente; la cache evita solo
        # il costo del re-parsing Python della stringa.
        _shaping_formula_cache["expr"] = formula
        _shaping_formula_cache["compiled"] = formula   # ne.evaluate accetta stringa direttamente
    return _shaping_formula_cache["compiled"]


def apply_reward_shaping(delta_v, velocity_term, shaping_config=None,
                          shaping_context: dict = None):
    """
    Calcola la reward di shaping.

    Modalità FORMULA (SHAPING_FORMULA non None):
        Valuta SHAPING_FORMULA con numexpr usando il dizionario shaping_context
        che deve contenere le variabili simboliche descritte in SHAPING_CONFIG.
        I parametri delta_v e velocity_term vengono ignorati (sono già
        codificati nella formula dall'utente).

    Modalità LEGACY (SHAPING_FORMULA is None):
        Usa alpha_energy * phi_energy + beta_velocity * phi_velocity
        come prima, con delta_v e velocity_term come input.

    Args:
        delta_v         : variazione energia potenziale (modalità legacy)
        velocity_term   : termine di velocità (modalità legacy)
        shaping_config  : override del dizionario SHAPING_CONFIG
        shaping_context : dizionario con le variabili simboliche per numexpr
                          (obbligatorio se SHAPING_FORMULA non è None)

    Returns:
        float : valore scalare della reward di shaping
    """
    if SHAPING_FORMULA is not None:
        if shaping_context is None:
            raise ValueError(
                "SHAPING_FORMULA è impostato ma shaping_context=None. "
                "Passa il dizionario con le variabili simboliche."
            )
        _compile_shaping_formula(SHAPING_FORMULA)
        try:
            result = ne.evaluate(SHAPING_FORMULA, local_dict=shaping_context)
            return float(result)
        except Exception as exc:
            raise RuntimeError(
                f"Errore nella valutazione di SHAPING_FORMULA: {exc}\n"
                f"Formula: {SHAPING_FORMULA}\n"
                f"Variabili disponibili: {list(shaping_context.keys())}"
            ) from exc

    # ── Modalità legacy ──────────────────────────────────────────────────────
    cfg = SHAPING_CONFIG if shaping_config is None else shaping_config
    shaping_type  = str(cfg.get("shaping_type", cfg.get("type", "linear"))).lower()
    alpha_energy  = float(cfg.get("alpha_energy", 1.0))
    beta_velocity = float(cfg.get("beta_velocity", 1.0))

    phi_energy = _apply_single_shaping(
        delta_v,
        shaping_type=shaping_type,
        tanh_gain=cfg.get("tanh_gain_energy", 1.0),
        rational_constant=cfg.get("rational_constant_energy", 1.0),
    )
    phi_velocity = _apply_single_shaping(
        velocity_term,
        shaping_type=shaping_type,
        tanh_gain=cfg.get("tanh_gain_velocity", 1.0),
        rational_constant=cfg.get("rational_constant_velocity", 1.0),
    )
    return alpha_energy * phi_energy + beta_velocity * phi_velocity


def _build_shaping_context(x1a, v1a, x2a, v2a,
                            x1b, v1b, x2b, v2b,
                            kb, kr, mb, mr, xb0, u, target) -> dict:
    """
    Costruisce il dizionario delle variabili simboliche per numexpr.

    Convenzione nomi:
        x1a / x2a  : posizione massa 1 (xb) e massa 2 (xr) allo stato CORRENTE  (a = after)
        x1b / x2b  : posizione massa 1 (xb) e massa 2 (xr) allo stato PRECEDENTE (b = before)
        v1a / v2a  : velocità massa 1 (xb_dot) e massa 2 (xr_dot) allo stato CORRENTE
        v1b / v2b  : velocità massa 1 (xb_dot) e massa 2 (xr_dot) allo stato PRECEDENTE
        kb, kr     : costanti delle molle
        mb, mr     : masse
        xb0        : posizione di riposo della molla della base
        u          : azione reale corrente (setpoint robot)
        target     : posizione target dell'episodio
    """
    return {
        "x1a": float(x1a), "v1a": float(v1a),
        "x2a": float(x2a), "v2a": float(v2a),
        "x1b": float(x1b), "v1b": float(v1b),
        "x2b": float(x2b), "v2b": float(v2b),
        "kb":  float(kb),  "kr":  float(kr),
        "mb":  float(mb),  "mr":  float(mr),
        "xb0": float(xb0), "u":   float(u),
        "target": float(target),
    }


def get_phase_target(phase_cfg):
    if phase_cfg.get("target_mode", "fixed") == "random_range":
        return random.uniform(phase_cfg["target_min"], phase_cfg["target_max"])
    return phase_cfg.get("target_value", TARGET_POSITION)


# ============================================================
# REWARD FUNCTION (goal-conditioned)
# ============================================================
def velocity_bonus(ydot, R_VEL_MAX=200.0, VEL_SIGMA=8.0):
    return R_VEL_MAX * np.exp(-(ydot / VEL_SIGMA) ** 2)

def compute_reward(xb, xb_dot, xr, xr_dot, target, u, delta_u_norm,
                   delta_v_potential, velocity_term, terminated, y_dot,
                   use_shaping_override=None, w_shaping_override=None,
                   final_reward_scale_override=None,
                   time_penalty_scale_override=None,
                   instability=False,
                   shaping_context: dict = None):
    """Reward goal-conditioned con shaping fisico su energia potenziale e velocità."""
    use_shaping = use_shaping_override if use_shaping_override is not None else USE_SHAPING
    w_shaping = w_shaping_override if w_shaping_override is not None else W_SHAPING

    time_penalty_scale = (
        time_penalty_scale_override
        if time_penalty_scale_override is not None
        else 1.0
    )
    r_time = -C_TIME * time_penalty_scale

    shaping_value = apply_reward_shaping(delta_v_potential, velocity_term, shaping_context=shaping_context) if use_shaping else 0.0
    r_shaping = w_shaping * shaping_value if use_shaping else 0.0

    final_reward_scale = (
        final_reward_scale_override
        if final_reward_scale_override is not None
        else 1.0
    )

    r_final = 0.0
    if terminated and not instability:
        r_goal = R_GOAL
        r_vel = velocity_bonus(y_dot, R_VEL_MAX=R_VEL_MAX, VEL_SIGMA=VEL_SIGMA)
        r_final = final_reward_scale * (r_goal + r_vel)

    r_reg = -LAMBDA_REG * (delta_u_norm ** 2) if LAMBDA_REG > 0 else 0.0

    return r_time + r_shaping + r_final + r_reg


# ============================================================
# 1. AMBIENTE (goal-conditioned)
# ============================================================
class GoalLinearSystemEnv:
    def __init__(self):
        self.u_min = U_MIN
        self.u_max = U_MAX
        self.dt    = DT
        self.T_max = T_MAX

        self.Mr, self.Mb, self.Kr, self.Kb, self.hr, self.hb = PARAMETRI_SISTEMA
        self.Dr  = 2 * self.hr * np.sqrt(self.Kr * self.Mr)
        self.Db  = 2 * self.hb * np.sqrt(self.Kb * self.Mb)
        self.xb0 = 0.0

        self.M_mat = np.array([
            [self.Mr + self.Mb, self.Mr],
            [self.Mr,           self.Mr],
        ], dtype=float)

        self.x          = np.zeros(4)
        self.target     = 0.0
        self.d_best     = 0.0
        self.step_count = 0
        self.u_prev     = 0.0

    def _rhs(self, z, xr0_val):
        xb, xb_dot, xr, xr_dot = z
        rhs = np.array([
            -self.Db * xb_dot - self.Kb * (xb - self.xb0),
            -self.Dr * xr_dot - self.Kr * (xr - xr0_val),
        ], dtype=float)
        xb_ddot, xr_ddot = np.linalg.solve(self.M_mat, rhs)
        return np.array([xb_dot, xb_ddot, xr_dot, xr_ddot], dtype=float)

    def _rk4_step(self, z, xr0_val):
        dt = self.dt
        k1 = self._rhs(z, xr0_val)
        k2 = self._rhs(z + 0.5 * dt * k1, xr0_val)
        k3 = self._rhs(z + 0.5 * dt * k2, xr0_val)
        k4 = self._rhs(z + dt * k3, xr0_val)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def reset(self, target_val):
        """
        Inizializza l'episodio con il target specificato.

        Args:
            target_val (float): posizione assoluta obiettivo [m]
        Returns:
            stato normalizzato (5 elementi)
        """
        self.x          = np.zeros(4)
        self.u_prev     = 0.0
        self.target     = float(target_val)
        self.d_best     = abs(self.target - 0.0)   # distanza iniziale
        self.step_count = 0
        state_raw = np.array([0.0, 0.0, 0.0, 0.0, self.target], dtype=np.float32)
        return normalize_state(state_raw)

    def step(self, u, use_shaping_override=None, w_shaping_override=None,
             final_reward_scale_override=None, time_penalty_scale_override=None,
             terminate_on_target=True):
        """
        Esegue un passo di simulazione.

        Args:
            u : azione (setpoint molla robot)
            use_shaping_override : override opzionale per compute_reward
            w_shaping_override   : override opzionale per compute_reward
            final_reward_scale_override : scala la reward finale
        Returns:
            (next_state_norm, reward, terminated, done)
        """
        u_target = np.clip(u, self.u_min, self.u_max)

        if ACTUATOR_PHYSICAL_LIMIT:
            max_delta_u  = V_MAX_ATTUATORE * self.dt
            delta_u      = np.clip(u_target - self.u_prev, -max_delta_u, max_delta_u)
            u_real       = self.u_prev + delta_u
            delta_u_norm = delta_u / (self.u_max - self.u_min)
        else:
            u_real       = u_target
            delta_u_norm = 0.0

        self.u_prev  = u_real
        self.step_count += 1

        x_prev = self.x.copy()
        xb_prev, xb_dot_prev, xr_prev, xr_dot_prev = x_prev
        v_potential_prev = potential_energy_total(
            xb_prev, xr_prev, u_real, self.Kb, self.Kr, self.xb0
        )

        self.x = self._rk4_step(self.x, float(u_real))
        xb, xb_dot, xr, xr_dot = self.x

        v_potential_next = potential_energy_total(
            xb, xr, u_real, self.Kb, self.Kr, self.xb0
        )
        delta_v_potential = v_potential_prev - v_potential_next
        velocity_term = xb_dot * xr_dot

        # Contesto simbolico per SHAPING_FORMULA (numexpr)
        _shaping_context = _build_shaping_context(
            x1a=xb,          v1a=xb_dot,
            x2a=xr,          v2a=xr_dot,
            x1b=xb_prev,     v1b=xb_dot_prev,
            x2b=xr_prev,     v2b=xr_dot_prev,
            kb=self.Kb,      kr=self.Kr,
            mb=self.Mb,      mr=self.Mr,
            xb0=self.xb0,    u=u_real,
            target=self.target,
        )

        y = xb + xr
        y_dot = xb_dot + xr_dot
        dist = abs(self.target - y)

        if dist < self.d_best:
            self.d_best = dist

        instability = False
        target_reached = (dist <= TARGET_THRESHOLD) and (abs(y_dot) <= VEL_THRESHOLD)
        terminated  = target_reached if terminate_on_target else False
        done        = self.step_count >= int(self.T_max / self.dt)

        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            terminated  = True
            instability = True

        reward = compute_reward(
            xb, xb_dot, xr, xr_dot,
            self.target, u_real, delta_u_norm, delta_v_potential, velocity_term,
            terminated, y_dot,
            use_shaping_override=use_shaping_override,
            w_shaping_override=w_shaping_override,
            final_reward_scale_override=final_reward_scale_override,
            time_penalty_scale_override=time_penalty_scale_override,
            instability=instability,
            shaping_context=_shaping_context,
        )

        state_raw = np.array([xb, xb_dot, xr, xr_dot, self.target], dtype=np.float32)
        return normalize_state(state_raw), reward, terminated, done


# ============================================================
# 2. REPLAY BUFFER
# ============================================================
class ReplayBuffer:
    def __init__(self):
        self._states      = np.zeros((BUFFER_CAPACITY, 5), dtype=np.float32)
        self._actions     = np.zeros((BUFFER_CAPACITY,), dtype=np.float32)
        self._rewards     = np.zeros((BUFFER_CAPACITY, 1), dtype=np.float32)
        self._next_states = np.zeros((BUFFER_CAPACITY, 5), dtype=np.float32)
        self._dones       = np.zeros((BUFFER_CAPACITY, 1), dtype=np.float32)
        self._ptr         = 0
        self._size        = 0

    def push(self, state, action, reward, next_state, terminated):
        self._states[self._ptr]      = state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr, 0]   = reward / REWARD_SCALE
        self._next_states[self._ptr]  = next_state
        self._dones[self._ptr, 0]     = float(terminated)
        self._ptr  = (self._ptr + 1) % BUFFER_CAPACITY
        self._size = min(self._size + 1, BUFFER_CAPACITY)

    def sample(self):
        idx = np.random.randint(0, self._size, size=BATCH_SIZE)
        return (
            torch.from_numpy(self._states[idx]).to(device),
            torch.from_numpy(self._actions[idx]).unsqueeze(1).to(device),
            torch.from_numpy(self._rewards[idx]).to(device),
            torch.from_numpy(self._next_states[idx]).to(device),
            torch.from_numpy(self._dones[idx]).to(device),
        )

    def __len__(self):
        return self._size


class OUNoise:
    def __init__(self, action_dim):
        self.mu    = OU_MU * np.ones(action_dim)
        self.theta = OU_THETA
        self.sigma = OU_SIGMA
        self.state = np.zeros(action_dim)

    def reset(self):
        self.state = np.zeros(len(self.mu))

    def sample(self):
        dx = self.theta * DT * (self.mu - self.state) \
             + self.sigma * np.sqrt(DT) * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state.copy()


class ColouredNoise:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self._seq = None
        self._idx = 0

    def reset(self):
        n_steps = int(T_MAX / DT) + 2
        self._seq = self._generate_coloured(n_steps, self.action_dim, COLOURED_BETA)
        self._idx = 0

    @staticmethod
    def _generate_coloured(n_steps, action_dim, beta):
        seq = np.zeros((n_steps, action_dim), dtype=np.float32)
        for d in range(action_dim):
            f = np.fft.rfft(np.random.randn(n_steps))
            freqs = np.fft.rfftfreq(n_steps)
            freqs[0] = 1.0
            power_filter = np.power(freqs, beta / 2.0)
            f_filtered = f / power_filter
            s = np.fft.irfft(f_filtered, n=n_steps)
            std = s.std()
            if std > 1e-8:
                s /= std
            seq[:, d] = s.astype(np.float32)
        return seq

    def sample(self, sigma):
        if self._seq is None or self._idx >= len(self._seq):
            self.reset()
        noise = self._seq[self._idx] * sigma
        self._idx += 1
        return noise.copy()


def _build_noise_engine(action_dim):
    if NOISE_TYPE == "ou":
        return OUNoise(action_dim)
    elif NOISE_TYPE == "coloured":
        return ColouredNoise(action_dim)
    else:
        return None


def _check_rescue_noise(noise_current, success_window):
    if not NOISE_RESCUE_ENABLE:
        return None, False
    if noise_current >= NOISE_RESCUE_TRIGGER_SIGMA:
        return None, False
    if len(success_window) != NOISE_RESCUE_WINDOW:
        return None, False

    sr_rescue = float(np.mean(success_window))
    if sr_rescue >= NOISE_RESCUE_TRIGGER_SR:
        return None, False
    if sr_rescue < NOISE_RESCUE_SR_VERY_LOW:
        return NOISE_RESCUE_VALUE_VERY_LOW, True
    if sr_rescue < NOISE_RESCUE_SR_LOW:
        return NOISE_RESCUE_VALUE_LOW, True
    return None, False



# ============================================================
# 3. RETI NEURALI
# ============================================================
_ACTIVATION_MAP = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "elu":        nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "none":       None,
}

def _get_activation(name):
    cls = _ACTIVATION_MAP.get(name)
    if cls is None and name != "none":
        raise ValueError(f"Attivazione '{name}' non riconosciuta. "
                         f"Valori ammessi: {list(_ACTIVATION_MAP.keys())}")
    return cls() if cls is not None else None


def _apply_weight_init(module, weight_init):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if weight_init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif weight_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _apply_output_init(linear_layer, output_init):
    if output_init == "tiny_uniform":
        nn.init.uniform_(linear_layer.weight, -3e-3, 3e-3)
        if linear_layer.bias is not None:
            nn.init.uniform_(linear_layer.bias, -3e-3, 3e-3)
    elif output_init == "he":
        nn.init.kaiming_normal_(linear_layer.weight, nonlinearity="relu")
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)
    elif output_init == "xavier":
        nn.init.xavier_uniform_(linear_layer.weight)
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)


def _build_mlp(input_dim, output_dim, hidden_size, hidden_layers,
               hidden_activation="relu", output_activation=None):
    layers = []
    in_dim = input_dim
    hidden_act = _get_activation(hidden_activation)
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        if hidden_act is not None:
            layers.append(type(hidden_act)())
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        out_act = _get_activation(output_activation)
        if out_act is not None:
            layers.append(out_act)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.register_buffer('u_min_t', torch.tensor(U_MIN, dtype=torch.float32))
        self.register_buffer('u_max_t', torch.tensor(U_MAX, dtype=torch.float32))
        self.network = _build_mlp(
            state_dim, action_dim,
            ACTOR_HIDDEN_SIZE, ACTOR_HIDDEN_LAYERS,
            hidden_activation=ACTOR_HIDDEN_ACT,
            output_activation=ACTOR_OUTPUT_ACT,
        )
        _apply_weight_init(self, ACTOR_HIDDEN_INIT)
        out_linear_idx = -2 if ACTOR_OUTPUT_ACT != "none" else -1
        _apply_output_init(self.network[out_linear_idx], ACTOR_OUTPUT_INIT)

    def forward(self, state):
        tanh_out = self.network(state)
        scaled   = 0.5 * (U_MAX - U_MIN) * tanh_out + 0.5 * (U_MAX + U_MIN)
        return torch.clamp(scaled, self.u_min_t, self.u_max_t)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = _build_mlp(
            state_dim + action_dim, 1,
            CRITIC_HIDDEN_SIZE, CRITIC_HIDDEN_LAYERS,
            hidden_activation=CRITIC_HIDDEN_ACT,
            output_activation=CRITIC_OUTPUT_ACT,
        )
        _apply_weight_init(self, CRITIC_HIDDEN_INIT)
        out_linear_idx = -2 if CRITIC_OUTPUT_ACT != "none" else -1
        _apply_output_init(self.network[out_linear_idx], CRITIC_OUTPUT_INIT)

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


# ============================================================
# 4. AGENTE DDPG
# ============================================================
class DDPGAgent:
    def __init__(self, state_dim=5, action_dim=1):
        self.actor         = Actor(state_dim, action_dim).to(device)
        self.critic        = Critic(state_dim, action_dim).to(device)
        self.actor_target  = Actor(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self._noise_engine = _build_noise_engine(action_dim)

        self._ep_critic_loss = []
        self._ep_q_mean      = []
        self._ep_actor_loss  = []

    def reset_noise(self):
        if self._noise_engine is not None:
            self._noise_engine.reset()

    def reset_episode_accum(self):
        self._ep_critic_loss = []
        self._ep_q_mean      = []
        self._ep_actor_loss  = []

    def get_episode_learning_metrics(self):
        cl = float(np.mean(self._ep_critic_loss)) if self._ep_critic_loss else float('nan')
        qm = float(np.mean(self._ep_q_mean))      if self._ep_q_mean      else float('nan')
        al = float(np.mean(self._ep_actor_loss))  if self._ep_actor_loss  else float('nan')
        return dict(critic_loss=cl, mean_q=qm, actor_loss=al)

    def select_action(self, state, noise_std=0.1):
        state_t = torch.tensor(state, dtype=torch.float32).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()
        self.actor.train()
        if noise_std > 0:
            if self._noise_engine is not None:
                if NOISE_TYPE == "ou":
                    action += noise_std * self._noise_engine.sample()
                else:
                    action += self._noise_engine.sample(noise_std)
            else:
                action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, U_MIN, U_MAX)

    def train_step(self, replay_buffer):
        if len(replay_buffer) < WARMUP_STEPS:
            return None, None

        states, actions, rewards, next_states, terminateds = replay_buffer.sample()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q     = rewards + (1 - terminateds) * GAMMA * self.critic_target(next_states, next_actions)

        q_pred      = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_pred, target_q)
        q_mean      = q_pred.mean().item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=CLIP_GRAD_NORM)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=CLIP_GRAD_NORM)
        self.actor_optimizer.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.lerp_(p.data, TAU)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.lerp_(p.data, TAU)

        cl = critic_loss.item()
        al = actor_loss.item()
        self._ep_critic_loss.append(cl)
        self._ep_q_mean.append(q_mean)
        self._ep_actor_loss.append(al)

        return cl, q_mean

    def save(self, folder, suffix=""):
        torch.save(self.actor.state_dict(),  os.path.join(folder, f"agent{suffix}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(folder, f"agent{suffix}_critic.pth"))
        print(f"  Agente salvato in: {folder}")


# ============================================================
# 5. LOGGING, DASHBOARD E SNAPSHOT
# ============================================================
def init_csv_logger(folder):
    path = os.path.join(folder, "training_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
    return path


def log_episode_csv(csv_path, row_dict):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow({k: row_dict.get(k, float('nan')) for k in CSV_FIELDNAMES})


def save_snapshot(agent, env, target_val, episode_global, snapshots_folder):
    max_steps = int(T_MAX / DT)
    state = env.reset(target_val)
    hist = {"t": [], "y": [], "target": [], "u": [], "sigma": []}
    t = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state, noise_std=0.0)
        raw = state * _STATE_SCALE_NP
        hist["t"].append(t)
        hist["y"].append(float(raw[0] + raw[2]))
        hist["target"].append(float(env.target))
        hist["u"].append(float(action[0]))
        hist["sigma"].append(0.0)
        state, _, terminated, done = env.step(action[0])
        t += DT
        if done or terminated:
            break
    path = os.path.join(snapshots_folder, f"snapshot_ep{episode_global:06d}.json")
    with open(path, "w") as f:
        json.dump(hist, f)


def _adaptive_noise_boost(sigma_current, sr):
    if sr > 0.20:
        return sigma_current, False
    if 0.10 <= sr <= 0.20:
        return 0.40, True
    if 0.05 <= sr < 0.10:
        return 0.60, True
    return 0.80, True


def _build_episode_metrics(episode_global, phase_idx, phase_name, target_val,
                           episode_reward, reward_components_accum,
                           success, ep_terminated,
                           actions_history, delta_u_history,
                           y_history, ydot_history,
                           t_history,
                           noise_current,
                           learning_metrics):
    metrics = {
        "episode": episode_global,
        "phase": phase_idx + 1,
        "target": target_val,
        "reward_total": episode_reward,
        "success": success,
        "episode_length": len(t_history),
        "sigma": noise_current,
    }
    metrics.update({k: float(np.sum([c.get(k, 0.0) for c in reward_components_accum])) for k in ("r_time", "r_shaping", "r_goal", "r_vel", "r_reg")})
    metrics["time_to_target"] = float(t_history[-1]) if (success and ep_terminated and t_history) else float('nan')
    metrics["final_dist"] = abs(target_val - y_history[-1]) if y_history else float('nan')
    metrics["final_velocity"] = float(ydot_history[-1]) if ydot_history else float('nan')
    metrics["peak_elongation"] = float(np.max(np.abs(y_history))) if y_history else float('nan')
    metrics["peak_velocity"] = float(np.max(np.abs(ydot_history))) if ydot_history else float('nan')
    if actions_history:
        arr_a = np.array(actions_history, dtype=float)
        metrics["mean_action"] = float(np.mean(arr_a))
        metrics["std_action"] = float(np.std(arr_a))
    else:
        metrics["mean_action"] = float('nan')
        metrics["std_action"] = float('nan')
    metrics["mean_abs_delta_u"] = float(np.mean(np.abs(delta_u_history))) if delta_u_history else float('nan')
    if len(y_history) >= 2:
        diffs = np.array(y_history) - target_val
        signs = np.sign(diffs)
        metrics["target_crossings"] = int(np.sum(np.abs(np.diff(signs)) > 0))
    else:
        metrics["target_crossings"] = 0
    metrics["overshoot"] = float(max(0.0, np.max(np.array(y_history)) - target_val)) if y_history else float('nan')
    Mr = PARAMETRI_SISTEMA[0]
    Kr = PARAMETRI_SISTEMA[2]
    if y_history and ydot_history:
        n = min(len(y_history), len(ydot_history))
        T = 0.5 * Mr * np.array(ydot_history[:n]) ** 2
        V = 0.5 * Kr * np.array(y_history[:n]) ** 2
        metrics["peak_mechanical_energy"] = float(np.max(T + V))
    else:
        metrics["peak_mechanical_energy"] = float('nan')
    metrics["critic_loss"] = learning_metrics.get("critic_loss", float('nan'))
    metrics["mean_q"] = learning_metrics.get("mean_q", float('nan'))
    metrics["actor_loss"] = learning_metrics.get("actor_loss", float('nan'))
    metrics["phase_name"] = phase_name
    return metrics


def plot_training_dashboard(metrics_history, run_folder, phase_changes, window=50,
                             threshold=SUCCESS_RATE_THRESHOLD):
    if len(metrics_history) < 2:
        return

    def _field(key):
        return np.array([m.get(key, float('nan')) for m in metrics_history], dtype=float)

    def _ma(arr, w):
        if len(arr) >= w:
            return np.convolve(arr, np.ones(w) / w, mode='valid'), np.arange(w - 1, len(arr))
        return None, None

    episodes = np.arange(len(metrics_history))
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    raw_r = _field("reward_total")
    axes[0].plot(episodes, raw_r, alpha=0.25, color='steelblue')
    ma, idx = _ma(raw_r, window)
    if ma is not None:
        axes[0].plot(idx, ma, color='navy', linewidth=2, label=f'MA-{window}')
    axes[0].set_title('1. Reward'); axes[0].set_ylabel('Reward'); axes[0].grid(True, alpha=0.3); axes[0].legend()

    sr = _field("success")
    ma_sr, idx_sr = _ma(sr, window)
    if ma_sr is not None:
        axes[1].plot(idx_sr, ma_sr * 100, color='green', linewidth=2, label=f'SR MA-{window}')
    axes[1].axhline(threshold * 100, color='red', linestyle='--', linewidth=1.5, label=f'Threshold {threshold*100:.0f}%')
    for sc in phase_changes:
        axes[1].axvline(sc, color='orange', linestyle=':', alpha=0.7)
    axes[1].set_ylim(0, 105); axes[1].set_title('2. Success Rate'); axes[1].set_ylabel('SR [%]'); axes[1].grid(True, alpha=0.3); axes[1].legend()

    pe = _field("peak_elongation")
    axes[2].plot(episodes, pe, alpha=0.3, color='darkorange')
    ma_pe, idx_pe = _ma(pe, window)
    if ma_pe is not None:
        axes[2].plot(idx_pe, ma_pe, color='red', linewidth=2, label=f'MA-{window}')
    tgt = _field("target")
    axes[2].plot(episodes, tgt, color='gray', linestyle='--', alpha=0.5, label='target')
    axes[2].set_title('3. Peak Elongation'); axes[2].set_ylabel('[m]'); axes[2].grid(True, alpha=0.3); axes[2].legend()

    sigma = _field("sigma")
    axes[3].plot(episodes, sigma, color='purple', linewidth=1.5, label='sigma')
    for sc in phase_changes:
        axes[3].axvline(sc, color='orange', linestyle=':', alpha=0.8)
    axes[3].set_title('4. Sigma'); axes[3].set_ylabel('sigma'); axes[3].grid(True, alpha=0.3); axes[3].legend()

    cl = _field("critic_loss")
    valid = ~np.isnan(cl)
    if valid.any():
        axes[4].plot(episodes[valid], cl[valid], alpha=0.3, color='tomato')
        ma_cl, idx_cl = _ma(cl[valid], max(10, min(window, max(2, valid.sum() // 2))))
        if ma_cl is not None:
            axes[4].plot(episodes[valid][idx_cl], ma_cl, color='firebrick', linewidth=2, label='MA')
    axes[4].set_title('5. Critic Loss'); axes[4].set_ylabel('Loss'); axes[4].grid(True, alpha=0.3); axes[4].legend()

    mq = _field("mean_q")
    valid = ~np.isnan(mq)
    if valid.any():
        axes[5].plot(episodes[valid], mq[valid], alpha=0.3, color='mediumpurple')
        ma_mq, idx_mq = _ma(mq[valid], max(10, min(window, max(2, valid.sum() // 2))))
        if ma_mq is not None:
            axes[5].plot(episodes[valid][idx_mq], ma_mq, color='indigo', linewidth=2, label='MA')
    axes[5].set_title('6. Mean Q'); axes[5].set_ylabel('Q'); axes[5].grid(True, alpha=0.3); axes[5].legend()

    phases_arr = _field("phase")
    axes[6].plot(episodes, phases_arr, color='teal', linewidth=2)
    last_m = metrics_history[-1]
    axes[6].set_title(f'7. Phase corrente\nUltima: Phase {last_m.get("phase", "?")} | Target {last_m.get("target", 0.0):.2f}')
    axes[6].set_ylabel('Phase'); axes[6].grid(True, alpha=0.3)

    t2t = _field("time_to_target")
    valid = ~np.isnan(t2t)
    if valid.any():
        axes[7].plot(episodes[valid], t2t[valid], alpha=0.3, color='dodgerblue')
        ma_t, idx_t = _ma(t2t[valid], max(10, min(window, max(2, valid.sum() // 2))))
        if ma_t is not None:
            axes[7].plot(episodes[valid][idx_t], ma_t, color='darkblue', linewidth=2, label='MA (successi)')
    axes[7].set_title('8. Tempo al successo [s]'); axes[7].set_ylabel('s'); axes[7].grid(True, alpha=0.3); axes[7].legend()

    for ax in axes:
        ax.set_xlabel('Episodio')
    plt.suptitle('Live Training Dashboard — Reward Curriculum', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(run_folder, 'live_dashboard.png'), dpi=120)
    plt.close(fig)


# ============================================================
# 5. GRAFICI
# ============================================================
def _moving_avg(data, window):
    if len(data) >= window:
        return np.convolve(data, np.ones(window) / window, mode='valid')
    return None


def plot_reward_per_episode(rewards, folder, window=20):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.4, color='steelblue', label='Reward grezza')
    ma = _moving_avg(rewards, window)
    if ma is not None:
        ax.plot(range(window - 1, len(rewards)), ma, color='navy', linewidth=2,
                label=f'Media mobile ({window} ep)')
    ax.set_xlabel('Episodio'); ax.set_ylabel('Reward totale')
    ax.set_title('Reward per Episodio — Goal-Conditioned DDPG (Curriculum + Shaping)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'reward_per_episode.png'), dpi=150)
    plt.close(fig)


def plot_critic_diagnostics(q_mean_history, critic_loss_history, folder, window=20):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, data, label, color_raw, color_ma in [
        (axes[0], q_mean_history,      'Q medio',      'mediumpurple', 'indigo'),
        (axes[1], critic_loss_history, 'Critic Loss',  'tomato',       'firebrick'),
    ]:
        ax.plot(data, alpha=0.4, color=color_raw, label=f'{label} (grezzo)')
        ma = _moving_avg(data, window)
        if ma is not None:
            ax.plot(range(window - 1, len(data)), ma,
                    color=color_ma, linewidth=2, label=f'Media mobile ({window} ep)')
        ax.set_ylabel(label); ax.legend(); ax.grid(True, alpha=0.3)
    axes[1].set_xlabel('Episodio')
    plt.suptitle('Diagnostica Critic — Q overestimation check', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'critic_diagnostics.png'), dpi=150)
    plt.close(fig)
    print("  critic_diagnostics.png salvato.")


def plot_peak_elongation_history(peak_history, folder, window=20):
    if not peak_history:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(peak_history, alpha=0.4, color='darkorange', label='Picco per episodio')
    ma = _moving_avg(peak_history, window)
    if ma is not None:
        ax.plot(range(window - 1, len(peak_history)), ma, color='red', linewidth=2,
                label=f'Media mobile ({window} ep)')
    ax.set_xlabel('Episodio'); ax.set_ylabel('max(xb + xr)  [m]')
    ax.set_title('Picco Massimo Elongazione per Episodio')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'peak_elongation_history.png'), dpi=150)
    plt.close(fig)


def plot_noise_decay(noise_history, folder):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(noise_history, color='darkorange', linewidth=2)
    ax.set_xlabel('Episodio'); ax.set_ylabel('sigma')
    ax.set_title('Decadimento Rumore di Esplorazione')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'noise_decay.png'), dpi=150)
    plt.close(fig)


def plot_best5_position(best_runs, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(best_runs)))
    for i, (run, color) in enumerate(zip(best_runs, colors)):
        label = f"Run #{i+1}  peak={run['peak_y']:.3f} m  R={run['reward']:.1f}"
        axes[0].plot(run['t'], run['xb'], color=color, linewidth=1.5, label=label)
        axes[1].plot(run['t'], run['xr'], color=color, linewidth=1.5)
    axes[0].set_ylabel('xb  [m]'); axes[0].set_title('Top-5: xb')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel('xr  [m]'); axes[1].set_xlabel('Tempo (s)')
    axes[1].set_title('Top-5: xr'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'best5_position.png'), dpi=150)
    plt.close(fig)


def plot_best5_velocities(best_runs, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(best_runs)))
    for i, (run, color) in enumerate(zip(best_runs, colors)):
        axes[0].plot(run['t'], run['xb_dot'], color=color, linewidth=1.5, label=f'Run #{i+1}')
        axes[1].plot(run['t'], run['xr_dot'], color=color, linewidth=1.5)
    for ax in axes:
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("xb_dot  [m/s]"); axes[0].set_title("Top-5: velocita' base")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("xr_dot  [m/s]"); axes[1].set_xlabel('Tempo (s)')
    axes[1].set_title("Top-5: velocita' robot")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'best5_velocities.png'), dpi=150)
    plt.close(fig)


def plot_best5_control(best_runs, folder):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(best_runs)))
    for i, (run, color) in enumerate(zip(best_runs, colors)):
        ax.plot(run['t'], run['u'], color=color, linewidth=1.5, label=f'Run #{i+1}')
    ax.axhline(U_MAX, color='gray', linestyle=':', linewidth=1)
    ax.axhline(U_MIN, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Tempo (s)'); ax.set_ylabel('u')
    ax.set_title('Top-5: Azione di Controllo')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'best5_control.png'), dpi=150)
    plt.close(fig)


def plot_heatmap_state_action(agent, folder, n_points=2000):
    """
    Heatmap stato-azione della policy appresa.
    Lo stato ha 5 elementi; il target viene campionato in [TARGET_POSITION, TARGET_POSITION].
    """
    xb_range    = STATE_SCALE[0]
    xbdot_range = STATE_SCALE[1]
    xr_range    = STATE_SCALE[2]
    xrdot_range = STATE_SCALE[3]

    xb_vals     = np.random.uniform(-xb_range,    xb_range,    n_points).astype(np.float32)
    xbdot_vals  = np.random.uniform(-xbdot_range, xbdot_range, n_points).astype(np.float32)
    xr_vals     = np.random.uniform(-xr_range,    xr_range,    n_points).astype(np.float32)
    xrdot_vals  = np.random.uniform(-xrdot_range, xrdot_range, n_points).astype(np.float32)
    target_vals = np.full(n_points, TARGET_POSITION, dtype=np.float32)

    y_vals  = xb_vals + xr_vals
    dy_vals = xbdot_vals + xrdot_vals

    actions = []
    agent.actor.eval()
    with torch.no_grad():
        for xb, xbdot, xr, xrdot, tgt in zip(xb_vals, xbdot_vals, xr_vals, xrdot_vals, target_vals):
            state_raw  = np.array([xb, xbdot, xr, xrdot, tgt], dtype=np.float32)
            state_norm = normalize_state(state_raw)
            a = agent.select_action(state_norm, noise_std=0.0)
            actions.append(a[0])
    agent.actor.train()
    actions = np.array(actions)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(y_vals, dy_vals, c=actions, cmap='RdBu', s=10,
                    vmin=U_MIN, vmax=U_MAX)
    fig.colorbar(sc, ax=ax, label='u (azione)')
    ax.set_xlabel('Elongazione totale  xb + xr  [m]')
    ax.set_ylabel("Velocita' totale  xb_dot + xr_dot  [m/s]")
    ax.set_title('Heatmap Stato-Azione della Policy Appresa')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'heatmap_state_action.png'), dpi=150)
    plt.close(fig)
    print("  heatmap_state_action.png salvato.")


def plot_success_rate(success_history, folder, window=20):
    """
    Percentuale di successo su finestra mobile.

    Args:
        success_history : lista di 0/1 per episodio (1 = target raggiunto)
        folder          : cartella di output
        window          : ampiezza finestra mobile
    """
    if not success_history:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    success_arr = np.array(success_history, dtype=float)
    ma = _moving_avg(success_arr * 100.0, window)
    if ma is not None:
        ax.plot(range(window - 1, len(success_history)), ma, color='green', linewidth=2,
                label=f'Success rate (finestra {window} ep)')
    ax.set_ylim(0, 105)
    ax.set_xlabel('Episodio'); ax.set_ylabel('Success rate [%]')
    ax.set_title('Tasso di Successo per Episodio (finestra mobile)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'success_rate.png'), dpi=150)
    plt.close(fig)
    print("  success_rate.png salvato.")


def plot_target_distribution(target_history, folder):
    """
    Istogramma dei valori target usati durante il training.

    Args:
        target_history : lista dei valori target per episodio
        folder         : cartella di output
    """
    if not target_history:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(target_history, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Target [m]'); ax.set_ylabel('Conteggio episodi')
    ax.set_title('Distribuzione dei Target Utilizzati nel Training')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'target_distribution.png'), dpi=150)
    plt.close(fig)
    print("  target_distribution.png salvato.")


# ============================================================
# 6. TEST EPISODE
# ============================================================
def run_test_episode(agent, env, target_val):
    """
    Esegue un episodio di test deterministico (no rumore) verso target_val.

    Args:
        agent      : DDPGAgent
        env        : GoalLinearSystemEnv
        target_val : valore target float [m]
    Returns:
        dict con traiettorie e metriche
    """
    max_steps = int(T_MAX / DT)
    state = env.reset(target_val)
    hist_t, hist_xb, hist_xb_dot, hist_xr, hist_xr_dot, hist_u = [], [], [], [], [], []
    total_reward = 0.0
    t = 0.0

    for _ in range(max_steps):
        action = agent.select_action(state, noise_std=0.0)
        raw = state * _STATE_SCALE_NP
        hist_t.append(t)
        hist_xb.append(raw[0]); hist_xb_dot.append(raw[1])
        hist_xr.append(raw[2]); hist_xr_dot.append(raw[3])
        hist_u.append(action[0])
        state, reward, terminated, done = env.step(action[0])
        total_reward += reward
        t += env.dt
        if done or terminated:
            raw = state * _STATE_SCALE_NP
            hist_t.append(t)
            hist_xb.append(raw[0]); hist_xb_dot.append(raw[1])
            hist_xr.append(raw[2]); hist_xr_dot.append(raw[3])
            hist_u.append(action[0])
            break

    y_hist    = [xb + xr for xb, xr in zip(hist_xb, hist_xr)]
    peak_y    = max(abs(y) for y in y_hist) if y_hist else 0.0
    final_y   = y_hist[-1] if y_hist else 0.0
    final_dist = abs(target_val - final_y)

    print(f"  Test → target={target_val:.3f} m | "
          f"y_finale={final_y:.4f} m | dist_finale={final_dist:.4f} m | "
          f"reward={total_reward:.2f}")

    return dict(
        t=hist_t, xb=hist_xb, xb_dot=hist_xb_dot,
        xr=hist_xr, xr_dot=hist_xr_dot, u=hist_u,
        y=y_hist, reward=total_reward, peak_y=peak_y,
        target=target_val, final_dist=final_dist,
    )


# ============================================================
# 7. MAIN — 3-Phase Reward Curriculum
# ============================================================
if __name__ == "__main__":

    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        print(f"Seed fissato a: {SEED}")

    max_steps = int(T_MAX / DT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_folder = os.path.join(script_dir, "RL_results", f"run_reward_curriculum_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    snapshots_folder = os.path.join(run_folder, "training_snapshots")
    os.makedirs(snapshots_folder, exist_ok=True)

    print(f"\nOutput: {run_folder}")
    print("Modalità: reward_curriculum_3_phases")
    print(f"LR actor={LR_ACTOR}  LR critic={LR_CRITIC}  gamma={GAMMA}  tau={TAU}  clip={CLIP_GRAD_NORM}")
    print(f"Actor  : {ACTOR_HIDDEN_LAYERS}x{ACTOR_HIDDEN_SIZE} [{ACTOR_HIDDEN_ACT}|init={ACTOR_HIDDEN_INIT}]"
          f"  -> output [{ACTOR_OUTPUT_ACT}|init={ACTOR_OUTPUT_INIT}]")
    print(f"Critic : {CRITIC_HIDDEN_LAYERS}x{CRITIC_HIDDEN_SIZE} [{CRITIC_HIDDEN_ACT}|init={CRITIC_HIDDEN_INIT}]"
          f"  -> output [{CRITIC_OUTPUT_ACT}|init={CRITIC_OUTPUT_INIT}]")
    print(f"T_max={T_MAX}s  dt={DT}s  max_steps={max_steps}")
    print(f"STATE_SCALE={STATE_SCALE}  REWARD_SCALE={REWARD_SCALE}  WARMUP_STEPS={WARMUP_STEPS}")
    print(f"Reward: C_TIME={C_TIME}  R_GOAL={R_GOAL}  R_VEL_MAX={R_VEL_MAX}  VEL_SIGMA={VEL_SIGMA}  LAMBDA_REG={LAMBDA_REG}")
    print(f"Phase eval window={PHASE_EVAL_WINDOW}  success threshold={SUCCESS_RATE_THRESHOLD}")
    print(f"Noise reset per phase: {NOISE_RESET_PER_PHASE}")
    print(f"Noise decay per phase: {NOISE_DECAY_PER_PHASE}")
    print(f"Noise type: {NOISE_TYPE}")
    print(f"Limiti fisici attuatore: {ACTUATOR_PHYSICAL_LIMIT} (v_max: {V_MAX_ATTUATORE} m/s)")

    assert len(NOISE_RESET_PER_PHASE) == len(CURRICULUM_PHASES)
    assert len(NOISE_DECAY_PER_PHASE) == len(CURRICULUM_PHASES)

    env = GoalLinearSystemEnv()
    agent = DDPGAgent(state_dim=5, action_dim=1)
    buffer = ReplayBuffer()
    csv_path = init_csv_logger(run_folder)

    print(f"  CSV logger: {csv_path}")

    rewards_history = []
    noise_history = []
    peak_history = []
    success_history = []
    q_mean_history = []
    critic_loss_history = []
    phase_history = []
    target_history = []
    metrics_history = []
    top_runs = []
    phase3_top_runs = []
    phase_changes = []

    global_episode = 0

    print("\n--- INIZIO TRAINING (3-Phase Reward Curriculum) ---")

    for phase_idx, phase in enumerate(CURRICULUM_PHASES):
        phase_name = phase["name"]
        phase_episodes = int(phase["episodes"])
        use_shaping = bool(phase["use_shaping"])
        w_shaping = float(phase["w_shaping"])
        final_reward_scale = float(phase["final_reward_scale"])
        time_penalty_scale = float(phase.get("time_penalty_scale", 1.0))
        terminate_on_target_phase = bool(phase.get("terminate_on_target", True))  # se False, fase 1 non termina al target
        noise_current = float(NOISE_RESET_PER_PHASE[phase_idx])
        noise_decay_current = float(NOISE_DECAY_PER_PHASE[phase_idx])
        success_window = deque(maxlen=NOISE_RESCUE_WINDOW)
        rescue_sigma = None
        rescue_episodes_left = 0

        if global_episode > 0:
            phase_changes.append(global_episode)

        print(f"\n=== PHASE {phase_idx + 1}/{len(CURRICULUM_PHASES)}: {phase_name} ===")
        print(f"  episodes={phase_episodes}  shaping={use_shaping}  w_shaping={w_shaping}  final_reward_scale={final_reward_scale}")
        print(f"  noise_reset={noise_current:.3f}  noise_decay={noise_decay_current:.4f}")

        rescue_trigger_start = phase_episodes // 2 + 1

        for phase_episode_idx in range(phase_episodes):
            target_val = get_phase_target(phase)
            state = env.reset(target_val)
            agent.reset_noise()
            agent.reset_episode_accum()

            episode_reward = 0.0
            episode_terminated = False
            reward_components_accum = []
            actions_history = []
            delta_u_history = []
            y_history = []
            ydot_history = []
            t_history = []
            t_ep = 0.0
            u_prev_ep = 0.0

            target_history.append(target_val)

            episode_noise = rescue_sigma if rescue_episodes_left > 0 and rescue_sigma is not None else noise_current
            agent.reset_noise()

            for _step in range(max_steps):
                action = agent.select_action(state, noise_std=episode_noise)
                raw = state * _STATE_SCALE_NP
                xb_cur, xbdot_cur, xr_cur, xrdot_cur = raw[:4]

                y_history.append(float(xb_cur + xr_cur))
                ydot_history.append(float(xbdot_cur + xrdot_cur))
                t_history.append(t_ep)
                actions_history.append(float(action[0]))
                delta_u_history.append(float(action[0]) - u_prev_ep)
                u_prev_ep = float(action[0])

                _current_raw   = raw
                next_state, reward, terminated, done = env.step(
                    action[0],
                    use_shaping_override=use_shaping,
                    w_shaping_override=w_shaping,
                    final_reward_scale_override=final_reward_scale,
                    time_penalty_scale_override=time_penalty_scale,
                    terminate_on_target=terminate_on_target_phase,
                )

                # Allinea il logging ai termini reali usati da compute_reward.
                _next_raw = next_state * _STATE_SCALE_NP
                _v_prev = potential_energy_total(
                    _current_raw[0], _current_raw[2], float(action[0]),
                    env.Kb, env.Kr, env.xb0
                )
                _v_next = potential_energy_total(
                    _next_raw[0], _next_raw[2], float(action[0]),
                    env.Kb, env.Kr, env.xb0
                )
                _delta_v_log = _v_prev - _v_next
                _velocity_term_log = _next_raw[1] * _next_raw[3]
                _log_shaping_ctx = _build_shaping_context(
                    x1a=_next_raw[0],    v1a=_next_raw[1],
                    x2a=_next_raw[2],    v2a=_next_raw[3],
                    x1b=_current_raw[0], v1b=_current_raw[1],
                    x2b=_current_raw[2], v2b=_current_raw[3],
                    kb=env.Kb,  kr=env.Kr,
                    mb=env.Mb,  mr=env.Mr,
                    xb0=env.xb0, u=float(action[0]),
                    target=env.target,
                ) if use_shaping else None
                _shaping_val      = apply_reward_shaping(
                    _delta_v_log, _velocity_term_log,
                    shaping_context=_log_shaping_ctx,
                ) if use_shaping else 0.0
                _r_shaping_log    = w_shaping * _shaping_val if use_shaping else 0.0
                # r_goal e r_vel: nonzero solo se terminato con successo (senza instabilità)
                if terminated and all(np.isfinite(v) for v in next_state):
                    _y_dot_fin   = _next_raw[1] + _next_raw[3]
                    _r_goal_log  = R_GOAL * final_reward_scale
                    _r_vel_log   = R_VEL_MAX * math.exp(-VEL_SIGMA * abs(_y_dot_fin)) * final_reward_scale
                else:
                    _r_goal_log  = 0.0
                    _r_vel_log   = 0.0
                reward_components_accum.append({
                    "r_time":    -C_TIME,
                    "r_shaping": _r_shaping_log,
                    "r_goal":    _r_goal_log,
                    "r_vel":     _r_vel_log,
                    "r_reg":     0.0,
                })

                buffer.push(state, action[0], reward, next_state, terminated)
                c_loss, q_mean = agent.train_step(buffer)
                if c_loss is None:
                    c_loss = float('nan')
                if q_mean is None:
                    q_mean = float('nan')

                state = next_state
                episode_reward += reward
                t_ep += DT

                if terminated:
                    episode_terminated = True
                    break  # il successo viene valutato fuori dal loop con controllo dist + VEL_THRESHOLD

                if done:
                    break

            final_raw = state * _STATE_SCALE_NP
            final_y = float(final_raw[0] + final_raw[2])
            final_ydot = float(final_raw[1] + final_raw[3])
            dist_fin = abs(target_val - final_y)
            success = 1 if (episode_terminated and dist_fin <= TARGET_THRESHOLD and abs(final_ydot) <= VEL_THRESHOLD) else 0

            if y_history:
                y_history[-1] = final_y
            else:
                y_history.append(final_y)
            if ydot_history:
                ydot_history[-1] = final_ydot
            else:
                ydot_history.append(final_ydot)

            if rescue_episodes_left > 0 and rescue_sigma is not None:
                rescue_episodes_left -= 1
                noise_for_metrics = episode_noise
                if rescue_episodes_left == 0:
                    rescue_sigma = None
                    noise_current = max(NOISE_MIN, noise_current * noise_decay_current)
                else:
                    noise_current = rescue_sigma
            else:
                rescue_enabled_now = (
                    phase_idx == len(CURRICULUM_PHASES) - 1 and
                    (phase_episode_idx + 1) >= rescue_trigger_start
                )
                rescue_sigma_new, rescued = _check_rescue_noise(noise_current, success_window) if rescue_enabled_now else (None, False)
                noise_for_metrics = episode_noise
                if rescued:
                    rescue_sigma = rescue_sigma_new
                    rescue_episodes_left = NOISE_RESCUE_HOLD_EPISODES
                    noise_current = rescue_sigma_new
                else:
                    noise_current = max(NOISE_MIN, noise_current * noise_decay_current)

            learning_metrics = agent.get_episode_learning_metrics()
            ep_metrics = _build_episode_metrics(
                global_episode, phase_idx, phase_name, target_val,
                episode_reward, reward_components_accum,
                success, episode_terminated,
                actions_history, delta_u_history,
                y_history, ydot_history,
                t_history,
                noise_for_metrics,
                learning_metrics,
            )
            metrics_history.append(ep_metrics)
            log_episode_csv(csv_path, ep_metrics)

            rewards_history.append(episode_reward)
            noise_history.append(noise_for_metrics)
            peak_history.append(ep_metrics.get("peak_elongation", float('nan')))
            success_history.append(success)
            q_mean_history.append(learning_metrics.get("mean_q", float('nan')))
            critic_loss_history.append(learning_metrics.get("critic_loss", float('nan')))
            phase_history.append(phase_idx + 1)

            candidate = {
                "reward": episode_reward,
                "peak_y": ep_metrics.get("peak_elongation", 0.0),
                "target": target_val,
                "phase": phase_name,
                "phase_idx": phase_idx + 1,
                "actor_state": {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()},
            }
            top_runs.append(candidate)
            top_runs = sorted(top_runs, key=lambda r: r["reward"], reverse=True)[:N_BEST_RUNS]
            if phase_idx == len(CURRICULUM_PHASES) - 1:
                phase3_top_runs.append(candidate)
                phase3_top_runs = sorted(phase3_top_runs, key=lambda r: r["reward"], reverse=True)[:N_BEST_RUNS]

            success_window.append(success)
            global_episode += 1

            if global_episode % LOG_EVERY == 0:
                q_str = f"{learning_metrics['mean_q']:.3f}" if not np.isnan(learning_metrics['mean_q']) else "  n/a"
                l_str = f"{learning_metrics['critic_loss']:.4f}" if not np.isnan(learning_metrics['critic_loss']) else "  n/a"
                print(
                    f"Ep {global_episode:5d} | phase={phase_name:<12} | Reward: {episode_reward:8.2f} | "
                    f"Target: {target_val:.2f} | Dist_finale: {dist_fin:.4f} | Success: {success} | "
                    f"Q medio: {q_str:>10} | Critic loss: {l_str:>10} | sigma: {noise_for_metrics:.4f}"
                )

            _sr_threshold = float(phase.get("success_rate_threshold", SUCCESS_RATE_THRESHOLD))
            _eval_window   = int(phase.get("phase_eval_window", PHASE_EVAL_WINDOW))
            _min_episodes  = int(phase.get("min_phase_episodes", 0))

            if len(success_window) >= _eval_window:
                sr = sum(list(success_window)[-_eval_window:]) / _eval_window
                if sr >= _sr_threshold and phase_episode_idx + 1 >= _min_episodes:
                    print(f" - Phase {phase_name} completata! Success rate {sr:.2%}")
                    break

            if global_episode % DASHBOARD_UPDATE_EVERY == 0:
                plot_training_dashboard(metrics_history, run_folder, phase_changes)

            if global_episode % SNAPSHOT_EVERY == 0:
                save_snapshot(agent, env, target_val, global_episode, snapshots_folder)
                print(f"  [SNAPSHOT] salvato ep {global_episode}")

        if phase_idx == 2:
            agent.save(run_folder, suffix="_phase3")

        if phase_idx == 3:
            agent.save(run_folder, suffix="_phase4")

    print(f"\nTraining completato. Totale episodi: {global_episode}")

    plot_training_dashboard(metrics_history, run_folder, phase_changes)
    print("  live_dashboard.png salvato.")

    print("\n--- SALVATAGGIO ---")
    agent.save(run_folder)

    plot_reward_per_episode(rewards_history, run_folder)
    plot_peak_elongation_history(peak_history, run_folder)
    plot_noise_decay(noise_history, run_folder)
    plot_critic_diagnostics(q_mean_history, critic_loss_history, run_folder)
    plot_heatmap_state_action(agent, run_folder)
    plot_success_rate(success_history, run_folder)
    print("  Tutti i grafici salvati.")

    # ------------------------------------------------------------------
    # SELEZIONE FINALE: evaluate_candidates
    # Ogni candidato viene valutato su N_TEST_RUNS episodi deterministici;
    # si sceglie il modello con maggiore success_rate (e a parità, reward).
    # ------------------------------------------------------------------
    print("\n--- SELEZIONE FINALE: evaluate_candidates ---")
    eval_candidates = phase3_top_runs[:N_BEST_RUNS] if phase3_top_runs else top_runs[:N_BEST_RUNS]
    print(f"  Candidati dalla fase 3: {len(eval_candidates)}  |  test runs per candidato: {N_TEST_RUNS}")

    def evaluate_candidate(agent, env, target_val, n_runs):
        """Valuta un candidato su n_runs episodi deterministici.
        Ritorna (success_rate, mean_reward, best_traj)."""
        results = []
        for _ in range(n_runs):
            traj = run_test_episode(agent, env, target_val)
            dist = traj["final_dist"]
            vel  = abs(traj["xb_dot"][-1] + traj["xr_dot"][-1]) if traj["xb_dot"] else float('inf')
            ok   = 1 if (dist <= TARGET_THRESHOLD and vel <= VEL_THRESHOLD) else 0
            results.append((ok, traj["reward"], traj))
        sr        = float(sum(r[0] for r in results)) / n_runs
        mr        = float(sum(r[1] for r in results)) / n_runs
        best_traj = max(results, key=lambda r: (r[0], r[1]))[2]
        return sr, mr, best_traj

    ranked = []
    for i, run_data in enumerate(eval_candidates):
        agent.actor.load_state_dict(run_data['actor_state'])
        sr, mr, best_traj = evaluate_candidate(agent, env, TARGET_POSITION, N_TEST_RUNS)
        ranked.append((sr, mr, best_traj, run_data, i))
        print(f"  Candidato {i+1}: success_rate={sr:.2%}  mean_reward={mr:.2f}")

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_candidate = ranked[0]
    agent.actor.load_state_dict(best_candidate[3]['actor_state'])
    print(f"\n  Miglior candidato: #{best_candidate[4]+1} | SR={best_candidate[0]:.2%} | mean_R={best_candidate[1]:.2f}")
    torch.save(agent.actor.state_dict(), os.path.join(run_folder, "best_actor.pth"))
    print("  best_actor.pth salvato.")

    best_test_runs = [r[2] for r in ranked[:N_BEST_RUNS]]

    plot_best5_position(best_test_runs, run_folder)
    plot_best5_velocities(best_test_runs, run_folder)
    plot_best5_control(best_test_runs, run_folder)

    config = {
        "training_mode": "reward_curriculum_3_phases",
        "seed": SEED,
        "sistema": dict(zip(["Mr", "Mb", "Kr", "Kb", "hr", "hb"], PARAMETRI_SISTEMA)),
        "dt": DT, "T_max": T_MAX, "u_min": U_MIN, "u_max": U_MAX,
        "target_threshold": TARGET_THRESHOLD,
        "vel_threshold": VEL_THRESHOLD,
        "target_position": TARGET_POSITION,

        "normalizzazione": {
            "state_scale": STATE_SCALE,
            "reward_scale": REWARD_SCALE,
        },
        "reward": {
            "c_time": C_TIME,
            "use_shaping": USE_SHAPING,
            "w_shaping": W_SHAPING,
            "r_goal": R_GOAL,
            "r_vel_max": R_VEL_MAX,
            "VEL_SIGMA": VEL_SIGMA,
            "lambda_reg": LAMBDA_REG,
        },
        "curriculum": {
            "phases": CURRICULUM_PHASES,
            "phase_eval_window": PHASE_EVAL_WINDOW,
            "success_rate_threshold": SUCCESS_RATE_THRESHOLD,
        },
        "training": {
            "lr_actor": LR_ACTOR,
            "lr_critic": LR_CRITIC,
            "gamma": GAMMA,
            "tau": TAU,
            "clip_grad_norm": CLIP_GRAD_NORM,
            "batch_size": BATCH_SIZE,
            "buffer_capacity": BUFFER_CAPACITY,
            "warmup_steps": WARMUP_STEPS,
        },
        "noise": {
            "noise_reset_per_phase": NOISE_RESET_PER_PHASE,
            "noise_decay_per_phase": NOISE_DECAY_PER_PHASE,
            "noise_type": NOISE_TYPE,
            "noise_rescue_enable": NOISE_RESCUE_ENABLE,
            "noise_rescue_trigger_sigma": NOISE_RESCUE_TRIGGER_SIGMA,
            "noise_rescue_trigger_sr": NOISE_RESCUE_TRIGGER_SR,
            "noise_rescue_window": NOISE_RESCUE_WINDOW,
            "noise_rescue_sr_low": NOISE_RESCUE_SR_LOW,
            "noise_rescue_sr_very_low": NOISE_RESCUE_SR_VERY_LOW,
            "noise_rescue_value_low": NOISE_RESCUE_VALUE_LOW,
            "noise_rescue_value_very_low": NOISE_RESCUE_VALUE_VERY_LOW,
            "noise_rescue_hold_episodes": NOISE_RESCUE_HOLD_EPISODES,
        },
        "attuatore": {
            "actuator_physical_limit": ACTUATOR_PHYSICAL_LIMIT,
            "v_max_attuatore": V_MAX_ATTUATORE,
        },
        "total_episodes": global_episode,
    }
    with open(os.path.join(run_folder, "simulation_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("  simulation_config.json salvato.")
    print(f"\n=== Fine training. Output in: {run_folder} ===")
