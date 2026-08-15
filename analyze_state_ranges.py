import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — modifica AGENT_FOLDER con il path della tua run
# ============================================================
AGENT_FOLDER = r"C:\Users\tomga\Desktop\Progetti ML\ROBOT\RL_results\run_200ep_reward2_20260629_2306"

# Parametri sistema (devono essere identici al training)
PARAMETRI_SISTEMA = [2.5, 70.0, 1500, 4000, 0.30, 0.25]
U_MIN = -1.0
U_MAX =  1.0
DT    =  0.01
T_MAX = 10.0

# Quanti episodi girare per raccogliere statistiche (piu' e' meglio, 50 e' un buon default)
NUM_EPISODES_ANALYSIS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Copia delle classi del training (devono coincidere esattamente)
# ============================================================
ACTOR_HIDDEN_SIZE    = 256
ACTOR_HIDDEN_LAYERS  = 2


def _build_mlp(input_dim, output_dim, hidden_size, hidden_layers, output_activation=None):
    layers = []
    in_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.ReLU())
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, u_min, u_max,
                 hidden_size=ACTOR_HIDDEN_SIZE, hidden_layers=ACTOR_HIDDEN_LAYERS):
        super().__init__()
        self.u_min = u_min
        self.u_max = u_max
        self.register_buffer('u_min_t', torch.tensor(u_min, dtype=torch.float32))
        self.register_buffer('u_max_t', torch.tensor(u_max, dtype=torch.float32))
        self.network = _build_mlp(state_dim, action_dim, hidden_size, hidden_layers,
                                  output_activation=nn.Tanh())

    def forward(self, state):
        tanh_out = self.network(state)
        scaled   = 0.5 * (self.u_max - self.u_min) * tanh_out + 0.5 * (self.u_max + self.u_min)
        return torch.clamp(scaled, self.u_min_t, self.u_max_t)


# ============================================================
# Ambiente (identico al training)
# ============================================================
class LinearSystemEnv:
    def __init__(self, parameters, u_min=-1.0, u_max=1.0, dt=0.01, T_max=10.0):
        self.u_min = u_min
        self.u_max = u_max
        self.dt    = dt
        self.T_max = T_max

        self.Mr, self.Mb, self.Kr, self.Kb, self.hr, self.hb = parameters
        self.Dr  = 2 * self.hr * np.sqrt(self.Kr * self.Mr)
        self.Db  = 2 * self.hb * np.sqrt(self.Kb * self.Mb)
        self.xb0 = 0.0

        self.M_mat = np.array([
            [self.Mr + self.Mb, self.Mr],
            [self.Mr,           self.Mr],
        ], dtype=float)

        self.x          = np.zeros(4)
        self.step_count = 0

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

    def reset(self):
        self.x          = np.zeros(4)
        self.step_count = 0
        return self.x.copy()

    def step(self, u):
        u_clipped = np.clip(u, self.u_min, self.u_max)
        self.step_count += 1
        self.x = self._rk4_step(self.x, float(u_clipped))
        done   = self.step_count >= int(self.T_max / self.dt)
        return self.x.copy(), done


# ============================================================
# ANALISI
# ============================================================
def run_analysis():
    # --- Carica agente ---
    actor_path = os.path.join(AGENT_FOLDER, "agent_actor.pth")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"File non trovato: {actor_path}\n"
            "Assicurati di aver impostato AGENT_FOLDER correttamente in cima allo script."
        )

    actor = Actor(state_dim=4, action_dim=1, u_min=U_MIN, u_max=U_MAX).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print(f"Agente caricato da: {AGENT_FOLDER}")

    env       = LinearSystemEnv(PARAMETRI_SISTEMA, u_min=U_MIN, u_max=U_MAX, dt=DT, T_max=T_MAX)
    max_steps = int(T_MAX / DT)

    # Colleziona tutti i valori di stato durante gli episodi
    all_xb     = []
    all_xb_dot = []
    all_xr     = []
    all_xr_dot = []

    print(f"\nRaccolta dati su {NUM_EPISODES_ANALYSIS} episodi...")
    for ep in range(NUM_EPISODES_ANALYSIS):
        state = env.reset()
        for _ in range(max_steps):
            all_xb.append(state[0])
            all_xb_dot.append(state[1])
            all_xr.append(state[2])
            all_xr_dot.append(state[3])

            s_t = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = actor(s_t).cpu().numpy()
            state, done = env.step(action[0])
            if done:
                break

        if (ep + 1) % 10 == 0:
            print(f"  Episodio {ep + 1}/{NUM_EPISODES_ANALYSIS} completato")

    all_xb     = np.array(all_xb)
    all_xb_dot = np.array(all_xb_dot)
    all_xr     = np.array(all_xr)
    all_xr_dot = np.array(all_xr_dot)

    # ----------------------------------------------------------------
    # Statistiche
    # ----------------------------------------------------------------
    def stats(arr, name):
        p5, p95 = np.percentile(arr, 5), np.percentile(arr, 95)
        print(f"  {name:20s}  min={arr.min():+.4f}  max={arr.max():+.4f}  "
              f"5%={p5:+.4f}  95%={p95:+.4f}  std={arr.std():.4f}")
        return arr.min(), arr.max(), p5, p95

    print("\n========== RANGE OSSERVATI DURANTE LE TRAIETTORIE ==========")
    xb_min,  xb_max,  xb_p5,  xb_p95  = stats(all_xb,     "xb  [m]")
    xbd_min, xbd_max, xbd_p5, xbd_p95 = stats(all_xb_dot, "xb_dot [m/s]")
    xr_min,  xr_max,  xr_p5,  xr_p95  = stats(all_xr,     "xr  [m]")
    xrd_min, xrd_max, xrd_p5, xrd_p95 = stats(all_xr_dot, "xr_dot [m/s]")

    # ----------------------------------------------------------------
    # Suggerimento range inizializzazione
    # - xb      : 20% del range p5-p95 (movimento base e' piccolo)
    # - xb_dot  : 20% del range p5-p95
    # - xr      : dentro +-1 [m] rispetto alla posizione iniziale di xb
    # - xr_dot  : 20% del range p5-p95
    # ----------------------------------------------------------------
    FRACTION  = 0.20   # frazione del range p5-p95 usata come bound di init
    XR_RADIUS = 1.0    # raggio entro cui xr puo' partire rispetto a xb_init

    xb_range_init  = FRACTION * (xb_p95  - xb_p5)  / 2
    xbd_range_init = FRACTION * (xbd_p95 - xbd_p5) / 2
    xrd_range_init = FRACTION * (xrd_p95 - xrd_p5) / 2

    print("\n========== SUGGERIMENTO INIZIALIZZAZIONE ==========")
    print(f"  (Basato sul {int(FRACTION*100)}% del range p5-p95 delle traiettorie osservate)")
    print(f"  xb_init    ~ Uniform( {-xb_range_init:+.5f},  {+xb_range_init:+.5f} ) m")
    print(f"  xb_dot_init~ Uniform( {-xbd_range_init:+.5f},  {+xbd_range_init:+.5f} ) m/s")
    print(f"  xr_init    ~ xb_init + Uniform( -{XR_RADIUS:.1f},  +{XR_RADIUS:.1f} ) m")
    print(f"  xr_dot_init~ Uniform( {-xrd_range_init:+.5f},  {+xrd_range_init:+.5f} ) m/s")
    print()
    print("  >>> Codice reset() suggerito:")
    print(f"""
    def reset(self, randomize=True):
        if randomize:
            xb_init  = np.random.uniform({-xb_range_init:.5f},  {xb_range_init:.5f})
            self.x = np.array([
                xb_init,
                np.random.uniform({-xbd_range_init:.5f}, {xbd_range_init:.5f}),   # xb_dot
                xb_init + np.random.uniform(-{XR_RADIUS:.1f}, {XR_RADIUS:.1f}),   # xr
                np.random.uniform({-xrd_range_init:.5f}, {xrd_range_init:.5f}),   # xr_dot
            ])
        else:
            self.x = np.zeros(4)
        self.max_y_reached = -np.inf
        self.step_count    = 0
        return self.x.copy()
    """)

    # ----------------------------------------------------------------
    # Plot distribuzioni
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    data_info = [
        (all_xb,     "xb [m]",       "steelblue",  xb_p5,  xb_p95),
        (all_xb_dot, "xb_dot [m/s]", "navy",       xbd_p5, xbd_p95),
        (all_xr,     "xr [m]",       "darkorange",  xr_p5,  xr_p95),
        (all_xr_dot, "xr_dot [m/s]", "firebrick",  xrd_p5, xrd_p95),
    ]
    for ax, (data, label, color, p5, p95) in zip(axes, data_info):
        ax.hist(data, bins=100, color=color, alpha=0.7, density=True)
        ax.axvline(p5,  color='black', linestyle='--', linewidth=1.3, label='5° percentile')
        ax.axvline(p95, color='black', linestyle=':',  linewidth=1.3, label='95° percentile')
        ax.set_title(f'Distribuzione {label}')
        ax.set_xlabel(label)
        ax.set_ylabel('Densita\'')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Distribuzione stati — {NUM_EPISODES_ANALYSIS} episodi (policy trainata, no rumore)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out_png  = os.path.join(AGENT_FOLDER, "state_distribution_analysis.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Plot salvato in: {out_png}")

    # ----------------------------------------------------------------
    # Salva JSON con range osservati e range suggeriti
    # ----------------------------------------------------------------
    result = {
        "num_episodes_analyzed": NUM_EPISODES_ANALYSIS,
        "observed_ranges": {
            "xb":     {"min": float(xb_min),  "max": float(xb_max),  "p5": float(xb_p5),  "p95": float(xb_p95)},
            "xb_dot": {"min": float(xbd_min), "max": float(xbd_max), "p5": float(xbd_p5), "p95": float(xbd_p95)},
            "xr":     {"min": float(xr_min),  "max": float(xr_max),  "p5": float(xr_p5),  "p95": float(xr_p95)},
            "xr_dot": {"min": float(xrd_min), "max": float(xrd_max), "p5": float(xrd_p5), "p95": float(xrd_p95)},
        },
        "suggested_init_ranges": {
            "fraction_of_p5p95_used": FRACTION,
            "xb":              [-round(xb_range_init,  5), round(xb_range_init,  5)],
            "xb_dot":          [-round(xbd_range_init, 5), round(xbd_range_init, 5)],
            "xr_relative_xb":  [-XR_RADIUS, XR_RADIUS],
            "xr_dot":          [-round(xrd_range_init, 5), round(xrd_range_init, 5)],
        }
    }
    json_path = os.path.join(AGENT_FOLDER, "init_range_suggestion.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON suggerimenti salvato in: {json_path}")

    return result


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    result = run_analysis()
    print("\nAnalisi completata.")
