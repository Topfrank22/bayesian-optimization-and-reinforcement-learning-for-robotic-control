import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation


# ============================================================
# CONFIGURAZIONE UTENTE
# ============================================================
SEED = 420
DT = 0.01
T_MAX = 10.0

NOISE_TYPE = "coloured"   # "ou" | "ou_no_sigma" | "coloured"

THETA = 4
MU = 0.0
SIGMA_OU = 1.2

COLOURED_BETA = 1.3

NOISE_START = 1.0
NOISE_DECAY = 0.993
NOISE_MIN = 0.05

U_MIN = -1.0
U_MAX = 1.0
BASE_ACTION = 0.0

SIN_FREQ = 3.8985
SIN_AMPLITUDE = 1.0
SIN_PHASE = 0.0

# La simulazione resta a DT=0.01, ma la visualizzazione usa frame precomputati.
DISPLAY_FPS = 30

MSD_PARAMS = {
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

BLOCK_W = 0.25
BLOCK_H = 0.08
SPRING_W = 0.04
N_COILS = 6

FIG_BG = 'white'
AX_BG = '#f5f7fa'
SPINE_CLR = '#cccccc'
TEXT_CLR = '#222222'
TEXT_MUTED = '#555555'
GRID_CLR = '#dddddd'

COL_KB = '#0a8a7e'
COL_KR = '#c97d00'
COL_BASE = '#2c6fcd'
COL_ROBOT = '#c0195e'
COL_CTRL = '#b36000'
COL_NOISE = '#1f77b4'
COL_SINE = '#ff7f0e'

GHOST_STEPS = 4
GHOST_ALPHA_MAX = 0.30


# ============================================================
# RUMORE
# ============================================================
class ColouredNoise:
    def __init__(self, action_dim=1):
        self.action_dim = action_dim
        self._seq = None
        self._idx = 0

    def reset(self, n_steps):
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

    def sample(self, sigma=1.0):
        noise = self._seq[self._idx] * sigma
        self._idx += 1
        return noise.copy()



def simulate_ou_noise(seed=42, dt=0.01, t_max=10.0, theta=2.0, mu=0.0,
                      sigma_ou=0.2, noise_start=1.0, noise_decay=0.993,
                      noise_min=0.05, base_action=0.0, u_min=-1.0, u_max=1.0):
    np.random.seed(seed)
    n_steps = int(round(t_max / dt))
    t = np.arange(n_steps) * dt
    x = 0.0
    sigma_general = max(noise_min, noise_start)

    ou_state_hist = []
    sigma_general_hist = []
    noisy_action_hist = []
    raw_action_hist = []
    noise_term_hist = []

    for _ in range(n_steps):
        eps = np.random.randn()
        dx = theta * dt * (mu - x) + sigma_ou * np.sqrt(dt) * eps
        x = x + dx
        noise_term = sigma_general * x
        raw_action = base_action + noise_term
        noisy_action = np.clip(raw_action, u_min, u_max)

        ou_state_hist.append(x)
        sigma_general_hist.append(sigma_general)
        noise_term_hist.append(noise_term)
        raw_action_hist.append(raw_action)
        noisy_action_hist.append(noisy_action)

    return {
        "t": t,
        "ou_state": np.array(ou_state_hist),
        "sigma_general": np.array(sigma_general_hist),
        "noise_term": np.array(noise_term_hist),
        "raw_action": np.array(raw_action_hist),
        "noisy_action": np.array(noisy_action_hist),
    }


def simulate_ou_noise_no_sigma(seed=42, dt=0.01, t_max=10.0, theta=2.0, mu=0.0,
                               sigma_ou=0.2, noise_start=1.0, noise_decay=0.993,
                               noise_min=0.05, base_action=0.0, u_min=-1.0, u_max=1.0):
    """
    Variante OU senza moltiplicatore esterno sull'azione.
    La sigma dell'episodio viene inserita direttamente nella dinamica OU:
        dx = theta*dt*(mu - x) + sigma_episode*sqrt(dt)*N(0,1)
    e poi:
        action_noisy = clip(base_action + x, u_min, u_max)
    """
    np.random.seed(seed)
    n_steps = int(round(t_max / dt))
    t = np.arange(n_steps) * dt
    x = 0.0
    sigma_episode = max(noise_min, noise_start)

    ou_state_hist = []
    sigma_general_hist = []
    noisy_action_hist = []
    raw_action_hist = []
    noise_term_hist = []

    for _ in range(n_steps):
        eps = np.random.randn()
        dx = theta * dt * (mu - x) + sigma_episode * np.sqrt(dt) * eps
        x = x + dx
        noise_term = x
        raw_action = base_action + noise_term
        noisy_action = np.clip(raw_action, u_min, u_max)

        ou_state_hist.append(x)
        sigma_general_hist.append(sigma_episode)
        noise_term_hist.append(noise_term)
        raw_action_hist.append(raw_action)
        noisy_action_hist.append(noisy_action)

    return {
        "t": t,
        "ou_state": np.array(ou_state_hist),
        "sigma_general": np.array(sigma_general_hist),
        "noise_term": np.array(noise_term_hist),
        "raw_action": np.array(raw_action_hist),
        "noisy_action": np.array(noisy_action_hist),
    }



def simulate_coloured_noise(seed=42, dt=0.01, t_max=10.0, noise_start=1.0,
                            base_action=0.0, u_min=-1.0, u_max=1.0):
    np.random.seed(seed)
    n_steps = int(round(t_max / dt))
    t = np.arange(n_steps) * dt

    noise_engine = ColouredNoise(action_dim=1)
    noise_engine.reset(n_steps=n_steps)
    sigma_general = noise_start

    coloured_state_hist = []
    sigma_general_hist = []
    noisy_action_hist = []
    raw_action_hist = []
    noise_term_hist = []

    for _ in range(n_steps):
        base_noise = noise_engine.sample(sigma=1.0)[0]
        noise_term = sigma_general * base_noise
        raw_action = base_action + noise_term
        noisy_action = np.clip(raw_action, u_min, u_max)

        coloured_state_hist.append(base_noise)
        sigma_general_hist.append(sigma_general)
        noise_term_hist.append(noise_term)
        raw_action_hist.append(raw_action)
        noisy_action_hist.append(noisy_action)

    return {
        "t": t,
        "coloured_state": np.array(coloured_state_hist),
        "sigma_general": np.array(sigma_general_hist),
        "noise_term": np.array(noise_term_hist),
        "raw_action": np.array(raw_action_hist),
        "noisy_action": np.array(noisy_action_hist),
    }



def simulate_noise(seed=42, dt=0.01, t_max=10.0, noise_type="ou", theta=2.0, mu=0.0,
                   sigma_ou=0.2, noise_start=1.0, noise_decay=0.993, noise_min=0.05,
                   base_action=0.0, u_min=-1.0, u_max=1.0):
    if noise_type == "coloured":
        return simulate_coloured_noise(seed=seed, dt=dt, t_max=t_max, noise_start=noise_start,
                                       base_action=base_action, u_min=u_min, u_max=u_max)
    if noise_type == "ou_no_sigma":
        return simulate_ou_noise_no_sigma(seed=seed, dt=dt, t_max=t_max, theta=theta, mu=mu,
                                          sigma_ou=sigma_ou, noise_start=noise_start,
                                          noise_decay=noise_decay, noise_min=noise_min,
                                          base_action=base_action, u_min=u_min, u_max=u_max)
    return simulate_ou_noise(seed=seed, dt=dt, t_max=t_max, theta=theta, mu=mu,
                             sigma_ou=sigma_ou, noise_start=noise_start,
                             noise_decay=noise_decay, noise_min=noise_min,
                             base_action=base_action, u_min=u_min, u_max=u_max)



def generate_sine(t, freq=3.8985, amplitude=1.0, phase=0.0):
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


# ============================================================
# MSD
# ============================================================
def _build_msd_params(overrides=None):
    p = MSD_PARAMS.copy()
    if overrides:
        p.update(overrides)
    p["Db"] = 2 * np.sqrt(p["Kb"] * p["Mb"]) * p["zeta_b"]
    p["Dr"] = 2 * np.sqrt(p["Kr"] * p["Mr"]) * p["zeta_r"]
    p["X_TARGET"] = p["X_MAX_ROBOT"] / 2 + p["GAINED_DISTANCE"]
    return p



def _msd_rhs(t, z, xr0_val, p):
    xb, xb_dot, xr, xr_dot = z
    m_mat = np.array([
        [p["Mr"] + p["Mb"], p["Mr"]],
        [p["Mr"], p["Mr"]],
    ])
    rhs = np.array([
        -p["Db"] * xb_dot - p["Kb"] * (xb - p["xb0"]),
        -p["Dr"] * xr_dot - p["Kr"] * (xr - xr0_val),
    ])
    xb_ddot, xr_ddot = np.linalg.solve(m_mat, rhs)
    return np.array([xb_dot, xb_ddot, xr_dot, xr_ddot], dtype=float)



def simulate_msd_with_noise(noise_signal, dt, p=None):
    if p is None:
        p = _build_msd_params()

    u_signal = np.asarray(noise_signal, dtype=float).copy()
    n_steps = len(u_signal)
    t_vals = np.arange(n_steps) * dt

    z = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xb_vals = np.empty(n_steps)
    xr_vals = np.empty(n_steps)
    xee_vals = np.empty(n_steps)
    xee_dot_vals = np.empty(n_steps)

    for i in range(n_steps):
        u = u_signal[i]
        k1 = _msd_rhs(0.0, z, u, p)
        k2 = _msd_rhs(0.0, z + 0.5 * dt * k1, u, p)
        k3 = _msd_rhs(0.0, z + 0.5 * dt * k2, u, p)
        k4 = _msd_rhs(0.0, z + dt * k3, u, p)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        xb_vals[i] = z[0]
        xr_vals[i] = z[2]
        xee_vals[i] = z[0] + z[2]
        xee_dot_vals[i] = z[1] + z[3]

    return {
        "t": t_vals,
        "xb": xb_vals,
        "xr": xr_vals,
        "xee": xee_vals,
        "xee_dot": xee_dot_vals,
        "u": u_signal,
    }


# ============================================================
# PRECOMPUTE VISUAL FRAMES
# ============================================================
def build_visual_frames(ts, xbs, xrs, xees, us, display_fps=30, speed=1):
    speed = max(1, int(speed))
    dt_vis = 1.0 / display_fps
    t_end = float(ts[-1]) if len(ts) > 0 else 0.0

    t_visual = np.arange(0.0, t_end + 0.5 * dt_vis / speed, dt_vis / speed)
    sample_idx = np.searchsorted(ts, t_visual, side='left')
    sample_idx = np.clip(sample_idx, 0, len(ts) - 1)

    return {
        "t_visual": t_visual,
        "sample_idx": sample_idx,
        "xb": xbs[sample_idx],
        "xr": xrs[sample_idx],
        "xee": xees[sample_idx],
        "u": us[sample_idx],
    }


# ============================================================
# HELPER GRAFICI
# ============================================================
def _spring_path_h(x_start, x_end, y_center=0.5):
    n_pts = N_COILS * 4 + 2
    xs = np.linspace(x_start, x_end, n_pts)
    ys = np.zeros(n_pts)
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


# ============================================================
# VISUALIZZAZIONE
# ============================================================
def run_animation(speed=1, save_gif=False):
    noise_type_str = "OU" if NOISE_TYPE == "ou" else ("OU no sigma" if NOISE_TYPE == "ou_no_sigma" else f"Coloured (1/f^{COLOURED_BETA})")
    print(f"=== Generazione rumore {noise_type_str} ===")

    noise_data = simulate_noise(
        seed=SEED, dt=DT, t_max=T_MAX, noise_type=NOISE_TYPE,
        theta=THETA, mu=MU, sigma_ou=SIGMA_OU,
        noise_start=NOISE_START, noise_decay=NOISE_DECAY, noise_min=NOISE_MIN,
        base_action=BASE_ACTION, u_min=U_MIN, u_max=U_MAX,
    )

    t_noise = noise_data["t"]
    noisy_action = np.asarray(noise_data["noisy_action"], dtype=float)
    raw_action = np.asarray(noise_data["raw_action"], dtype=float)
    sigma_general = np.asarray(noise_data["sigma_general"], dtype=float)

    sine_wave = generate_sine(t_noise, freq=SIN_FREQ, amplitude=SIN_AMPLITUDE, phase=SIN_PHASE)

    print("=== Mostra grafico rumore (chiudi la finestra per continuare) ===")
    plt.figure(figsize=(14, 7))
    plt.plot(t_noise, noisy_action, label=f"Rumore {noise_type_str} sull'action (clip [-1,1])", linewidth=1.8, color=COL_NOISE)
    plt.plot(t_noise, raw_action, label="Action pre-clip", linewidth=1.0, color=COL_CTRL, alpha=0.6)
    plt.plot(t_noise, sine_wave, label="Sinusoide 3.8985 Hz", linewidth=1.2, alpha=0.9, color=COL_SINE)
    plt.axhline(U_MAX, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Cap +1")
    plt.axhline(U_MIN, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Cap -1")
    if np.allclose(sigma_general, sigma_general[0]):
        plt.axhline(sigma_general[0], color="green", linestyle="--", linewidth=1.2, alpha=0.7,
                    label="σ_generale (costante)")
    plt.title(f"Action rumorosa usata come input MSD — {noise_type_str}")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Valore")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)

    print("=== Simulazione MSD con rumore come input ===")
    p = _build_msd_params()
    msd_data = simulate_msd_with_noise(noisy_action, DT, p=p)

    ts = msd_data["t"]
    xbs = msd_data["xb"]
    xrs = msd_data["xr"]
    xees = msd_data["xee"]
    us = np.asarray(msd_data["u"], dtype=float)

    if not np.allclose(us, noisy_action):
        raise RuntimeError("Il segnale usato nel visualizzatore non coincide con quello plottato.")

    visual = build_visual_frames(ts, xbs, xrs, xees, us, display_fps=DISPLAY_FPS, speed=speed)
    sample_idx = visual["sample_idx"]
    n_anim_frames = len(sample_idx)
    interval_ms = int(round(1000 / DISPLAY_FPS))

    xee_max = float(np.max(xees))
    xee_min = float(np.min(xees))
    print(f"[visualizer] xee max = {xee_max:.4f} m, xee min = {xee_min:.4f} m")
    print(f"[visualizer] precomputed frames = {n_anim_frames}, display_fps = {DISPLAY_FPS}")

    LANE_Y = 0.5
    X_MIN = -0.15
    X_MAX = max(xee_max + 0.20, 1.5)
    if xee_min < -0.1:
        X_MIN = min(X_MIN, xee_min - 0.20)

    fig = plt.figure(figsize=(14, 9.5), facecolor=FIG_BG)
    fig.suptitle(
        f"Visualizer {noise_type_str} → MSD | SEED={SEED} | σ_gen={NOISE_START} | xee_max={xee_max:.3f} m | precomputed",
        color=TEXT_CLR, fontsize=10, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 2, left=0.06, right=0.97, top=0.92, bottom=0.22,
                           wspace=0.30, hspace=0.42, height_ratios=[1.6, 1.0])
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
    ax_msd.set_title(f'Sistema MSD — input: {noise_type_str}', color=TEXT_CLR, fontsize=10, fontweight='bold')
    ax_msd.axvline(0.0, color='#888888', linewidth=2.0, zorder=1)
    ax_msd.axhline(LANE_Y, color='#cccccc', linewidth=1.0, linestyle='-', zorder=1)

    spring_base_line, = ax_msd.plot([], [], color=COL_KB, linewidth=2.2, zorder=3)
    spring_robot_line, = ax_msd.plot([], [], color=COL_KR, linewidth=2.2, zorder=3)

    ghost_base_patches = []
    ghost_robot_patches = []
    for _ in range(GHOST_STEPS):
        gp_b = _make_block_h(ax_msd, 0.0, LANE_Y, COL_BASE)
        gp_r = _make_block_h(ax_msd, 0.0, LANE_Y, COL_ROBOT)
        gp_b.set_alpha(0.0)
        gp_r.set_alpha(0.0)
        gp_b.set_zorder(3)
        gp_r.set_zorder(3)
        ax_msd.add_patch(gp_b)
        ax_msd.add_patch(gp_r)
        ghost_base_patches.append(gp_b)
        ghost_robot_patches.append(gp_r)

    base_patch = _make_block_h(ax_msd, 0.0, LANE_Y, COL_BASE)
    robot_patch = _make_block_h(ax_msd, 0.0, LANE_Y, COL_ROBOT)
    ax_msd.add_patch(base_patch)
    ax_msd.add_patch(robot_patch)

    state_text = ax_msd.text(X_MIN + abs(X_MAX - X_MIN) * 0.02, 0.97, '', color=TEXT_CLR,
                             fontsize=8, va='top', fontfamily='monospace', zorder=10)

    ax_pos.set_xlim(ts[0], ts[-1])
    y_margin = max(0.05, 0.1 * (np.max(xees) - np.min(xees) + 1e-6))
    ax_pos.set_ylim(min(xees) - y_margin, max(xees) + y_margin)
    ax_pos.axhline(0.0, color='#aaaaaa', linewidth=0.8, linestyle=':')
    ax_pos.set_xlabel('Tempo (s)', fontsize=8)
    ax_pos.set_ylabel('xb + xr (m)', fontsize=8)
    ax_pos.set_title('Posizione punta (xee)', fontsize=9, fontweight='bold')
    pos_line, = ax_pos.plot([], [], color=COL_KB, linewidth=1.8)
    pos_cursor = ax_pos.axvline(ts[0], color='#888888', linewidth=0.8, alpha=0.6)

    ax_ctrl.set_xlim(ts[0], ts[-1])
    ax_ctrl.set_ylim(-1.05, 1.05)
    ax_ctrl.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax_ctrl.axhline(U_MAX, color='#e03030', linewidth=0.8, linestyle=':', alpha=0.6, label=f'+{U_MAX}')
    ax_ctrl.axhline(U_MIN, color='#e03030', linewidth=0.8, linestyle=':', alpha=0.6, label=f'{U_MIN}')
    ax_ctrl.set_xlabel('Tempo (s)', fontsize=8)
    ax_ctrl.set_ylabel(f'u = rumore {noise_type_str}', fontsize=8)
    ax_ctrl.set_title('Input realmente inviato al MSD', fontsize=9, fontweight='bold')
    ax_ctrl.legend(fontsize=7, facecolor='white', framealpha=0.8)
    ctrl_line, = ax_ctrl.plot([], [], color=COL_CTRL, linewidth=1.8)
    ctrl_cursor = ax_ctrl.axvline(ts[0], color='#888888', linewidth=0.8, alpha=0.6)

    legend_elements = [
        Line2D([0], [0], color=COL_KB, linewidth=2, label='Molla base (Kb)'),
        Line2D([0], [0], color=COL_KR, linewidth=2, label='Molla robot (Kr)'),
        mpatches.Patch(facecolor=COL_BASE, label='Massa base (Mb)'),
        mpatches.Patch(facecolor=COL_ROBOT, label='Massa robot (Mr)'),
    ]
    ax_msd.legend(handles=legend_elements, loc='lower right', fontsize=7, facecolor='white', framealpha=0.8)

    bw_half = BLOCK_H * 0.75
    state = {"frame": 0, "playing": True}

    def _update_patch_pos(patch, x_center, y_center):
        bw = BLOCK_H * 1.5
        bh = BLOCK_W * 0.45
        patch.set_width(bw)
        patch.set_height(bh)
        patch.set_x(x_center - bw / 2)
        patch.set_y(y_center - bh / 2)

    ax_slider = fig.add_axes([0.10, 0.12, 0.80, 0.03], facecolor='#f0f0f0')
    ax_pp = fig.add_axes([0.35, 0.05, 0.12, 0.045])
    ax_rw = fig.add_axes([0.21, 0.05, 0.12, 0.045])
    ax_back = fig.add_axes([0.28, 0.05, 0.06, 0.045])
    ax_fwd = fig.add_axes([0.48, 0.05, 0.06, 0.045])

    slider_frame = Slider(ax_slider, 'Frame', 0, n_anim_frames - 1, valinit=0, valstep=1, color=COL_KB)
    btn_pp = Button(ax_pp, '⏸ Pause', color='#e8edf2', hovercolor='#cfd8e3')
    btn_rw = Button(ax_rw, '⏮ Rewind', color='#e8edf2', hovercolor='#cfd8e3')
    btn_back = Button(ax_back, '◀ -10', color='#e8edf2', hovercolor='#cfd8e3')
    btn_fwd = Button(ax_fwd, '+10 ▶', color='#e8edf2', hovercolor='#cfd8e3')

    for btn in [btn_pp, btn_rw, btn_back, btn_fwd]:
        btn.label.set_fontsize(9)
        btn.label.set_color(TEXT_CLR)

    def draw_frame(anim_frame_idx):
        anim_frame_idx = int(np.clip(anim_frame_idx, 0, n_anim_frames - 1))
        i = int(sample_idx[anim_frame_idx])

        xb = float(xbs[i])
        xr = float(xrs[i])
        x_base = xb
        x_robot = float(xees[i])

        sx, sy = _spring_path_h(0.0, x_base - bw_half, LANE_Y)
        sx2, sy2 = _spring_path_h(x_base + bw_half, x_robot - bw_half, LANE_Y)
        spring_base_line.set_data(sx, sy)
        spring_robot_line.set_data(sx2, sy2)

        for g in range(GHOST_STEPS):
            idx_vis = anim_frame_idx - (g + 1)
            if idx_vis >= 0:
                i_ghost = int(sample_idx[idx_vis])
                alpha = GHOST_ALPHA_MAX * (1.0 - (g + 1) / (GHOST_STEPS + 1))
                _update_patch_pos(ghost_base_patches[g], float(xbs[i_ghost]), LANE_Y)
                _update_patch_pos(ghost_robot_patches[g], float(xees[i_ghost]), LANE_Y)
                ghost_base_patches[g].set_alpha(alpha)
                ghost_robot_patches[g].set_alpha(alpha)
            else:
                ghost_base_patches[g].set_alpha(0.0)
                ghost_robot_patches[g].set_alpha(0.0)

        _update_patch_pos(base_patch, x_base, LANE_Y)
        _update_patch_pos(robot_patch, x_robot, LANE_Y)

        t_cur = float(ts[i])
        u_cur = float(us[i])
        state_text.set_text(
            f"t={t_cur:.2f}s   xb={xb:.3f}m   xr={xr:.3f}m   xee={float(xees[i]):.4f}m   u(t)={u_cur:.3f}"
        )

        pos_line.set_data(ts[:i + 1], xees[:i + 1])
        pos_cursor.set_xdata([t_cur, t_cur])
        ctrl_line.set_data(ts[:i + 1], us[:i + 1])
        ctrl_cursor.set_xdata([t_cur, t_cur])

        slider_frame.eventson = False
        slider_frame.set_val(anim_frame_idx)
        slider_frame.eventson = True

        return spring_base_line, spring_robot_line, pos_line, pos_cursor, ctrl_line, ctrl_cursor, state_text

    def on_slider(_val):
        state["frame"] = int(slider_frame.val)
        draw_frame(state["frame"])
        fig.canvas.draw_idle()

    def on_play_pause(_event):
        state["playing"] = not state["playing"]
        btn_pp.label.set_text('⏸ Pause' if state["playing"] else '▶ Play')
        fig.canvas.draw_idle()

    def on_rewind(_event):
        state["frame"] = 0
        state["playing"] = False
        btn_pp.label.set_text('▶ Play')
        draw_frame(0)
        fig.canvas.draw_idle()

    def on_back(_event):
        state["frame"] = max(0, state["frame"] - 10)
        state["playing"] = False
        btn_pp.label.set_text('▶ Play')
        draw_frame(state["frame"])
        fig.canvas.draw_idle()

    def on_fwd(_event):
        state["frame"] = min(n_anim_frames - 1, state["frame"] + 10)
        state["playing"] = False
        btn_pp.label.set_text('▶ Play')
        draw_frame(state["frame"])
        fig.canvas.draw_idle()

    slider_frame.on_changed(on_slider)
    btn_pp.on_clicked(on_play_pause)
    btn_rw.on_clicked(on_rewind)
    btn_back.on_clicked(on_back)
    btn_fwd.on_clicked(on_fwd)

    draw_frame(0)

    def update(_tick):
        if not state["playing"]:
            return draw_frame(state["frame"])
        artists = draw_frame(state["frame"])
        if state["frame"] >= n_anim_frames - 1:
            state["playing"] = False
            btn_pp.label.set_text('▶ Play')
            return artists
        state["frame"] += 1
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=n_anim_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )

    if save_gif:
        gif_path = "noise_visualizer.gif"
        print(f"[visualizer] Salvo GIF in: {gif_path}...")
        anim.save(gif_path, writer='pillow', fps=DISPLAY_FPS)
        print("[visualizer] GIF salvata.")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualizer animato: rumore (OU o Coloured) → sistema MSD")
    parser.add_argument("--speed", "-s", type=int, default=1,
                        help="Velocità visiva relativa (1 = real-time circa)")
    parser.add_argument("--gif", action="store_true", help="Salva l'animazione come GIF")
    parser.add_argument("--noise-type", type=str, default=None, choices=["ou", "ou_no_sigma", "coloured"],
                        help=f"Tipo di rumore (default: {NOISE_TYPE})")

    args = parser.parse_args()

    if args.noise_type is not None:
        NOISE_TYPE = args.noise_type

    noise_type_str = "OU" if NOISE_TYPE == "ou" else ("OU no sigma" if NOISE_TYPE == "ou_no_sigma" else f"Coloured (1/f^{COLOURED_BETA})")
    print("=== Parametri simulazione ===")
    print(f"SEED = {SEED}")
    print(f"DT = {DT}")
    print(f"T_MAX = {T_MAX}")
    print(f"DISPLAY_FPS = {DISPLAY_FPS}")
    print()
    print(f"=== Tipo rumore: {noise_type_str} ===")
    if NOISE_TYPE in ["ou", "ou_no_sigma"]:
        print(f"THETA = {THETA}")
        print(f"MU = {MU}")
        print(f"SIGMA_OU = {SIGMA_OU}")
    else:
        print(f"COLOURED_BETA = {COLOURED_BETA}")
    print()
    print("=== Parametri rumore generale ===")
    print(f"NOISE_START = {NOISE_START}")
    print(f"NOISE_DECAY = {NOISE_DECAY}")
    print(f"NOISE_MIN = {NOISE_MIN}")
    print()
    print("=== Azione ===")
    print(f"BASE_ACTION = {BASE_ACTION}")
    print(f"U_MIN = {U_MIN}")
    print(f"U_MAX = {U_MAX}")
    print()
    print(f"Il rumore {noise_type_str} viene usato come input di controllo per il sistema MSD.")
    if NOISE_TYPE == "ou":
        print("Formula OU standard: action_noisy = clip(base_action + sigma_episode * x_t, U_MIN, U_MAX)")
        print("con x_{t+1} = x_t + theta*dt*(mu - x_t) + sigma_ou*sqrt(dt)*N(0,1)")
    elif NOISE_TYPE == "ou_no_sigma":
        print("Formula OU no sigma: action_noisy = clip(base_action + x_t, U_MIN, U_MAX)")
        print("con x_{t+1} = x_t + theta*dt*(mu - x_t) + sigma_episode*sqrt(dt)*N(0,1)")
    else:
        print("Formula coloured: action_noisy = clip(base_action + sigma_episode*coloured_noise, U_MIN, U_MAX)")
    print("La visualizzazione usa frame precomputati per mantenere il tempo coerente.")
    print()

    run_animation(speed=args.speed, save_gif=args.gif)
