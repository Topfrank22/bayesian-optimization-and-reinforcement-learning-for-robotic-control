"""
interactive_game.py
===================
Gioco interattivo per il sistema MSD: comanda il robot in tempo reale
tramite uno slider xr0 ∈ [-1, 1] → scalato a [-1 m, +1 m] (X_MAX_ROBOT=1.0).

CAMPO VISIVO: -3 m (sinistra) → +7 m (destra)

CARATTERISTICHE:
  - Simulazione in tempo reale con Euler semiimplicito (dt = 0.02 s)
  - Slider di controllo: xr0 ∈ [-1, 1]  (1:1 in metri)
  - Linea tratteggiata rossa = record di distanza raggiunto
  - Grafici in tempo reale: posizione xee e setpoint xr0
  - Bottone Reset per ricominciare da capo

USO:
    python ROBOT/interactive_game.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ═══════════════════════════════════════════════════════════════
# PARAMETRI FISICI
# ═══════════════════════════════════════════════════════════════
Mb     = 70.0
Kb     = 4000.0
zeta_b = 0.25
Mr     = 2.5
Kr     = 1500.0
zeta_r = 0.30
xb0    = 0.0

Db = 2 * np.sqrt(Kb * Mb) * zeta_b
Dr = 2 * np.sqrt(Kr * Mr) * zeta_r

# Slider ±1 corrisponde a ±1 m reale
X_MAX_ROBOT = 1.0   # massimo setpoint (m)  ← modificato
DT          = 0.02  # passo di integrazione (s)
HISTORY_LEN = 600   # n. step di storia nei grafici

# ═══════════════════════════════════════════════════════════════
# GRAFICA
# ═══════════════════════════════════════════════════════════════
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
COL_RECORD = '#e03030'
COL_CTRL   = '#b36000'
COL_WALL   = '#555555'

GHOST_STEPS     = 4
GHOST_ALPHA_MAX = 0.30

# Campo visivo: -3 m sinistra, +7 m destra
VIS_LEFT  = -3.0
VIS_RIGHT =  7.0   # ← modificato

# ═══════════════════════════════════════════════════════════════
# HELPER GRAFICI
# ═══════════════════════════════════════════════════════════════
def _spring_path_h(x_start, x_end, y_center=0.5):
    n_pts = N_COILS * 4 + 2
    xs = np.linspace(x_start, x_end, n_pts)
    ys = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        phase = (i - 1) / (n_pts - 2) * N_COILS * 2 * np.pi
        ys[i] = SPRING_W * np.sin(phase)
    ys += y_center
    return xs, ys


def _make_block(ax, x_center, y_center, color, alpha=1.0):
    bw = BLOCK_H * 1.5
    bh = BLOCK_W * 0.45
    p = mpatches.FancyBboxPatch(
        (x_center - bw / 2, y_center - bh / 2), bw, bh,
        boxstyle="round,pad=0.005",
        linewidth=1.2,
        edgecolor='#444444',
        facecolor=color,
        zorder=4,
        alpha=alpha,
        transform=ax.transData,
    )
    return p


# ═══════════════════════════════════════════════════════════════
# STATO SIMULAZIONE
# ═══════════════════════════════════════════════════════════════
class SimState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.xb      = 0.0
        self.xb_dot  = 0.0
        self.xr      = 0.0
        self.xr_dot  = 0.0
        self.t       = 0.0
        self.xr0     = 0.0
        self.xee_record = 0.0
        self.t_hist   = []
        self.xee_hist = []
        self.u_hist   = []

    def step(self, xr0_val):
        """Integrazione Euler semi-implicito (DT=0.02 s)."""
        xb, xb_dot, xr, xr_dot = self.xb, self.xb_dot, self.xr, self.xr_dot

        Fb = -Db * xb_dot - Kb * (xb - xb0)
        Fr = -Dr * xr_dot - Kr * (xr - xr0_val)

        M   = np.array([[Mr + Mb, Mr], [Mr, Mr]])
        rhs = np.array([Fb, Fr])
        xb_ddot, xr_ddot = np.linalg.solve(M, rhs)

        self.xb_dot += xb_ddot * DT
        self.xr_dot += xr_ddot * DT
        self.xb     += self.xb_dot * DT
        self.xr     += self.xr_dot * DT
        self.t      += DT
        self.xr0     = xr0_val

        xee = self.xb + self.xr
        if xee > self.xee_record:
            self.xee_record = xee

        self.t_hist.append(self.t)
        self.xee_hist.append(xee)
        self.u_hist.append(xr0_val)

        if len(self.t_hist) > HISTORY_LEN:
            self.t_hist   = self.t_hist[-HISTORY_LEN:]
            self.xee_hist = self.xee_hist[-HISTORY_LEN:]
            self.u_hist   = self.u_hist[-HISTORY_LEN:]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    sim = SimState()

    fig = plt.figure(figsize=(14, 9.5), facecolor=FIG_BG)
    fig.suptitle(
        "MSD Interactive Game  —  slider xr0 ∈ [-1, 1]  →  ±1 m setpoint robot",
        color=TEXT_CLR, fontsize=11, fontweight='bold'
    )

    LANE_Y = 0.5

    gs = gridspec.GridSpec(
        2, 2,
        left=0.06, right=0.97,
        top=0.92,  bottom=0.22,
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

    # ── pannello MSD ──
    ax_msd.set_xlim(VIS_LEFT, VIS_RIGHT)
    ax_msd.set_ylim(0.0, 1.0)
    ax_msd.set_yticks([])
    ax_msd.set_xlabel('Posizione (m)', color=TEXT_MUTED, fontsize=9)
    ax_msd.set_title(
        f'Sistema MSD  —  Controllo interattivo  |  campo visivo [{VIS_LEFT} m, {VIS_RIGHT} m]',
        color=TEXT_CLR, fontsize=10, fontweight='bold'
    )

    ax_msd.axvline(0.0, color='#888888', linewidth=2.0, zorder=1)
    ax_msd.fill_betweenx([0, 1], VIS_LEFT, 0.0, color='#e8e8e8', alpha=0.5, zorder=0)
    ax_msd.text(VIS_LEFT + 0.10, 0.08, 'SUOLO', color='#999999', fontsize=8, va='bottom')

    ax_msd.axvline(VIS_LEFT,  color=COL_WALL, linewidth=1.2, linestyle=':', alpha=0.5)
    ax_msd.axvline(VIS_RIGHT, color=COL_WALL, linewidth=1.2, linestyle=':', alpha=0.5)

    ax_msd.axhline(LANE_Y, color='#cccccc', linewidth=1.0, zorder=1)

    # linea record
    record_line = ax_msd.axvline(0.0, color=COL_RECORD, linewidth=2.0,
                                  linestyle='--', zorder=2, alpha=0.9)
    record_text = ax_msd.text(0.02, 0.78, 'Record: 0.000 m',
                               color=COL_RECORD, fontsize=8.5,
                               va='top', fontweight='bold',
                               transform=ax_msd.transAxes)

    spring_base_line,  = ax_msd.plot([], [], color=COL_KB, linewidth=2.2, zorder=3)
    spring_robot_line, = ax_msd.plot([], [], color=COL_KR, linewidth=2.2, zorder=3)

    ghost_patches  = []
    base_patch     = [None]
    robot_patch    = [None]
    xr0_marker     = [None]

    state_text = ax_msd.text(
        VIS_LEFT + 0.10, 0.97, '',
        color=TEXT_CLR, fontsize=8, va='top',
        fontfamily='monospace', zorder=10,
        transform=ax_msd.transData,
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COL_KB,    linewidth=2, label='Molla base (Kb)'),
        Line2D([0], [0], color=COL_KR,    linewidth=2, label='Molla robot (Kr)'),
        mpatches.Patch(facecolor=COL_BASE,  label='Massa base (Mb)'),
        mpatches.Patch(facecolor=COL_ROBOT, label='Massa robot (Mr)'),
        Line2D([0], [0], color=COL_CTRL, linewidth=0,
               marker='v', markersize=8, label='Setpoint xr0'),
        Line2D([0], [0], color=COL_RECORD, linewidth=2,
               linestyle='--', label='Record xee'),
    ]
    ax_msd.legend(handles=legend_elements, loc='lower right',
                  fontsize=7, facecolor='white', framealpha=0.8)

    # ── pannello posizione ──
    ax_pos.set_xlabel('Tempo (s)', fontsize=8)
    ax_pos.set_ylabel('xb + xr  (m)', fontsize=8)
    ax_pos.set_title('Posizione punta (xee)', fontsize=9, fontweight='bold')
    pos_line, = ax_pos.plot([], [], color=COL_KB, linewidth=1.8)
    record_line_pos = ax_pos.axhline(0.0, color=COL_RECORD, linewidth=1.2,
                                      linestyle='--', alpha=0.8)

    # ── pannello setpoint ──
    ax_ctrl.set_ylim(-X_MAX_ROBOT * 1.15, X_MAX_ROBOT * 1.15)
    ax_ctrl.set_xlabel('Tempo (s)', fontsize=8)
    ax_ctrl.set_ylabel('xr0  (m)', fontsize=8)
    ax_ctrl.set_title(f'Setpoint robot (xr0)  [±{X_MAX_ROBOT} m]',
                      fontsize=9, fontweight='bold')
    ax_ctrl.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax_ctrl.axhline( X_MAX_ROBOT, color='#ddaaaa', linewidth=0.8, linestyle=':')
    ax_ctrl.axhline(-X_MAX_ROBOT, color='#ddaaaa', linewidth=0.8, linestyle=':')
    ctrl_line, = ax_ctrl.plot([], [], color=COL_CTRL, linewidth=1.8)

    # ── WIDGETS ──
    BTN_CLR = '#e8edf2'
    BTN_HOV = '#cfd8e3'

    ax_slider_ctrl = fig.add_axes([0.02, 0.22, 0.025, 0.70], facecolor='#f0f0f0')
    slider_xr0 = Slider(
        ax_slider_ctrl, 'xr0', -1.0, 1.0,
        valinit=0.0, orientation='vertical',
        color=COL_ROBOT,
    )
    slider_xr0.label.set_color(TEXT_CLR)
    slider_xr0.label.set_fontsize(9)
    slider_xr0.valtext.set_color(TEXT_CLR)
    slider_xr0.valtext.set_fontsize(7)

    ax_reset = fig.add_axes([0.44, 0.05, 0.12, 0.05])
    btn_reset = Button(ax_reset, '⟳  Reset', color=BTN_CLR, hovercolor=BTN_HOV)
    btn_reset.label.set_fontsize(10)
    btn_reset.label.set_color(TEXT_CLR)

    # ── storia ghost ──
    xb_history     = []
    xr_abs_history = []

    # ── funzione disegno ──
    def draw(frame):
        # slider ±1 → ±1 m (X_MAX_ROBOT = 1.0)
        xr0_val = float(slider_xr0.val) * X_MAX_ROBOT

        sim.step(xr0_val)

        xb      = sim.xb
        xr      = sim.xr
        x_base  = xb
        x_robot = xb + xr

        xb_history.append(x_base)
        xr_abs_history.append(x_robot)
        if len(xb_history) > GHOST_STEPS * 3:
            xb_history.pop(0)
            xr_abs_history.pop(0)

        bw_half = BLOCK_H * 0.75
        sx,  sy  = _spring_path_h(0.0,              x_base  - bw_half, LANE_Y)
        sx2, sy2 = _spring_path_h(x_base + bw_half, x_robot - bw_half, LANE_Y)
        spring_base_line.set_data(sx, sy)
        spring_robot_line.set_data(sx2, sy2)

        for gp in ghost_patches:
            try: gp.remove()
            except Exception: pass
        ghost_patches.clear()

        n_hist = len(xb_history)
        stride = max(1, n_hist // GHOST_STEPS)
        for g_idx, h_idx in enumerate(range(0, n_hist - stride, stride)):
            alpha = GHOST_ALPHA_MAX * (g_idx + 1) / (GHOST_STEPS + 1)
            gb = _make_block(ax_msd, xb_history[h_idx],     LANE_Y, COL_BASE,  alpha)
            gr = _make_block(ax_msd, xr_abs_history[h_idx], LANE_Y, COL_ROBOT, alpha)
            ax_msd.add_patch(gb); ax_msd.add_patch(gr)
            ghost_patches.append(gb); ghost_patches.append(gr)

        for p_ in [base_patch[0], robot_patch[0]]:
            if p_ is not None:
                try: p_.remove()
                except Exception: pass
        bp = _make_block(ax_msd, x_base,  LANE_Y, COL_BASE)
        rp = _make_block(ax_msd, x_robot, LANE_Y, COL_ROBOT)
        ax_msd.add_patch(bp); ax_msd.add_patch(rp)
        base_patch[0]  = bp
        robot_patch[0] = rp

        if xr0_marker[0] is not None:
            try: xr0_marker[0].remove()
            except Exception: pass
        marker, = ax_msd.plot(
            xr0_val, LANE_Y + 0.18, 'v',
            color=COL_CTRL, markersize=10, zorder=5,
        )
        xr0_marker[0] = marker

        record_line.set_xdata([sim.xee_record, sim.xee_record])
        record_text.set_text(f'Record: {sim.xee_record:.3f} m')

        state_text.set_text(
            f"t={sim.t:.1f}s   xb={xb:.3f}m   xr={xr:.3f}m   "
            f"xee={x_robot:.3f}m   xr0={xr0_val:.3f}m   "
            f"record={sim.xee_record:.3f}m"
        )

        t_arr   = sim.t_hist
        xee_arr = sim.xee_hist
        u_arr   = sim.u_hist

        pos_line.set_data(t_arr, xee_arr)
        ctrl_line.set_data(t_arr, u_arr)
        record_line_pos.set_ydata([sim.xee_record, sim.xee_record])

        if len(t_arr) > 1:
            t_min, t_max = t_arr[0], t_arr[-1]
            t_pad = max(0.5, (t_max - t_min) * 0.05)
            ax_pos.set_xlim(t_min - t_pad, t_max + t_pad)
            ax_ctrl.set_xlim(t_min - t_pad, t_max + t_pad)

            xee_min = min(xee_arr)
            xee_max = max(max(xee_arr), sim.xee_record)
            y_pad = max(0.05, (xee_max - xee_min) * 0.15)
            ax_pos.set_ylim(xee_min - y_pad, xee_max + y_pad)

        fig.canvas.draw_idle()

    def on_reset(event):
        sim.reset()
        xb_history.clear()
        xr_abs_history.clear()
        slider_xr0.set_val(0.0)
        ghost_patches.clear()
        pos_line.set_data([], [])
        ctrl_line.set_data([], [])
        record_line.set_xdata([0.0, 0.0])
        record_line_pos.set_ydata([0.0, 0.0])
        record_text.set_text('Record: 0.000 m')
        fig.canvas.draw_idle()

    btn_reset.on_clicked(on_reset)

    anim = FuncAnimation(
        fig, draw,
        interval=int(DT * 1000),
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
