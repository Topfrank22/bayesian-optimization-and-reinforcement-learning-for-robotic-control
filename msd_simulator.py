"""
Simulatore sistema robot-base compliante (Roveda et al., Mechatronics 2016)
============================================================================
Equazioni del moto (sistema accoppiato, senza environment):

  Eq.1 (base):   Mr*(xr_ddot + xb_ddot) + Mb*xb_ddot + Db*xb_dot + Kb*(xb - xb0) = 0
  Eq.2 (robot):  Mr*(xr_ddot + xb_ddot) + Dr*xr_dot + Kr*(xr - xr0(t))          = 0

Dove:
  xb  = posizione assoluta della base compliante
  xr  = posizione RELATIVA del robot rispetto alla base
  xee = xb + xr = posizione ASSOLUTA end-effector
  xr0 = setpoint di comando del robot (input: chirp o HOLD)
  xb0 = posizione di equilibrio della base (= 0)

Parametri dal paper (Section 4.1, Roveda et al. 2016):
  Mb = 70 kg,   Kb = 4000 N/m,  zeta_b = 0.25  =>  Db = 2*sqrt(Kb*Mb)*zeta_b
  Mr = 2.5 kg,  Kr = 1500 N/m,  zeta_r = 0.30  =>  Dr = 2*sqrt(Kr*Mr)*zeta_r
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════
# SEZIONE 1 — PARAMETRI SISTEMA (dal paper, Section 4.1)
# ═══════════════════════════════════════════════════════════════
Mb     = 70.0          # [kg]   massa base compliante
Kb     = 4000.0        # [N/m]  rigidezza base
zeta_b = 0.25          # [-]    smorzamento base
Db     = 2 * np.sqrt(Kb * Mb) * zeta_b   # [N·s/m]

Mr     = 2.5           # [kg]   massa robot (Cartesian impedance)
Kr     = 1500.0        # [N/m]  rigidezza robot
zeta_r = 0.30          # [-]    smorzamento robot
Dr     = 2 * np.sqrt(Kr * Mr) * zeta_r   # [N·s/m]

xb0    = 0.0           # [m]    equilibrio base

X_MAX_ROBOT = 2.0      # [m]    ampiezza massima spostamento robot

print("=" * 55)
print("PARAMETRI SISTEMA")
print(f"  Db = {Db:.2f} N·s/m")
print(f"  Dr = {Dr:.2f} N·s/m")
print(f"  X_MAX_ROBOT = {X_MAX_ROBOT} m")

# ═══════════════════════════════════════════════════════════════
# SEZIONE 2 — PARAMETRI INPUT CHIRP
# ═══════════════════════════════════════════════════════════════
F0_CHIRP     = 1     # [Hz]   frequenza iniziale chirp
CHIRP_RATE   = 0.05    # [Hz/s] pendenza (rate di crescita frequenza)
AMP_FRACTION = 1     # [-]    ampiezza come frazione di X_MAX_ROBOT

AMP_CHIRP = AMP_FRACTION * X_MAX_ROBOT   # [m] ampiezza effettiva (calcolata)

print("\nPARAMETRI CHIRP")
print(f"  F0_CHIRP     = {F0_CHIRP} Hz")
print(f"  CHIRP_RATE   = {CHIRP_RATE} Hz/s")
print(f"  AMP_FRACTION = {AMP_FRACTION}  →  AMP_CHIRP = {AMP_CHIRP} m")

# ═══════════════════════════════════════════════════════════════
# SEZIONE 3 — TARGET
# ═══════════════════════════════════════════════════════════════
GAINED_DISTANCE = 0.3  # [m]  distanza guadagnata oltre la metà corsa robot
X_TARGET = X_MAX_ROBOT / 2 + GAINED_DISTANCE   # [m] posizione assoluta target

print("\nTARGET")
print(f"  GAINED_DISTANCE = {GAINED_DISTANCE} m")
print(f"  X_TARGET = X_MAX_ROBOT/2 + GAINED_DISTANCE = {X_TARGET} m")

# ═══════════════════════════════════════════════════════════════
# SEZIONE 4 — CONDIZIONI DI STOP
# ═══════════════════════════════════════════════════════════════
T_MAX         = 200.0  # [s]    tempo limite simulazione
V_ZERO_THRESH = 0.01   # [m/s]  soglia velocità "quasi zero" (x_ee_dot)
T_ZERO_VEL    = 10.0   # [s]    durata continua sotto soglia → stop
T_STAGNANT    = 50.0   # [s]    durata stagnazione max(x_ee) → stop
T_POST_TARGET = 10.0   # [s]    stop dopo raggiungimento target

DT = 0.001             # [s]    passo di campionamento

print("\nCONDIZIONI DI STOP")
print(f"  T_MAX         = {T_MAX} s")
print(f"  V_ZERO_THRESH = {V_ZERO_THRESH} m/s  per {T_ZERO_VEL} s")
print(f"  T_STAGNANT    = {T_STAGNANT} s")
print(f"  T_POST_TARGET = {T_POST_TARGET} s (dopo raggiungimento target)")
print("=" * 55)

# ═══════════════════════════════════════════════════════════════
# SEZIONE 5 — COSTRUZIONE CHIRP (vettore lungo T_MAX)
# ═══════════════════════════════════════════════════════════════
t_chirp = np.arange(0, T_MAX + DT, DT)
# Frequenza istantanea: f(t) = F0_CHIRP + CHIRP_RATE * t
# Fase istantanea: phi(t) = 2*pi*(F0_CHIRP*t + 0.5*CHIRP_RATE*t^2)
phase = 2 * np.pi * (F0_CHIRP * t_chirp + 0.5 * CHIRP_RATE * t_chirp**2)
chirp_signal = (AMP_CHIRP / 2) * np.sin(phase)

# Interpolatore continuo del chirp (usato nell'ODE)
chirp_func = interp1d(t_chirp, chirp_signal, kind='linear', fill_value='extrapolate')

# ═══════════════════════════════════════════════════════════════
# SEZIONE 6 — INTEGRAZIONE A STEP CON CONTROLLO STOP
# ═══════════════════════════════════════════════════════════════

# Stato iniziale: tutto a riposo
z0 = [0.0, 0.0, 0.0, 0.0]   # [xb, xb_dot, xr, xr_dot]

# Variabili di stato della simulazione
target_reached    = False
t_target          = None
xr0_hold          = None   # valore congelato del setpoint al momento del target

t_vel_zero_start  = None   # inizio intervallo velocità quasi zero
xee_max_ever      = 0.0    # massimo storico di x_ee
t_stagnant_start  = None   # inizio intervallo stagnazione

stop_reason = "T_MAX raggiunto"

# Liste risultati
t_list   = [0.0]
xb_list  = [z0[0]]
xr_list  = [z0[2]]
xr0_list = [chirp_func(0.0)]

# Funzione ODE (usa xr0_current aggiornato esternamente)
def msd_coupled(t, z, xr0_val):
    xb, xb_dot, xr, xr_dot = z
    M_mat = np.array([[Mr + Mb, Mr],
                      [Mr,      Mr]])
    rhs = np.array([
        -Db * xb_dot - Kb * (xb - xb0),
        -Dr * xr_dot - Kr * (xr - xr0_val)
    ])
    accs = np.linalg.solve(M_mat, rhs)
    return [xb_dot, accs[0], xr_dot, accs[1]]

print("\nIntegrazione in corso (step-by-step)...")

z_current = np.array(z0, dtype=float)
t_current = 0.0
STEP_SIZE = 0.1   # [s] blocco di integrazione per ogni iterazione di controllo

while t_current < T_MAX:
    t_next = min(t_current + STEP_SIZE, T_MAX)
    t_span_block = (t_current, t_next)
    t_eval_block = np.arange(t_current, t_next + DT, DT)
    t_eval_block = t_eval_block[t_eval_block <= t_next]

    # Determina xr0: HOLD o chirp
    if target_reached:
        xr0_val = xr0_hold
        xr0_func_block = lambda t: xr0_hold
    else:
        xr0_val = None  # viene calcolato dentro l'ODE per ogni t

    # Integrazione del blocco
    sol = solve_ivp(
        lambda t, z: msd_coupled(t, z, xr0_hold if target_reached else chirp_func(t)),
        t_span=t_span_block,
        y0=z_current,
        t_eval=t_eval_block,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )

    # Salva output (escludi il primo punto già salvato)
    skip = 1 if len(t_list) > 1 else 0
    t_list.extend(sol.t[skip:].tolist())
    xb_list.extend(sol.y[0][skip:].tolist())
    xr_list.extend(sol.y[2][skip:].tolist())

    for ti in sol.t[skip:]:
        xr0_list.append(xr0_hold if target_reached else float(chirp_func(ti)))

    z_current = sol.y[:, -1]
    t_current = sol.t[-1]

    # ── Calcola grandezze per le condizioni di stop ──
    xb_now  = z_current[0]
    xbdot   = z_current[1]
    xr_now  = z_current[2]
    xrdot   = z_current[3]
    xee_now = xb_now + xr_now
    xee_dot = xbdot + xrdot

    # ── Check target ──
    if not target_reached and xee_now >= X_TARGET:
        target_reached = True
        t_target       = t_current
        xr0_hold       = float(chirp_func(t_current))
        print(f"  >>> TARGET RAGGIUNTO a t = {t_target:.3f} s  (x_ee = {xee_now:.4f} m)")

    # ── Condizioni di stop ──
    if target_reached:
        if t_current >= t_target + T_POST_TARGET:
            stop_reason = f"T_POST_TARGET ({T_POST_TARGET}s dopo il target)"
            break
    else:
        # Velocità quasi zero
        if abs(xee_dot) < V_ZERO_THRESH:
            if t_vel_zero_start is None:
                t_vel_zero_start = t_current
            elif t_current - t_vel_zero_start >= T_ZERO_VEL:
                stop_reason = f"Velocità quasi zero per {T_ZERO_VEL}s"
                break
        else:
            t_vel_zero_start = None

        # Stagnazione max(x_ee)
        if xee_now > xee_max_ever:
            xee_max_ever     = xee_now
            t_stagnant_start = t_current
        elif t_stagnant_start is not None and (t_current - t_stagnant_start) >= T_STAGNANT:
            stop_reason = f"max(x_ee) stagnante per {T_STAGNANT}s"
            break

print(f"Integrazione terminata. Motivo stop: {stop_reason}")

# ═══════════════════════════════════════════════════════════════
# SEZIONE 7 — KPI
# ═══════════════════════════════════════════════════════════════
t_out   = np.array(t_list)
xb_out  = np.array(xb_list)
xr_out  = np.array(xr_list)
xr0_out = np.array(xr0_list)
xee_out = xb_out + xr_out

xee_max_final  = np.max(xee_out)
kpi_accuracy   = xee_max_final - X_TARGET
kpi_time2tgt   = t_target if target_reached else None

print("\n" + "=" * 55)
print("KPI")
print(f"  X_TARGET           = {X_TARGET:.4f} m")
print(f"  max(x_ee) raggiunto= {xee_max_final:.4f} m")
print(f"  Accuracy (err)     = {kpi_accuracy:+.4f} m  (ideale → 0)")
if kpi_time2tgt is not None:
    print(f"  Time to target     = {kpi_time2tgt:.3f} s")
else:
    print(f"  Time to target     = N/A (target non raggiunto)")
print(f"  Stop reason        = {stop_reason}")
print("=" * 55)

# ═══════════════════════════════════════════════════════════════
# SEZIONE 8 — PLOT
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

# ── Subplot 1: Setpoint chirp vs posizione relativa robot ──
axes[0].plot(t_out, xr0_out, 'k--', lw=1.2, label='$x_r^0(t)$ - setpoint (chirp / HOLD)')
axes[0].plot(t_out, xr_out,  'b',   lw=1.5, label='$x_r(t)$ - posizione relativa robot')
if target_reached:
    axes[0].axvline(t_target, color='orange', lw=1.2, ls=':', label=f'Target raggiunto (t={t_target:.2f}s)')
axes[0].set_ylabel('Posizione [m]')
axes[0].set_title('Posizione relativa robot vs setpoint')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.4)

# ── Subplot 2: Deformazione base compliante ──
axes[1].plot(t_out, xb_out * 1e3, 'r', lw=1.5, label='$x_b(t)$ - deformazione base')
if target_reached:
    axes[1].axvline(t_target, color='orange', lw=1.2, ls=':')
axes[1].set_ylabel('Posizione [mm]')
axes[1].set_title('Deformazione base compliante')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.4)

# ── Subplot 3: Posizione assoluta end-effector + target ──
axes[2].plot(t_out, xee_out, 'g', lw=1.5, label='$x_{ee}(t) = x_b + x_r$ - end-effector assoluto')
axes[2].axhline(X_TARGET, color='red', lw=1.5, ls='--',
                label=f'$X_{{target}}$ = {X_TARGET:.3f} m  (X_MAX/2 + {GAINED_DISTANCE} m)')
if target_reached:
    axes[2].axvline(t_target, color='orange', lw=1.2, ls=':',
                    label=f'Target raggiunto (t={t_target:.2f}s)')
axes[2].set_ylabel('Posizione [m]')
axes[2].set_xlabel('Tempo [s]')
axes[2].set_title('Posizione assoluta end-effector')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.4)

# ── Testo KPI nel grafico ──
kpi_str  = f"KPI\n"
kpi_str += f"max(x_ee)  = {xee_max_final:.4f} m\n"
kpi_str += f"Accuracy   = {kpi_accuracy:+.4f} m\n"
if kpi_time2tgt is not None:
    kpi_str += f"Time2Tgt   = {kpi_time2tgt:.2f} s\n"
else:
    kpi_str += f"Time2Tgt   = N/A\n"
kpi_str += f"Stop: {stop_reason}"

axes[2].text(0.01, 0.97, kpi_str, transform=axes[2].transAxes,
             fontsize=7.5, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

# Frequenza istantanea finale del chirp
f_final = F0_CHIRP + CHIRP_RATE * t_out[-1]
plt.suptitle(
    f'Simulazione robot-base compliante (Roveda et al. 2016)\n'
    f'Chirp: F0={F0_CHIRP} Hz, rate={CHIRP_RATE} Hz/s | '
    f'Ampiezza={AMP_FRACTION}·X_MAX={AMP_CHIRP:.2f} m | '
    f'Stop: {stop_reason}',
    fontsize=10
)
plt.tight_layout()
plt.savefig('ROBOT/msd_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot salvato in ROBOT/msd_simulation.png")
