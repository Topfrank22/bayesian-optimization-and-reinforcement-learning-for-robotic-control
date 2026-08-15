"""
Core simulator for Bayesian Optimization.

This module exposes a single public function:
    run_simulation(F0_CHIRP, CHIRP_RATE, AMP_FRACTION, sim_params=None)

Design notes for BO usage:
- no plots
- no verbose prints
- lighter numerical settings than msd_simulator.py on purpose
- STEP_SIZE and DT removed: a single solve_ivp call over [0, T_MAX] is used.
  RK45 manages its own adaptive step internally; DT was only controlling
  output sampling density, which is now handled by max_step.
- RTOL/ATOL tolerances are relaxed because BO needs many runs,
  so execution speed matters more than plotting-grade trajectories.
"""

import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_PARAMS = {
    # System
    "Mb": 70.0,
    "Kb": 4000.0,
    "zeta_b": 0.25,
    "Mr": 2.5,
    "Kr": 1500.0,
    "zeta_r": 0.30,
    "xb0": 0.0,
    "X_MAX_ROBOT": 1.0,
    # Target
    "GAINED_DISTANCE": 1.0,
    # Stop conditions
    "T_MAX": 20.0,
    "V_ZERO_THRESH": 0.01,
    "T_ZERO_VEL": 10.0,
    "T_STAGNANT": 50.0,
    "T_POST_TARGET": 10.0,
    # Integration settings
    # RK45 uses its own adaptive step internally.
    # MAX_STEP caps the internal step to ensure stop conditions
    # (target crossing, stagnation, near-zero velocity) are not missed.
    "MAX_STEP": 0.05,
    "RTOL": 1e-5,
    "ATOL": 1e-7,
}


def _build_params(sim_params=None):
    p = DEFAULT_PARAMS.copy()
    if sim_params is not None:
        p.update(sim_params)

    p["Db"] = 2 * np.sqrt(p["Kb"] * p["Mb"]) * p["zeta_b"]
    p["Dr"] = 2 * np.sqrt(p["Kr"] * p["Mr"]) * p["zeta_r"]
    p["X_TARGET"] = p["X_MAX_ROBOT"] / 2 + p["GAINED_DISTANCE"]
    return p


def _chirp_value(t, f0, chirp_rate, amp_chirp):
    """Chirp sinusoidale con ampiezza piena.
    amp_chirp = AMP_FRACTION * X_MAX_ROBOT, quindi u oscilla in [-amp_chirp, +amp_chirp].
    Con AMP_FRACTION=1.0 e X_MAX_ROBOT=1.0 si sfrutta tutto il range [-1, +1].
    """
    phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)
    return amp_chirp * np.sin(phase)


def _msd_rhs(t, z, xr0_val, p):
    xb, xb_dot, xr, xr_dot = z

    M_mat = np.array([
        [p["Mr"] + p["Mb"], p["Mr"]],
        [p["Mr"], p["Mr"]],
    ])

    rhs = np.array([
        -p["Db"] * xb_dot - p["Kb"] * (xb - p["xb0"]),
        -p["Dr"] * xr_dot - p["Kr"] * (xr - xr0_val),
    ])

    xb_ddot, xr_ddot = np.linalg.solve(M_mat, rhs)
    return [xb_dot, xb_ddot, xr_dot, xr_ddot]


def run_simulation(F0_CHIRP, CHIRP_RATE, AMP_FRACTION, sim_params=None):
    p = _build_params(sim_params)
    amp_chirp = AMP_FRACTION * p["X_MAX_ROBOT"]

    target_reached = False
    t_target = None
    xr0_hold = None

    t_vel_zero_start = None
    xee_max_ever = 0.0
    t_stagnant_start = 0.0
    stop_reason = "T_MAX reached"

    t_stop = p["T_MAX"]

    def rhs(t, z):
        xr0 = xr0_hold if target_reached else _chirp_value(t, F0_CHIRP, CHIRP_RATE, amp_chirp)
        return _msd_rhs(t, z, xr0, p)

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, p["T_MAX"]),
        y0=[0.0, 0.0, 0.0, 0.0],
        method="RK45",
        max_step=p["MAX_STEP"],
        rtol=p["RTOL"],
        atol=p["ATOL"],
        dense_output=False,
    )

    t_values = sol.t
    xb_arr = sol.y[0]
    xb_dot_arr = sol.y[1]
    xr_arr = sol.y[2]
    xr_dot_arr = sol.y[3]
    xee_arr = xb_arr + xr_arr
    xee_dot_arr = xb_dot_arr + xr_dot_arr

    for k in range(1, len(t_values)):
        ti = t_values[k]
        xee = xee_arr[k]
        xee_dot = xee_dot_arr[k]

        if (not target_reached) and (xee >= p["X_TARGET"]):
            target_reached = True
            t_target = ti
            xr0_hold = _chirp_value(ti, F0_CHIRP, CHIRP_RATE, amp_chirp)

        if target_reached:
            if ti >= t_target + p["T_POST_TARGET"]:
                stop_reason = f"post_target_{p['T_POST_TARGET']}s"
                t_stop = ti
                break
        else:
            if abs(xee_dot) < p["V_ZERO_THRESH"]:
                if t_vel_zero_start is None:
                    t_vel_zero_start = ti
                elif ti - t_vel_zero_start >= p["T_ZERO_VEL"]:
                    stop_reason = f"near_zero_velocity_{p['T_ZERO_VEL']}s"
                    t_stop = ti
                    break
            else:
                t_vel_zero_start = None

            if xee > xee_max_ever:
                xee_max_ever = xee
                t_stagnant_start = ti
            elif ti - t_stagnant_start >= p["T_STAGNANT"]:
                stop_reason = f"stagnation_{p['T_STAGNANT']}s"
                t_stop = ti
                break

    mask = t_values <= t_stop
    xee_max = float(np.max(xee_arr[mask]))
    accuracy = xee_max - p["X_TARGET"]

    return {
        "accuracy": float(accuracy),
        "time_to_target": None if t_target is None else float(t_target),
        "stop_reason": stop_reason,
        "xee_max": xee_max,
    }
