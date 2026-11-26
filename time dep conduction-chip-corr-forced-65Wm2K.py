# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:38:45 2025

@author: VK
"""

# =========================================================
# Time-dependent conduction + steady convection simulation
# Parallel version (Numba) + safe GIF generation
# =========================================================

import matplotlib
matplotlib.use('Agg')      # no GUI windows; safe for Spyder
import numpy as np
import matplotlib.pyplot as plt
import imageio
from numba import njit, prange

# ---------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------
T_inf = 20.0            # ambient air temperature [°C]
rho = 2700.0            # density [kg/m3] (aluminum)
cp = 900.0              # specific heat [J/kg·K]
k = 205.0               # thermal conductivity [W/m·K]
alpha = k / (rho * cp)  # thermal diffusivity [m²/s]
h = 65                 # convection coefficient [W/m²·K]

# Geometry
W_chip = 0.02           # chip width [m]
L_chip = 0.02           # chip length [m]
t_base = 0.003          # base thickness [m]
L_fin = 0.015           # fin height [m]
t_fin = 0.0002          # fin thickness [m]
Nfins = 11              # number of fins
Lx = W_chip             # width of domain
Ly = t_base + L_fin     # vertical height

# Interface resistance (chip ↔ base)
R_int = 2e-6            # m²·K/W

# ---------------------------------------------------------
# Numerical grid
# ---------------------------------------------------------
nx, ny = 40, 80
dx, dy = Lx / nx, Ly / ny
dt = 0.25* min(dx, dy) ** 2 / alpha * 0.9   # stability-limited
t_end = 900
nsteps = int(t_end / dt)
x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)

# ---------------------------------------------------------
# Initial condition
# ---------------------------------------------------------
T = np.ones((ny, nx)) * T_inf
A_chip = W_chip * L_chip

# ---------------------------------------------------------
# Chip power profile [W]
# ---------------------------------------------------------
def chip_power(t):
    if t < 10:
        return 10
    elif t < 20:
        return 50
    elif t < 30:
        return 10
    elif t < 35:
        return 60
    else:
        return 0

# ---------------------------------------------------------
# Parallel conduction step (Numba)
# ---------------------------------------------------------
@njit(parallel=True)
def update_temperature(T_old, T, nx, ny, dx, dy, alpha, dt):
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            d2Tdx2 = (T_old[j, i + 1] - 2*T_old[j, i] + T_old[j, i - 1]) / dx**2
            d2Tdy2 = (T_old[j + 1, i] - 2*T_old[j, i] + T_old[j - 1, i]) / dy**2
            T[j, i] = T_old[j, i] + alpha * dt * (d2Tdx2 + d2Tdy2)
    return T

# ---------------------------------------------------------
# Time loop
# ---------------------------------------------------------
tlist, base_avg, fin_avg, frames = [], [], [], []

for step in range(nsteps):
    t = step * dt
    P = chip_power(t)
    T_old = T.copy()

    # parallel conduction update
    T = update_temperature(T_old, T, nx, ny, dx, dy, alpha, dt)

    # boundary conditions
    q_chip = P / A_chip
    T[0, :] = T[1, :] + q_chip * dy / k                     # bottom heat flux
    T[-1, :] = T_old[-2, :] - (h * dy / k) * (T_old[-1, :] - T_inf)  # top convection
    T[:, 0] = T[:, 1]                                       # adiabatic sides
    T[:, -1] = T[:, -2]

    # record data
    if step % 200 == 0 or step == nsteps - 1:
        tlist.append(t)
        base_avg.append(np.mean(T[0:int(ny*0.1), :]))
        fin_avg.append(np.mean(T[int(ny*0.6):, :]))
        frames.append((T.copy(), t, P))

# ---------------------------------------------------------
# Post-processing
# ---------------------------------------------------------
tlist, base_avg, fin_avg = np.array(tlist), np.array(base_avg), np.array(fin_avg)
t_power = np.linspace(0, t_end, 500)
P_profile = np.array([chip_power(tt) for tt in t_power])

# ---------------------------------------------------------
# 1️⃣ Base temperature + power profile
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(tlist, base_avg, color='tab:red', lw=2, label='Base Temperature')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Base Temperature (°C)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(t_power, P_profile, color='tab:blue', lw=2, linestyle='--', label='Chip Power')
ax2.set_ylabel('Chip Power (W)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
plt.title('Base Temperature and Chip Power vs Time')
fig.tight_layout()
plt.savefig('base_temperature_and_power_vs_time.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------
# 2️⃣ Fin temperature vs time
# ---------------------------------------------------------
fig = plt.figure(figsize=(6,4))
plt.plot(tlist, fin_avg, color='tab:blue', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Fin average temperature (°C)')
plt.title('Fin Temperature vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('fin_temperature_vs_time.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------
# 3️⃣ Base + fin comparison
# ---------------------------------------------------------
fig = plt.figure(figsize=(6,4))
plt.plot(tlist, base_avg, label='Base', color='tab:red', lw=2)
plt.plot(tlist, fin_avg, label='Fin', color='tab:blue', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Base and Fin Temperatures vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('base_fin_temps.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------
# 4️⃣ Safe GIF animation (no open figures)
# ---------------------------------------------------------
vmin = T_inf
vmax = np.max([np.max(f[0]) for f in frames])

with imageio.get_writer('temperature_evolution.gif', mode='I', fps=5) as writer:
    for Tframe, tframe, Pframe in frames:
        fig, ax = plt.subplots(figsize=(3,4), dpi=100)
        im = ax.imshow(Tframe, origin='lower', extent=(0, Lx, 0, Ly),
                       cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(f"t={tframe:.1f}s  P={Pframe:.0f}W")
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='°C')
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        image = buf[:, :, :3].copy()
        writer.append_data(image)
        plt.close(fig)

print("✅ Saved temperature_evolution.gif")

# ---------------------------------------------------------
# 5️⃣ Final temperature field
# ---------------------------------------------------------
fig = plt.figure(figsize=(4,5))
plt.imshow(T, origin='lower', extent=(0, Lx, 0, Ly),
           cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar(label='°C')
plt.title('Final Temperature Field')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('final_temperature_field.png', dpi=150)
plt.close(fig)

print("✅ All plots saved")
print(f"Max final temperature = {np.max(T):.2f} °C (ambient {T_inf:.1f} °C)")
