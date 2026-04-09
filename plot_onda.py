import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Leer datos separados por bloques ---
frames = []
current = []

with open("wave.dat", "r") as f:
    for line in f:
        if line.strip() == "":
            if current:
                frames.append(np.array(current, dtype=float))
                current = []
        else:
            current.append([float(x) for x in line.split()])

# por si el archivo no termina en línea vacía
if current:
    frames.append(np.array(current, dtype=float))

print(f"Frames cargados: {len(frames)}")

# --- Extraer primer frame ---
x = frames[0][:,0]
y = frames[0][:,1]

# rango global de colores (MUY IMPORTANTE)
u_all = np.concatenate([f[:,2] for f in frames])
vmin, vmax = u_all.min(), u_all.max()

# --- Crear figura ---
fig, ax = plt.subplots()
scat = ax.scatter(x, y, c=frames[0][:,2],
                  cmap='viridis',
                  vmin=vmin, vmax=vmax)

plt.colorbar(scat)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

# --- Animación ---
def update(frame):
    u = frames[frame][:,2]
    scat.set_array(u)
    ax.set_title(f"Frame {frame}")
    return scat,

ani = FuncAnimation(fig, update,
                    frames=len(frames),
                    interval=100)

# --- Guardar GIF ---
ani.save("wave.gif", writer="pillow", fps=10)

print("GIF guardado como wave.gif")
