import numpy as np
import matplotlib.pyplot as plt
G = 1
class Body:
    def __init__(self, name, mass, position, velocity):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(3)
def compute_accelerations(bodies):
    for i, bi in enumerate(bodies):
        acc = np.zeros(3)
        for j, bj in enumerate(bodies):
            if i != j:
                r = bj.position - bi.position
                acc += G * bj.mass * r / np.linalg.norm(r)**3
        bi.acceleration = acc

# === Velocity Verlet Integration ===
def velocity_verlet(bodies, dt, steps):
    positions = {b.name: np.zeros((steps, 3)) for b in bodies}
    compute_accelerations(bodies)

    for t in range(steps):
        for b in bodies:
            b.position += b.velocity * dt + 0.5 * b.acceleration * dt**2
        for b in bodies:
            positions[b.name][t] = b.position.copy()
        old_accs = [b.acceleration.copy() for b in bodies]
        compute_accelerations(bodies)
        for b, a_old in zip(bodies, old_accs):
            b.velocity += 0.5 * (a_old + b.acceleration) * dt

    return positions

# === Editable Initial Conditions ===
initial_conditions = [
    {
        "name": "Sun",
        "mass": 1.0,
        "position": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
    },
    {
        "name": "Jupiter",
        "mass": 0.000954536311,
        "position": [22, 7, 9.0],
        "velocity": [-0, -0.1, -0.1],  # AU/yr
    },
    {
        "name": "Saturn",
        "mass": 0.000294709314,
        "position": [19, 6.5, 10.5],
        "velocity": [-0.1, -0.1, -0.1],  # AU/yr
    }
]

bodies = [Body(**kwargs) for kwargs in initial_conditions]

dt = 0.001       
steps = 1000000    
positions = velocity_verlet(bodies, dt, steps)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for name, path in positions.items():
    if name != "Sun":
        ax.plot(path[:,0], path[:,1], path[:,2], label=name)
ax.scatter(0, 0, 0, color='orange', s=150, label='Sun')
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_zlabel("z [AU]")
ax.set_title("3D Orbits of Jupiter and Saturn")
ax.legend()
plt.show()


