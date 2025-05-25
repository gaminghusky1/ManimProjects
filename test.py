import subprocess
from manim import *
import numpy as np

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080


class Test(ThreeDScene):
    def construct(self):
        # ---------- Part 1: Display the n-body Equation in Large Text ----------
        eq = MathTex(
            r"m_i \frac{d^2 \mathbf{q}_i}{dt^2}",
            r"=",
            r"\sum_{\substack{j=1 \\ j \neq i}}^{n}",
            r"\frac{G m_i m_j (\mathbf{q}_j - \mathbf{q}_i)}{\|\mathbf{q}_j - \mathbf{q}_i\|^3}",
            font_size=72
        )
        eq.scale(0.8)
        self.play(Write(eq))
        self.wait(3)
        self.play(FadeOut(eq))

        # ---------- Set up the 3D Camera ----------
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # ---------- Part 2: 3-Body Simulation with Time-Lapse Motion ----------
        # Simulation parameters.
        G = 1  # Gravitational constant
        mass = 1  # Mass of each body
        d = 4  # Initial side length of the equilateral triangle
        epsilon = 0.2  # Softening parameter (avoids singularity)
        time_lapse = 10  # Factor to speed up the simulation (time-lapse)

        # Compute initial absolute positions (vertices of an equilateral triangle in the xy-plane).
        theta_values = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        positions = [
            np.array([d / np.sqrt(3) * np.cos(theta), d / np.sqrt(3) * np.sin(theta), 0])
            for theta in theta_values
        ]

        # For a stable Lagrange solution the orbital speed is v = sqrt(G*m/R)
        # Here R = d/√3. We choose a fraction (e.g., 0.5) of that speed so gravity dominates and pulls them together.
        R = d / np.sqrt(3)
        v_orb = np.sqrt(G * mass / R)
        v_mag = 0.5 * v_orb * time_lapse  # scaled by time_lapse for a fast simulation

        # Compute initial velocities, perpendicular to the radius (in the xy-plane).
        velocities = []
        for pos in positions:
            x, y, _ = pos
            # Perpendicular direction (rotate by 90°)
            perp = np.array([-y, x, 0])
            perp = perp / np.linalg.norm(perp)
            velocities.append(v_mag * perp)

        # Create spherical bodies with three distinct colors.
        colors = [RED, GREEN, BLUE]
        bodies = VGroup(*[
            Sphere(radius=0.2, resolution=(16, 32), color=color)
            for color in colors
        ])

        # Create trails for each body.
        trails = VGroup(*[
            TracedPath(lambda body=body: body.get_center(), stroke_color=body.get_color(), stroke_width=2)
            for body in bodies
        ])

        # Add bodies and trails to the scene.
        self.add(bodies, trails)

        # Simulation time step for Euler integration.
        sim_dt = 0.01 * time_lapse

        def update_bodies(mob, dt):
            nonlocal positions, velocities
            new_positions = []
            new_velocities = []
            # Euler integration: compute gravitational forces and update positions/velocities.
            for i, pos in enumerate(positions):
                force = np.zeros(3)
                for j, pos_j in enumerate(positions):
                    if i == j:
                        continue
                    r_vec = pos_j - pos
                    r = np.linalg.norm(r_vec)
                    # Newton's law of gravitation with softening.
                    force += G * mass * mass * r_vec / ((r ** 2 + epsilon ** 2) ** (1.5))
                # Compute acceleration and update using Euler's method.
                acc = force / mass
                new_vel = velocities[i] + acc * sim_dt
                new_pos = pos + new_vel * sim_dt
                new_velocities.append(new_vel)
                new_positions.append(new_pos)
            positions = new_positions
            velocities = new_velocities

            # Compute the center of mass.
            center_of_mass = np.mean(positions, axis=0)
            # Update each body's displayed position relative to the center of mass.
            for body, pos in zip(bodies, positions):
                body.move_to(pos - center_of_mass)
            # (Trails will follow these recentered positions.)

        # Attach the updater.
        bodies.add_updater(update_bodies)

        # Run the simulation (adjust duration as desired).
        self.wait(20)


def render_manim():
    command = ["manim", "-pql", "test.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()
    # os.system("open media/videos/test/480p15/Test.mp4")