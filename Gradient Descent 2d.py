import math
import subprocess
from manim import *

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080

class GradientDescent2d(Scene):
    def construct(self):
        # 1) Axes + plot
        axes = Axes(
            x_range=[-3.5, 2.67, 1],
            y_range=[0, 11, 1],
            x_length=4,
            y_length=6,
            axis_config={"include_tip": False}
        ).to_edge(DOWN)
        axes.add_coordinates(font_size=20)
        self.play(Create(axes), run_time=1)

        func = lambda x: 0.125 * x ** 4 + 0.25 * x ** 3 - 0.5 * x + 1
        d_func = lambda x: 0.5 * x ** 3 + 0.75 * x ** 2 - 0.5
        graph = axes.plot(func, color=BLUE)
        graph_label = MathTex(r"C(x) = \frac{1}{8}x^4 + \frac{1}{4}x^3 -\frac{1}{2}x + 2").next_to(axes, UP)
        self.wait(1)
        self.play(
            FadeIn(graph_label),
            Create(graph),
            run_time=1
        )

        graph_group = VGroup(axes, graph)
        self.wait(2)

        # ValueTracker
        x_val = ValueTracker(-3.0)
        learning_rate = 0.03
        tolerance = 1e-3

        # 2) Labels and live values
        # x =
        x_label = MathTex(r"x =")
        x_value = DecimalNumber(x_val.get_value(), num_decimal_places=2)
        x_value.add_updater(lambda m: m.set_value(x_val.get_value()))
        x_row = VGroup(x_label, x_value).arrange(RIGHT, buff=0.2)

        # ∇C =
        deriv_label = MathTex(r"\nabla C = C'(x) =")
        deriv_value = DecimalNumber(d_func(x_val.get_value()), num_decimal_places=2)
        deriv_value.add_updater(lambda m: m.set_value(d_func(x_val.get_value())))
        deriv_row = VGroup(deriv_label, deriv_value).arrange(RIGHT, buff=0.2)

        # η =
        eta_label = MathTex(fr"\eta = {learning_rate}")

        # Δx =
        dx_label = MathTex(r"\Delta x = - \eta \nabla C =")
        dx_value = DecimalNumber(-learning_rate * d_func(x_val.get_value()), num_decimal_places=2)
        dx_value.add_updater(lambda m: m.set_value(-learning_rate * d_func(x_val.get_value())))
        dx_row = VGroup(dx_label, dx_value).arrange(RIGHT, buff=0.2)

        # Combine and position top-left
        info_group = VGroup(x_row, deriv_row, eta_label, dx_row).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        info_group.to_corner(UL)

        # 3) Moving dot
        dot = Dot(color=RED).add_updater(
            lambda d: d.move_to(
                axes.coords_to_point(x_val.get_value(), func(x_val.get_value()))
            )
        )

        # Animate text and move graph
        self.play(
            FadeOut(graph_label),
            graph_group.animate.scale(1.2).move_to(RIGHT),
            Write(x_row),
            Create(dot),
            run_time=1
        )

        # 4) Tangent line
        tangent_graph = always_redraw(lambda: axes.plot(
            lambda x: func(x_val.get_value()) + d_func(x_val.get_value()) * (x - x_val.get_value()),
            color=ORANGE,
            x_range=[
                x_val.get_value() - math.sqrt(1 / (1 + d_func(x_val.get_value()) ** 2)),
                x_val.get_value() + math.sqrt(1 / (1 + d_func(x_val.get_value()) ** 2))
            ]
        ))
        self.wait(6)

        # Show ∇C and value
        self.play(
            Create(tangent_graph),
            FadeIn(deriv_row),
            run_time=1
        )

        self.wait(8)

        # Show Δx and η
        self.play(
            FadeIn(dx_row),
            FadeIn(eta_label),
            run_time=1
        )

        self.wait(3)

        trail = TracedPath(dot.get_center, stroke_color=YELLOW)
        self.add(trail)

        # 7) Gradient-descent updater
        self.converged = False

        def gradient_step(mob):
            current_x = x_val.get_value()
            grad = d_func(current_x)  # replace with f'(current_x)
            new_x = current_x - learning_rate * grad

            if abs(grad) < tolerance:
                self.converged = True
                mob.remove_updater(gradient_step)
            else:
                x_val.set_value(new_x)

        def update_dot(mobject):
            grad = d_func(x_val.get_value())
            new_x = x_val.get_value() - learning_rate * grad
            x_val.set_value(new_x)

        # self.play(UpdateFromFunc(dot, update_dot), run_time=5)
        dot.add_updater(gradient_step)

        # 8) Wait until converged
        self.wait_until(lambda: self.converged)

        scene_group = VGroup(
            axes,
            graph_group,
            info_group,
            tangent_graph,
            trail,
            dot
        )

        self.play(FadeOut(scene_group))

        transition = Text("What if we had a higher learning rate?")
        self.play(FadeIn(transition))
        self.wait(1)
        self.play(FadeOut(transition))
        self.play(FadeIn(scene_group))

        # Demonstrate higher learning rate
        learning_rate = 0.7

        dx_value.clear_updaters()
        dx_value.add_updater(lambda m: m.set_value(-learning_rate * d_func(x_val.get_value())))

        dot.remove_updater(gradient_step)

        def gradient_step(mob):
            current_x = x_val.get_value()
            grad = d_func(current_x)
            new_x = current_x - learning_rate * grad

            if abs(grad) < tolerance:
                self.converged = True
                mob.remove_updater(gradient_step)
            else:
                x_val.set_value(new_x)

        self.play(x_val.animate.set_value(-3.0), Uncreate(trail), run_time=1)

        self.wait(2)

        self.play(eta_label.animate.become(MathTex(fr"\eta = {learning_rate}").move_to(eta_label)), run_time=1)

        self.wait(2)

        trail = TracedPath(dot.get_center, stroke_color=YELLOW)
        self.add(trail)

        self.converged = False
        dot.add_updater(gradient_step)

        self.wait_until(lambda: self.converged)
        self.wait(15)
        self.play(FadeOut(*self.mobjects))


def render_manim():
    command = ["manim", "-pql", "Gradient Descent 2d.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()