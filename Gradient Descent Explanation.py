import math
import subprocess
from manim import *

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080

class ExplainGradientDescent(Scene):
    def construct(self):
        # Title
        text = Text("What is Gradient Descent?", font_size=60)
        self.play(Write(text))
        self.wait(1)
        self.play(FadeOut(text))

        # Description
        ans = Text("An optimization algorithm used \nto find the minimum of a function.")
        self.play(Write(ans))
        self.wait(2)
        self.play(FadeOut(ans))

        # Equation
        title = Text("General Equation").to_edge(UP)
        self.play(FadeIn(title))

        eq = MathTex(
            r"x_{i+1}", "=", r"x_{i}", "-", r"\eta", r"\nabla f(x_{i})",
            font_size=100
        )

        self.play(Write(eq), run_time=2)
        self.wait(1)

        terms = [
            ("Next position", 0, DOWN, 2),
            ("Current position", 2, UP, 2),
            ("Gradient", 5, UP, 4),
            ("Learning rate", 4, DOWN, 7),
        ]

        for label_text, idx, direction, wait_time in terms:
            brace = Brace(eq[idx], direction)
            label = brace.get_text(label_text)
            self.play(Create(brace), Write(label))
            self.wait(wait_time)

        self.play(FadeOut(*self.mobjects))


def render_manim():
    command = ["manim", "-pql", "Gradient Descent Explanation.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()