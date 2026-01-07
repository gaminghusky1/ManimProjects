import math
import subprocess
from manim import *

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080

class ExplainGradientDescent(Scene):
    def construct(self):
        main_title = Text("What will be explained in this video?").to_edge(UP, buff=1)
        ul = Underline(main_title)
        topics = BulletedList("Gradient Descent", "How it is used in neural networks", font_size=60).next_to(main_title, DOWN, buff=0.5)
        self.play(Write(main_title))
        self.play(Create(ul))
        self.wait(1)
        self.play(Write(topics[0]))
        self.wait(2)
        self.play(Write(topics[1]))
        self.wait(2)
        self.play(FadeOut(topics, main_title, ul))

        why = Text("Why did I choose this topic?")
        self.play(Write(why))
        self.wait(17)
        self.play(FadeOut(why))

        # Title
        text = Text("What is Gradient Descent?", font_size=60)
        self.play(Write(text))
        self.wait(1)
        self.play(FadeOut(text))
        self.wait(1)

        # Description
        ans = Text("An optimization algorithm used \nto find the minimum of a function.")
        self.play(Write(ans), run_time=4)
        self.wait(4)
        self.play(FadeOut(ans))

        # Equation
        title = Text("General Equation").to_edge(UP)
        self.play(FadeIn(title))
        self.wait(1)

        eq = MathTex(
            r"x_{i+1}", "=", r"x_{i}", "-", r"\eta", r"\nabla f(x_{i})",
            font_size=100
        )

        self.play(Write(eq), run_time=2)
        self.wait(1)

        terms = [
            ("Next position", 0, DOWN, 8),
            ("Current position", 2, UP, 6),
            ("Gradient", 5, UP, 11),
            ("Learning rate", 4, DOWN, 28),
        ]

        for label_text, idx, direction, wait_time in terms:
            brace = Brace(eq[idx], direction)
            label = brace.get_text(label_text)
            self.play(Create(brace), Write(label))
            self.wait(wait_time)

        self.play(FadeOut(*self.mobjects))
        self.wait(2)


def render_manim():
    command = ["manim", "-pql", "Gradient Descent Explanation.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()