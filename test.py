import subprocess
from manim import *
import numpy as np

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080


class Test(Scene):
    def construct(self):
        # Title
        sources = Text("Sources").to_edge(UP, buff=0.5)
        ul = Underline(sources)

        # Items (manual bullets)
        item1 = Text(
            "• 3Blue1Brown Neural Networks Series Videos 1–4\n"
            "(https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)",
            font_size=28
        ).scale_to_fit_width(config.frame_width - 1)

        item2 = Text("• ChatGPT (For Manim help)", font_size=28)

        # Group and arrange items
        all_sources = VGroup(item1, item2).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        all_sources.next_to(sources, DOWN, buff=0.5)

        # Play animations
        self.play(Write(sources), Create(ul))
        self.play(Write(all_sources))
        self.wait()


def render_manim():
    command = ["manim", "-pql", "test.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()
    # os.system("open media/videos/test/480p15/Test.mp4")