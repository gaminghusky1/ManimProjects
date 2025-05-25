import math
import subprocess
from manim import *

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080

class Backprop(Scene):
    def construct(self):
        # Speak - final step; how the neural network actually "learns"
        # Gradient descent - minimize cost
        goal = Text("Goal: Minimize Cost", font_size=72)
        self.play(FadeIn(goal))
        self.wait(2)
        self.play(FadeOut(goal))
        method = Text("Backpropagation", font_size=72)
        self.play(FadeIn(method))
        self.wait(2)
        self.play(FadeOut(method))

        # Speak - weights are only thing we can actually change
        # Goal: gradient = dc/dw; can't do directly.
        dc_dw = MathTex(r"\frac{\partial C}{\partial w_i}")
        text_dc_dw = Text("Gradient: ").next_to(dc_dw, LEFT, buff=0.3)
        dc_dw_group = VGroup(text_dc_dw, dc_dw).move_to(ORIGIN)
        self.play(Write(dc_dw_group))
        self.play(dc_dw_group.animate.to_corner(UL), run_time=1)

        # Recall forward prop functions
        z_label = MathTex(r"z_i ")
        z_func = MathTex(r"= a_{i - 1} \cdot w_i").next_to(z_label, RIGHT, buff=0.3)
        z_group = VGroup(z_label, z_func).move_to(UP)
        a_label = MathTex(r"a_{i} ")
        a_func = MathTex(r"= \sigma(z_i)").next_to(a_label, RIGHT, buff=0.3)
        a_group = VGroup(a_label, a_func).move_to(ORIGIN)
        c_label = MathTex(r"C ")
        c_func = MathTex(r"= (a_n - y)^2").next_to(c_label, RIGHT, buff=0.3)
        c_group = VGroup(c_label, c_func).move_to(DOWN)

        dz_dw_label = MathTex(r"\frac{\partial z_i}{\partial w_i} ")
        dz_dw = MathTex(r"= a_{i - 1}").next_to(dz_dw_label, RIGHT, buff=0.3)
        dz_dw_group = VGroup(dz_dw_label, dz_dw).move_to(UP * 2)
        da_dz_label = MathTex(r"\frac{\partial a_i}{\partial z_i} ")
        da_dz = MathTex(r"= \sigma'(z_i)").next_to(da_dz_label, RIGHT, buff=0.3)
        da_dz_group = VGroup(da_dz_label, da_dz).move_to(ORIGIN)
        dc_da_label = MathTex(r"\frac{\partial C}{\partial a_n} ")
        dc_da = MathTex(r"= 2(a_n - y)").next_to(dc_da_label, RIGHT, buff=0.3)
        dc_da_group = VGroup(dc_da_label, dc_da).move_to(DOWN * 2)

        # Animate writing the original formulas
        self.play(Write(z_group))
        self.play(Write(a_group))
        self.play(Write(c_group))
        self.wait(2)

        # Use TransformMatchingShapes with expressions that share structure
        self.play(TransformMatchingShapes(z_label, dz_dw_label), z_func.animate.become(dz_dw), run_time=2)
        self.wait(1)
        self.play(TransformMatchingShapes(a_label, da_dz_label), TransformMatchingShapes(a_func, da_dz), run_time=2)
        self.wait(1)
        self.play(TransformMatchingShapes(c_label, dc_da_label), TransformMatchingShapes(c_func, dc_da), run_time=2)
        self.wait(3)

        eq_sign = MathTex("=").next_to(dc_dw, RIGHT, buff=0.3)
        self.remove(z_func)
        self.play(FadeOut(dc_dw_group), dz_dw_group.animate.to_corner(UL))
        self.play(da_dz_group.animate.next_to(dz_dw_group, RIGHT, buff=0.3))
        self.play(dc_da_group.animate.next_to(da_dz_group, RIGHT, buff=0.3))

        dc_da_label_in_chain = MathTex(r"\frac{\partial C}{\partial a_i}").next_to(eq_sign, RIGHT, buff=0.3)
        da_dz_label_in_chain = MathTex(r"\frac{\partial a_i}{\partial z_i}").next_to(dc_da_label_in_chain, RIGHT, buff=0.1)
        dz_dw_label_in_chain = MathTex(r"\frac{\partial z_i}{\partial w_i}").next_to(da_dz_label_in_chain, RIGHT, buff=0.1)

        chain_expr = VGroup(
            dc_da_label_in_chain,
            da_dz_label_in_chain,
            dz_dw_label_in_chain
        )


        chain_rule_group = VGroup(
            dc_dw,
            eq_sign,
            chain_expr
        ).move_to(ORIGIN)

        dc_dz_label_in_chain = MathTex(r"\frac{\partial C}{\partial z_i}").next_to(eq_sign, RIGHT, buff=0.3)
        second_dz_dw_label_in_chain = MathTex(r"\frac{\partial z_i}{\partial w_i}").next_to(dc_dz_label_in_chain, RIGHT, buff=0.1)

        chain_expr_2 = VGroup(
            dc_dz_label_in_chain,
            second_dz_dw_label_in_chain
        )

        dc_dw_label_in_chain = MathTex(r"\frac{\partial C}{\partial w_i}").next_to(eq_sign, RIGHT, buff=0.3)

        dc_da_label_mover = dc_da_label.copy()
        da_dz_label_mover = da_dz_label.copy()
        dz_dw_label_mover = dz_dw_label.copy()


        self.play(Write(dc_dw), Write(eq_sign))
        self.wait(1)
        self.play(TransformMatchingShapes(dc_da_label_mover, dc_da_label_in_chain))
        self.wait(1)
        self.play(da_dz_label_mover.animate.move_to(da_dz_label_in_chain))
        self.wait(1)
        self.play(dz_dw_label_mover.animate.move_to(dz_dw_label_in_chain))
        self.wait(3)

        self.add(chain_expr)
        self.remove(dc_da_label_mover, da_dz_label_mover, dz_dw_label_mover)

        cross_line = Line(dc_da_label_in_chain.get_corner(DL), da_dz_label_in_chain.get_corner(UR), color=RED)

        self.play(Create(cross_line))
        self.wait(2)
        self.play(TransformMatchingShapes(chain_expr, chain_expr_2), FadeOut(cross_line))
        self.add(chain_expr_2)
        self.remove(chain_expr)
        self.play(Create(cross_line))
        self.wait(2)
        self.play(TransformMatchingShapes(chain_expr_2, dc_dw_label_in_chain), FadeOut(cross_line))
        self.add(dc_dw_label_in_chain)
        self.remove(chain_expr_2)
        self.wait(2)

        self.play(FadeOut(dc_dw_label_in_chain))

        new_dc_dw = VGroup(
            MathTex(r"a_{n - 1}"),
            MathTex(r"\cdot"),
            MathTex(r"\sigma'(z_n)"),
            MathTex(r"\cdot"),
            MathTex(r"2(a_n - y)")
        )

        for i in range(1, 5):
            new_dc_dw[i].next_to(new_dc_dw[i - 1], RIGHT, buff=0.1)

        new_dc_dw.next_to(eq_sign, RIGHT, buff=0.3)
        new_dc_dw_eq = VGroup(
            dc_dw,
            eq_sign,
            new_dc_dw
        )

        offset_to_origin = ORIGIN - new_dc_dw_eq.get_center()
        self.play(dc_dw.animate.become(MathTex(r"\frac{\partial C}{\partial w_n}").move_to(dc_dw.get_center() + offset_to_origin)), eq_sign.animate.shift(offset_to_origin))
        new_dc_dw.shift(offset_to_origin)

        dz_dw_mover = new_dc_dw[0].copy().move_to(dz_dw, aligned_edge=RIGHT)
        da_dz_mover = new_dc_dw[2].copy().move_to(da_dz, aligned_edge=RIGHT)
        dc_da_mover = new_dc_dw[4].copy().move_to(dc_da, aligned_edge=RIGHT)

        self.play(dz_dw_mover.animate.move_to(new_dc_dw[0]))
        self.play(Create(new_dc_dw[1]))
        self.add(new_dc_dw[0])
        self.remove(dz_dw_mover)
        self.play(da_dz_mover.animate.move_to(new_dc_dw[2]))
        self.play(Create(new_dc_dw[3]))
        self.add(new_dc_dw[2])
        self.remove(da_dz_mover)
        self.play(dc_da_mover.animate.move_to(new_dc_dw[4]))
        self.add(new_dc_dw[4])
        self.remove(dc_da_mover)
        self.wait(1)

        self.play(FadeOut(dz_dw_group, da_dz_group, dc_da_group))
        self.wait(3)

        self.play(FadeOut(new_dc_dw_eq))
        chain_expr.next_to(eq_sign, RIGHT, buff=0.3)
        chain_rule_group.move_to(ORIGIN)
        dc_dw.become(MathTex(r"\frac{\partial C}{\partial w_i}").move_to(dc_dw.get_center()))

        self.play(FadeIn(chain_rule_group))
        self.wait(1)
        self.play(
            dz_dw_label_in_chain.animate.become(MathTex(r"\frac{\partial z_i}{\partial a_{i - 1}").next_to(da_dz_label_in_chain, RIGHT, buff=0.1)),
            dc_dw.animate.become(MathTex(r"\frac{\partial C}{\partial a_{i - 1}}").move_to(dc_dw.get_center()))
        )
        self.wait(2)
        self.play(FadeOut(chain_rule_group))

        dz_da_label = MathTex(r"\frac{\partial z_i}{\partial a_{i - 1}} ")
        dz_da = MathTex(r"= w_i").next_to(dz_da_label, RIGHT, buff=0.3)

        temp = VGroup(dz_da_label, dz_da)
        offset_to_origin = ORIGIN - temp.get_center()
        dz_da_label.shift(offset_to_origin)
        dz_da.shift(offset_to_origin)

        z_func = MathTex(r"= a_{i - 1} \cdot w_i").next_to(z_label, RIGHT, buff=0.3)
        z_group = VGroup(z_label, z_func).move_to(ORIGIN)
        self.play(FadeIn(z_group))
        self.wait(1)
        self.play(TransformMatchingShapes(z_label, dz_da_label), z_func.animate.become(dz_da))
        self.add(dz_da_label, dz_da)
        self.remove(z_label, z_func)
        self.wait(2)
        self.play(FadeOut(dz_da_label, dz_da))

        eq_sign.next_to(new_dc_dw, LEFT, buff=0.3)
        dc_dw.become(MathTex(r"\frac{\partial C}{\partial w_n}").next_to(eq_sign, LEFT, buff=0.3))

        self.play(FadeIn(new_dc_dw_eq))
        self.wait(1)
        self.play(new_dc_dw_eq[0].animate.become(MathTex(r"\frac{\partial C}{\partial a_{n - 1}}").next_to(eq_sign, LEFT, buff=0.3)), new_dc_dw[0].animate.become(MathTex(r"w_i").move_to(new_dc_dw[0].get_center())))
        self.wait(2)
        self.play(FadeOut(new_dc_dw_eq))

        final_eqs = VGroup(
            MathTex(r"\frac{\partial C}{\partial w_n} = a_{n - 1}\cdot \sigma'(z_n)\cdot 2(a_n - y)"),
            MathTex(r"\frac{\partial C}{\partial a_{n - 1}} = w_{n}\cdot \sigma'(z_n)\cdot 2(a_n - y)"),
            MathTex(r"\frac{\partial C}{\partial w_i} = a_{i - 1}\cdot \sigma'(z_i)\cdot \frac{\partial C}{\partial a_{i}}"),
            MathTex(r"\frac{\partial C}{\partial a_{i - 1}} = w_{i}\cdot \sigma'(z_i)\cdot \frac{\partial C}{\partial a_{i}}")
        )
        for i in range(1, len(final_eqs)):
            final_eqs[i].next_to(final_eqs[i - 1], DOWN, buff=0.5)
        final_eqs.move_to(ORIGIN)

        self.play(Write(final_eqs), run_time=2)
        self.wait(2)
        self.play(Circumscribe(final_eqs[0], stroke_width=2), run_time=1)
        self.wait(3)
        self.play(Circumscribe(final_eqs[1], stroke_width=2), run_time=1)
        self.wait(3)
        self.play(Circumscribe(final_eqs[2], stroke_width=2), run_time=1)
        self.wait(3)
        self.play(Circumscribe(final_eqs[3], stroke_width=2), run_time=1)
        self.wait(3)

        self.play(FadeOut(final_eqs))
        self.wait(2)


def render_manim():
    command = ["manim", "-pql", "Backpropagation.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()