import math
import subprocess
from manim import *

# High-quality settings for a crisp, full-screen render.
config.pixel_width = 1920
config.pixel_height = 1080

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetworkBasics(Scene):
    def construct(self):
        input_neuron = Circle(radius=0.5, color=BLUE).move_to(LEFT * 4.5)
        hidden_neuron1 = Circle(radius=0.5, color=GREEN).move_to(LEFT * 1.5)
        hidden_neuron2 = Circle(radius=0.5, color=GREEN).move_to(RIGHT * 1.5)
        output_neuron = Circle(radius=0.5, color=RED).move_to(RIGHT * 4.5)

        self.play(Create(input_neuron), run_time=0.5)
        self.play(Create(hidden_neuron1), run_time=0.5)
        self.play(Create(hidden_neuron2), run_time=0.5)
        self.play(Create(output_neuron), run_time=0.5)

        hidden_neurons = VGroup(hidden_neuron1, hidden_neuron2)

        # Create connections (lines)
        lines = VGroup(
            Line(input_neuron, hidden_neuron1, color=GRAY),
            Line(hidden_neuron1, hidden_neuron2, color=GRAY),
            Line(hidden_neuron2, output_neuron, color=GRAY)
        )

        # Animate connections
        self.play(*[Create(line) for line in lines], run_time=1)

        neural_network = VGroup(
            input_neuron, hidden_neuron1, hidden_neuron2, output_neuron,
            lines
        )

        self.wait(2)

        terms = [
            ("Input Layer", input_neuron, DOWN, 2),
            ("Hidden Layers", hidden_neurons, DOWN, 2),
            ("Output Layer", output_neuron, DOWN, 2),
        ]

        labels = VGroup()

        for label_text, loc, direction, wait_time in terms:
            brace = Brace(loc, direction)
            label = brace.get_text(label_text)
            labels.add(label, brace)
            self.play(Create(brace), Write(label))
            self.wait(wait_time)

        self.play(FadeOut(labels))

        self.play(ScaleInPlace(neural_network, 1.4))


        weight_vals = [0.87, -0.38, 1.56]

        weight_labels = VGroup(
            MathTex("w_1 =").next_to(Line(input_neuron.get_right(), hidden_neuron1.get_left()), UP, buff=0.2),
            MathTex("w_2 =").next_to(Line(hidden_neuron1.get_right(), hidden_neuron2.get_left()), UP, buff=0.2),
            MathTex("w_3 =").next_to(Line(hidden_neuron2.get_right(), output_neuron.get_left()), UP, buff=0.2),
        )

        weight_groups = VGroup(
            VGroup(weight_labels[0], DecimalNumber(weight_vals[0], num_decimal_places=2).next_to(weight_labels[0], RIGHT, buff=0.2)).next_to(Line(input_neuron.get_right(), hidden_neuron1.get_left()), UP, buff=0.2),
            VGroup(weight_labels[1], DecimalNumber(weight_vals[1], num_decimal_places=2).next_to(weight_labels[1], RIGHT, buff=0.2)).next_to(Line(hidden_neuron1.get_right(), hidden_neuron2.get_left()), UP, buff=0.2),
            VGroup(weight_labels[2], DecimalNumber(weight_vals[2], num_decimal_places=2).next_to(weight_labels[2], RIGHT, buff=0.2)).next_to(Line(hidden_neuron2.get_right(), output_neuron.get_left()), UP, buff=0.2),
        )

        self.play(Write(weight_groups))

        self.wait(2)

        activations = VGroup()
        activation_vals = []
        z_labels = VGroup()
        z_vals = []

        def animate_forward_prop(input_val):
            activation_vals.clear()
            z_vals.clear()
            if len(activations) and len(z_labels):
                self.play(FadeOut(*activations.submobjects), FadeOut(*z_labels.submobjects))
            activations.remove(*activations.submobjects)
            z_labels.remove(*z_labels.submobjects)
            a_prev = DecimalNumber(input_val, num_decimal_places=2).move_to(input_neuron.get_center())
            activations.add(VGroup(MathTex(r"a_{0}").next_to(input_neuron, UP, buff=0.2), a_prev))
            activation_vals.append(input_val)
            self.play(Write(a_prev))
            self.wait(0.5)

            # Loop over each layer connection
            for idx, (from_neuron, to_neuron) in enumerate([
                (input_neuron,   hidden_neuron1),
                (hidden_neuron1, hidden_neuron2),
                (hidden_neuron2, output_neuron),
            ]):
                w_mob = weight_groups[idx][1]
                w_val = weight_vals[idx]


                z_val = float(a_prev.get_value()) * w_val
                z_label = MathTex(
                    fr"z_{idx + 1}="
                )
                a_copy = a_prev.copy().next_to(z_label, RIGHT, buff=0.2)
                a_copy_original = a_prev.copy()
                dot_sym = MathTex(r"\cdot").next_to(a_copy, RIGHT, buff=0.1)
                w_copy = w_mob.copy().next_to(dot_sym, RIGHT, buff=0.1)
                w_copy_original = w_mob.copy()

                eq_sign = MathTex(r"=").next_to(w_copy, RIGHT, buff=0.2)
                z_res = MathTex(
                    fr"{z_val:.2f}"
                ).next_to(eq_sign, RIGHT, buff=0.2)

                expr_group = VGroup(a_copy, dot_sym, w_copy)

                eq_group = VGroup(z_label, expr_group, eq_sign, z_res).move_to(UP * 2.5)

                originals = VGroup(a_copy_original, w_copy_original)
                self.play(
                    Transform(originals, expr_group),
                    FadeIn(z_label)
                )
                self.add(expr_group); self.remove(originals)
                self.wait(0.3)
                self.play(FadeIn(eq_sign))
                expr_group_animator = expr_group.copy()
                self.play(Transform(expr_group_animator, z_res), run_time=1)
                self.add(z_res); self.remove(expr_group_animator)
                self.wait(0.5)

                # Apply sigmoid
                a_val = sigmoid(z_val)
                sigma_expr_left = MathTex(f"a_{idx + 1} =")
                sigma_func_left = MathTex(r"\sigma(").next_to(sigma_expr_left, RIGHT, buff=0.2)
                sigma_z_ins = MathTex(f"{z_val:.2f}").next_to(sigma_func_left, RIGHT, buff=0)
                sigma_func_right = MathTex(")").next_to(sigma_z_ins, RIGHT, buff=0)
                sigma_expr_eq_sign = MathTex(r"=").next_to(sigma_func_right, RIGHT, buff=0.2)
                sigma_res = MathTex(f"{a_val:.2f}").next_to(sigma_expr_eq_sign, RIGHT, buff=0.2)
                sigma_func_group = VGroup(sigma_func_left, sigma_z_ins, sigma_func_right)
                sigma_group = VGroup(
                    sigma_expr_left,
                    sigma_func_group,
                    sigma_expr_eq_sign,
                    sigma_res
                ).move_to(UP * 1.5)
                z_res_duplicator = z_res.copy()
                self.play(TransformMatchingShapes(z_res_duplicator, sigma_z_ins), FadeIn(sigma_expr_left, sigma_func_left, sigma_func_right, sigma_expr_eq_sign))
                self.add(sigma_z_ins); self.remove(z_res_duplicator)
                self.wait(0.3)
                sigma_func_group_animator = sigma_func_group.copy()
                self.play(Transform(sigma_func_group_animator, sigma_res))
                self.add(sigma_res); self.remove(sigma_func_group_animator)
                self.wait(0.5)

                # 4) Move this activation into the next neuron
                a_prev = DecimalNumber(a_val, num_decimal_places=2).move_to(to_neuron.get_center())
                a_label = MathTex(fr"a_{idx + 1}").next_to(to_neuron, UP, buff=0.2)
                a_group = VGroup(a_label, a_prev)
                z_prev = DecimalNumber(z_val, num_decimal_places=2)
                z_label = MathTex(fr"z_{idx + 1}=").next_to(z_prev, LEFT, buff=0.3)
                z_group = VGroup(z_label, z_prev).next_to(Line(from_neuron, to_neuron), DOWN, buff=0.2)
                z_res_copy = z_res.copy()
                sigma_res_copy = sigma_res.copy()
                self.play(
                    Transform(sigma_res_copy, a_prev),
                    FadeIn(a_label),
                    Transform(z_res_copy, z_prev),
                    FadeIn(z_label),
                    FadeOut(eq_group, sigma_group)
                )
                self.add(a_prev, z_prev)
                self.remove(sigma_res_copy, z_res_copy)
                activations.add(a_group)
                activation_vals.append(a_val)
                z_labels.add(z_group)
                z_vals.append(z_val)
                self.wait(0.5)

            self.wait(1)

        # Run forward propagation
        animate_forward_prop(2.00)
        self.wait(2)

        def animate_cost_calculation(expected_output):
            expected = VGroup(
                MathTex(r"y = "),
                MathTex(f"{expected_output:.2f}")
            )
            expected[1].next_to(expected[0], RIGHT, buff=0.3)
            expected.move_to(UP * 1.5)
            self.play(Write(expected))
            self.wait(1)
            expected_val_copy_to_move = expected[1].copy()

            cost_label = MathTex(r"C =")
            paren = MathTex("(").next_to(cost_label, RIGHT, buff=0.2)
            output_copy = activations[-1][1].copy().next_to(paren, RIGHT, buff=0)
            output_copy_to_move = activations[-1][1].copy()
            mse_remaining = MathTex(fr"- {expected_output:.2f})^2").next_to(output_copy, RIGHT, buff=0.2)
            expr = VGroup(paren, output_copy, mse_remaining)
            cost_val = (activation_vals[-1] - expected_output)**2
            cost_res = MathTex(fr"= {cost_val:.2f}").next_to(mse_remaining, RIGHT, buff=0.2)

            cost_group = VGroup(cost_label, expr, cost_res).move_to(DOWN * 1.5)

            self.play(
                FadeIn(cost_label, paren),
                TransformMatchingShapes(expected_val_copy_to_move, mse_remaining),
                TransformMatchingShapes(output_copy_to_move, output_copy),
                run_time=1
            )
            self.add(mse_remaining, output_copy)
            self.remove(expected_val_copy_to_move, output_copy_to_move)


            self.wait(2)
            expr_copy = expr.copy()
            self.play(
                Transform(expr_copy, cost_res),
                run_time=1
            )
            self.add(cost_res)
            self.remove(expr_copy)

            self.wait(2)
            self.play(FadeOut(cost_group, expected))

        animate_cost_calculation(1.00)

        learning_rate_group = VGroup(
            MathTex(r"\eta ="),
            DecimalNumber(5, num_decimal_places=2)
        )

        learning_rate_group[1].next_to(learning_rate_group[0], RIGHT, buff=0.3)
        learning_rate_group.scale(0.8).to_corner(UL)
        self.play(Write(learning_rate_group))
        self.wait(2)

        dc_dw_i_spacing = [0.2, 0.2, 0.1, 0.1, 0, 0, 0.1, 0.1, 0.2, 0.2]
        dc_dw_n_spacing = [0.2, 0.2, 0.1, 0.1, 0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0, 0.2, 0.2]
        w_change_spacing = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]

        def animate_backward_prop(expected_output):
            lr = learning_rate_group[1].get_value()
            a_n_mover = activations[-1][1].copy()
            a_i_min_1_val = activation_vals[-2]
            a_i_min_1_mover = activations[-2][1].copy()
            z_i_val = z_vals[-1]
            z_i_mover = z_labels[-1][1].copy()

            w_gradient_val = DecimalNumber(a_i_min_1_val * sigmoid_derivative(z_i_val) * 2*(activation_vals[-1] - expected_output), num_decimal_places=2)

            w_gradient_eq = VGroup(
                MathTex(r"\frac{\partial C}{\partial w_3}"),
                MathTex(r"="),
                a_i_min_1_mover.copy(),
                MathTex(r"\cdot"),
                MathTex(r"\sigma'("),
                z_i_mover.copy(),
                MathTex(r")"),
                MathTex(r"\cdot"),
                MathTex(r"2("),
                a_n_mover.copy(),
                MathTex(r"-"),
                DecimalNumber(expected_output, num_decimal_places=2),
                MathTex(r")"),
                MathTex(r"="),
                w_gradient_val
            )

            for i in range(len(dc_dw_n_spacing)):
                w_gradient_eq[i + 1].next_to(w_gradient_eq[i], RIGHT, buff=dc_dw_n_spacing[i])

            w_gradient_eq.move_to(UP * 2.5)
            w_gradient_solve_animator = w_gradient_eq[2:13].copy()

            self.play(Write(w_gradient_eq[0]), Write(w_gradient_eq[1]))
            self.wait(2)
            self.play(a_i_min_1_mover.animate.move_to(w_gradient_eq[2]))
            self.add(w_gradient_eq[2]); self.remove(a_i_min_1_mover)
            self.play(Write(w_gradient_eq[3]))
            self.wait(1)
            self.play(FadeIn(w_gradient_eq[4], w_gradient_eq[6]), z_i_mover.animate.move_to(w_gradient_eq[5]))
            self.add(w_gradient_eq[5]); self.remove(z_i_mover)
            self.play(Write(w_gradient_eq[7]))
            self.wait(1)
            self.play(FadeIn(w_gradient_eq[8], w_gradient_eq[12]), a_n_mover.animate.move_to(w_gradient_eq[9]))
            self.add(w_gradient_eq[9]); self.remove(a_n_mover)
            self.play(Write(w_gradient_eq[10]))
            self.wait(1)
            self.play(Write(w_gradient_eq[11]))
            self.play(Write(w_gradient_eq[13]))
            self.wait(1)
            self.play(w_gradient_solve_animator.animate.become(w_gradient_eq[14]))
            self.add(w_gradient_eq[14]); self.remove(w_gradient_solve_animator)
            self.wait(3)

            w_mover = weight_groups[-1][1].copy()
            w_gradient_val_mover = w_gradient_val.copy()
            lr_mover = learning_rate_group[1].copy()
            new_w_val = DecimalNumber(w_mover.get_value() - lr * w_gradient_val.get_value(), num_decimal_places=2)
            weight_vals[-1] = new_w_val.get_value()

            w_change_eq = VGroup(
                MathTex(r"w_3"),
                MathTex(r"="),
                w_mover.copy(),
                MathTex(r"-"),
                lr_mover.copy(),
                MathTex(r"\cdot"),
                w_gradient_val_mover.copy(),
                MathTex(r"="),
                new_w_val
            )

            for i in range(len(w_change_spacing)):
                w_change_eq[i + 1].next_to(w_change_eq[i], RIGHT, buff=w_change_spacing[i])

            w_change_eq.move_to(UP * 1.5)

            w_change_solve_animator = w_change_eq[2:7].copy()

            self.play(Write(w_change_eq[0]), Write(w_change_eq[1]))
            self.wait(2)
            self.play(w_mover.animate.move_to(w_change_eq[2]))
            self.add(w_change_eq[2]); self.remove(w_mover)
            self.play(Write(w_change_eq[3]))
            self.wait(1)
            self.play(lr_mover.animate.move_to(w_change_eq[4]))
            self.add(w_change_eq[4]); self.remove(lr_mover)
            self.play(Write(w_change_eq[5]))
            self.wait(1)
            self.play(w_gradient_val_mover.animate.move_to(w_change_eq[6]))
            self.add(w_change_eq[6]); self.remove(w_gradient_val_mover)
            self.play(Write(w_change_eq[7]))
            self.wait(1)
            self.play(w_change_solve_animator.animate.become(w_change_eq[8]))
            self.add(w_change_eq[8]); self.remove(w_change_solve_animator)
            self.wait(2)

            w_mover.move_to(weight_groups[-1][1])
            z_i_mover.move_to(z_labels[-1][1])
            a_n_mover.move_to(activations[-1][1])

            dc_da_val = DecimalNumber(w_mover.get_value() * sigmoid_derivative(z_i_val) * 2*(activation_vals[-1] - expected_output), num_decimal_places=2)

            dc_da_eq = VGroup(
                MathTex(r"\frac{\partial C}{\partial a_2}"),
                MathTex(r"="),
                w_mover.copy(),
                MathTex(r"\cdot"),
                MathTex(r"\sigma'("),
                z_i_mover.copy(),
                MathTex(r")"),
                MathTex(r"\cdot"),
                MathTex(r"2("),
                a_n_mover.copy(),
                MathTex(r"-"),
                DecimalNumber(expected_output, num_decimal_places=2),
                MathTex(r")"),
                MathTex(r"="),
                dc_da_val
            )

            for i in range(len(dc_dw_n_spacing)):
                dc_da_eq[i + 1].next_to(dc_da_eq[i], RIGHT, buff=dc_dw_n_spacing[i])

            dc_da_eq.move_to(DOWN * 2.5)
            dc_da_solve_animator = dc_da_eq[2:13].copy()

            self.play(Write(dc_da_eq[0]), Write(dc_da_eq[1]))
            self.wait(2)
            self.play(w_mover.animate.move_to(dc_da_eq[2]))
            self.add(dc_da_eq[2]); self.remove(w_mover)
            self.play(Write(dc_da_eq[3]))
            self.wait(1)
            self.play(FadeIn(dc_da_eq[4], dc_da_eq[6]), z_i_mover.animate.move_to(dc_da_eq[5]))
            self.add(dc_da_eq[5]); self.remove(z_i_mover)
            self.play(Write(dc_da_eq[7]))
            self.wait(1)
            self.play(FadeIn(dc_da_eq[8], dc_da_eq[12]), a_n_mover.animate.move_to(dc_da_eq[9]))
            self.add(dc_da_eq[9]); self.remove(a_n_mover)
            self.play(Write(dc_da_eq[10]))
            self.wait(1)
            self.play(Write(dc_da_eq[11]))
            self.play(Write(dc_da_eq[13]))
            self.wait(1)
            self.play(dc_da_solve_animator.animate.become(dc_da_eq[14]))
            self.add(dc_da_eq[14]); self.remove(dc_da_solve_animator)
            self.wait(3)

            self.play(weight_groups[-1][1].animate.become(DecimalNumber(new_w_val.get_value(), num_decimal_places=2).move_to(weight_groups[-1][1])))
            self.wait(2)
            self.play(FadeOut(w_gradient_eq, w_change_eq))

            for i in reversed(range(2)):
                a_i_min_1_val = activation_vals[i]
                a_i_min_1_mover = activations[i][1].copy()
                z_i_val = z_vals[i]
                z_i_mover = z_labels[i][1].copy()
                dc_da_val_mover = dc_da_val.copy()

                w_gradient_val = DecimalNumber(a_i_min_1_val * sigmoid_derivative(z_i_val) * dc_da_val.get_value(), num_decimal_places=2)

                w_gradient_eq = VGroup(
                    MathTex(r"\frac{\partial C}{\partial",f"w_{i + 1}","}"),
                    MathTex(r"="),
                    a_i_min_1_mover.copy(),
                    MathTex(r"\cdot"),
                    MathTex(r"\sigma'("),
                    z_i_mover.copy(),
                    MathTex(r")"),
                    MathTex(r"\cdot"),
                    dc_da_val_mover.copy(),
                    MathTex(r"="),
                    w_gradient_val
                )

                for j in range(len(dc_dw_i_spacing)):
                    w_gradient_eq[j + 1].next_to(w_gradient_eq[j], RIGHT, buff=dc_dw_i_spacing[j])

                w_gradient_eq.move_to(UP * 2.5)
                w_gradient_solve_animator = w_gradient_eq[2:9].copy()

                self.play(Write(w_gradient_eq[0]), Write(w_gradient_eq[1]))
                self.wait(2)
                self.play(a_i_min_1_mover.animate.move_to(w_gradient_eq[2]))
                self.add(w_gradient_eq[2]); self.remove(a_i_min_1_mover)
                self.play(Write(w_gradient_eq[3]))
                self.wait(1)
                self.play(FadeIn(w_gradient_eq[4], w_gradient_eq[6]), z_i_mover.animate.move_to(w_gradient_eq[5]))
                self.add(w_gradient_eq[5]); self.remove(z_i_mover)
                self.play(Write(w_gradient_eq[7]))
                self.wait(1)
                self.play(dc_da_val_mover.animate.move_to(w_gradient_eq[8]))
                self.add(w_gradient_eq[8]); self.remove(dc_da_val_mover)
                self.play(Write(w_gradient_eq[9]), FadeOut(dc_da_eq))
                self.wait(1)
                self.play(w_gradient_solve_animator.animate.become(w_gradient_eq[10]))
                self.add(w_gradient_eq[10]); self.remove(w_gradient_solve_animator)
                self.wait(3)

                w_mover = weight_groups[i][1].copy()
                w_gradient_val_mover = w_gradient_val.copy()
                lr_mover = learning_rate_group[1].copy()
                new_w_val = DecimalNumber(w_mover.get_value() - lr * w_gradient_val.get_value(), num_decimal_places=2)
                weight_vals[i] = new_w_val.get_value()

                w_change_eq = VGroup(
                    MathTex(f"w_{i + 1}"),
                    MathTex(r"="),
                    w_mover.copy(),
                    MathTex(r"-"),
                    lr_mover.copy(),
                    MathTex(r"\cdot"),
                    w_gradient_val_mover.copy(),
                    MathTex(r"="),
                    new_w_val
                )

                for j in range(len(w_change_spacing)):
                    w_change_eq[j + 1].next_to(w_change_eq[j], RIGHT, buff=w_change_spacing[j])

                w_change_eq.move_to(UP * 1.5)

                w_change_solve_animator = w_change_eq[2:7].copy()

                self.play(Write(w_change_eq[0]), Write(w_change_eq[1]))
                self.wait(2)
                self.play(w_mover.animate.move_to(w_change_eq[2]))
                self.add(w_change_eq[2]); self.remove(w_mover)
                self.play(Write(w_change_eq[3]))
                self.wait(1)
                self.play(lr_mover.animate.move_to(w_change_eq[4]))
                self.add(w_change_eq[4]); self.remove(lr_mover)
                self.play(Write(w_change_eq[5]))
                self.wait(1)
                self.play(w_gradient_val_mover.animate.move_to(w_change_eq[6]))
                self.add(w_change_eq[6]); self.remove(w_gradient_val_mover)
                self.play(Write(w_change_eq[7]))
                self.wait(1)
                self.play(w_change_solve_animator.animate.become(w_change_eq[8]))
                self.add(w_change_eq[8]); self.remove(w_change_solve_animator)
                self.wait(2)

                if i != 0:
                    w_mover.move_to(weight_groups[i][1])
                    z_i_mover.move_to(z_labels[i][1])
                    dc_da_val_mover.move_to(dc_da_val)

                    dc_da_val = DecimalNumber(w_mover.get_value() * sigmoid_derivative(z_i_val) * dc_da_val.get_value(), num_decimal_places=2)

                    dc_da_eq = VGroup(
                        MathTex(r"\frac{\partial C}{\partial ",f"a_{i}","}"),
                        MathTex(r"="),
                        w_mover.copy(),
                        MathTex(r"\cdot"),
                        MathTex(r"\sigma'("),
                        z_i_mover.copy(),
                        MathTex(r")"),
                        MathTex(r"\cdot"),
                        dc_da_val_mover.copy(),
                        MathTex(r"="),
                        dc_da_val
                    )

                    for j in range(len(dc_dw_i_spacing)):
                        dc_da_eq[j + 1].next_to(dc_da_eq[j], RIGHT, buff=dc_dw_i_spacing[j])

                    dc_da_eq.move_to(DOWN * 2.5)
                    dc_da_solve_animator = dc_da_eq[2:9].copy()

                    self.play(Write(dc_da_eq[0]), Write(dc_da_eq[1]))
                    self.wait(2)
                    self.play(w_mover.animate.move_to(dc_da_eq[2]))
                    self.add(dc_da_eq[2]); self.remove(w_mover)
                    self.play(Write(dc_da_eq[3]))
                    self.wait(1)
                    self.play(FadeIn(dc_da_eq[4], dc_da_eq[6]), z_i_mover.animate.move_to(dc_da_eq[5]))
                    self.add(dc_da_eq[5]); self.remove(z_i_mover)
                    self.play(Write(dc_da_eq[7]))
                    self.wait(1)
                    self.play(dc_da_val_mover.animate.move_to(dc_da_eq[8]))
                    self.add(dc_da_eq[8]); self.remove(dc_da_val_mover)
                    self.play(Write(dc_da_eq[9]))
                    self.wait(1)
                    self.play(dc_da_solve_animator.animate.become(dc_da_eq[10]))
                    self.add(dc_da_eq[10]); self.remove(dc_da_solve_animator)
                    self.wait(3)

                self.play(weight_groups[i][1].animate.become(DecimalNumber(new_w_val.get_value(), num_decimal_places=2).move_to(weight_groups[i][1])))
                self.wait(2)
                self.play(FadeOut(w_gradient_eq, w_change_eq))

        animate_backward_prop(1.00)
        self.wait(2)
        animate_forward_prop(2.00)
        self.wait(2)


def render_manim():
    command = ["manim", "-pql", "Neural Network Basics.py"]
    subprocess.run(command)

if __name__ == "__main__":
    render_manim()