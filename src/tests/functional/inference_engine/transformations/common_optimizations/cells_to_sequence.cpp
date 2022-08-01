// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <transformations/common_optimizations/cells_to_sequence.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph/shape.hpp"

#include <openvino/pass/visualize_tree.hpp>
#include <openvino/pass/manager.hpp>


#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;
using namespace std;


TEST_F(TransformationTestsF, GRUCellsToSequence_inputs_not_shared) {
    {
        const size_t batch_size = 2;
        const size_t input_size = 3;
        const size_t hidden_size = 3;
        const size_t gates_count = 3;

        const auto X_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
        const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
        const auto W_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, input_size}, std::vector<float>{1});
        const auto R_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, hidden_size}, std::vector<float>{2});
        const auto B_a = make_shared<op::Constant>(element::f32, Shape{3 * hidden_size}, std::vector<float>{3});

        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, hidden_size);

        const auto H_t_b = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
        const auto W_b = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, input_size}, std::vector<float>{1});
        const auto R_b = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, hidden_size}, std::vector<float>{2});
        const auto B_b = make_shared<op::Constant>(element::f32, Shape{3 * hidden_size}, std::vector<float>{3});

        const auto gru_cell_b = make_shared<opset8::GRUCell>(gru_cell_a, H_t_b, W_b, R_b, hidden_size);

        function = std::make_shared<Function>(NodeVector{gru_cell_b}, ParameterVector{X_a, Ht_a, H_t_b});
        manager.register_pass<pass::GRUCellsToSequence>();
    }
    // No fusion as W, R, B inputs are not shared
}

TEST_F(TransformationTestsF, GRUCellsToSequence_not_connected) {
    {
        const size_t batch_size = 2;
        const size_t input_size = 3;
        const size_t hidden_size = 3;
        const size_t gates_count = 3;

        const auto X_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
        const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
        const auto W_a = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
        const auto R_a = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});

        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, hidden_size);

        const auto X_b = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
        const auto H_t_b = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
        const auto W_b = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
        const auto R_b = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});

        const auto gru_cell_b = make_shared<opset8::GRUCell>(X_b, H_t_b, W_b, R_b, hidden_size);

        function = std::make_shared<Function>(NodeVector{gru_cell_a, gru_cell_b}, ParameterVector{X_a, Ht_a, W_a, R_a, X_b, H_t_b, W_b, R_b});
        manager.register_pass<pass::GRUCellsToSequence>();
    }
    // No fusion as cells are not connected
}

TEST_F(TransformationTestsF, GRUCellsToSequence_params_connected) {
    const size_t batch_size = 8;
    const size_t input_size = 4;
    const size_t hidden_size = 128;
    const size_t gates_count = 3;
    {
        const auto X_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
        const auto W_a = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
        const auto R_a = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
        const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, hidden_size);

        const auto X_b = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
        const auto W_b = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
        const auto R_b = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
        const auto H_t_b = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

        const auto gru_cell_b = make_shared<opset8::GRUCell>(X_b, gru_cell_a, W_b, R_b, hidden_size);

        function = std::make_shared<Function>(NodeVector{gru_cell_b}, ParameterVector{X_a, Ht_a, W_a, R_a, X_b, W_b, R_b});
        manager.register_pass<pass::GRUCellsToSequence>();
    }
    {
        const size_t num_directions = 1;
        const size_t seq_length = 2;

        const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

        const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
        const auto Ht =
            make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
        const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
        const auto W = make_shared<op::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size, input_size});
        const auto R = make_shared<op::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size, hidden_size});
        const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, 3 * hidden_size});

        const auto direction = op::RecurrentSequenceDirection::FORWARD;
        const auto sequence =
            make_shared<opset8::GRUSequence>(X, Ht, sequence_lengths, W, R, B, hidden_size, direction);

        const auto squeeze_seq = std::make_shared<opset8::Squeeze>(sequence->output(1), axis_1);
        function_ref = std::make_shared<Function>(NodeVector{squeeze_seq}, ParameterVector{X, Ht, sequence_lengths, W, R, B});
    }
}

TEST_F(TransformationTestsF, GRUCellsToSequence_params_connected_few) {
    const size_t batch_size = 8;
    const size_t input_size = 4;
    const size_t hidden_size = 128;
    const size_t gates_count = 3;
    const size_t seq_length = 4;

    NodeVector gru_cells_x;
    ParameterVector gru_params;

    const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});
    const auto X_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    gru_cells_x.push_back(make_shared<op::Unsqueeze>(X_a, axis_1));

    const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W_a = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R_a = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto B_a = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});
    {
        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, B_a, hidden_size);
        NodeVector gru_cells{gru_cell_a};

        gru_params.insert(gru_params.end(), {X_a, Ht_a, W_a, R_a, B_a});

        for (auto i = 0; i < seq_length - 1; ++i) {
            const auto X_n = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
            gru_cells.push_back(make_shared<opset8::GRUCell>(X_n, gru_cells.back(), W_a, R_a, B_a, hidden_size));
            gru_params.push_back(X_n);
            gru_cells_x.push_back(make_shared<op::Unsqueeze>(X_n, axis_1));
        }

        function = std::make_shared<Function>(gru_cells.back(), gru_params);
        manager.register_pass<pass::GRUCellsToSequence>();
    }
    {
        const auto axis_0 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});

        const auto X_seq = make_shared<opset8::Concat>(gru_cells_x, 1);
        const auto Ht_seq = make_shared<op::Unsqueeze>(Ht_a, axis_1);

        const auto sequence_lengths = make_shared<op::Constant>(element::i64, Shape{batch_size}, seq_length);


        // sequence_lengths subgraph:
        // const auto& shape_node = make_shared<opset9::ShapeOf>(gru_cells_x.front());
        // const auto& batch_dimension = make_shared<opset9::Gather>(shape_node,
        //                                 op::Constant::create(ngraph::element::i64, {1}, {0}),
        //                                 axis_0);
        // auto seq_lengths_scalar = op::Constant::create(ngraph::element::i64, {}, {seq_length});
        // auto sequence_lengths = make_shared<opset8::Broadcast>(seq_lengths_scalar, batch_dimension);

        const auto W_seq = make_shared<op::Unsqueeze>(W_a, axis_0);
        const auto R_seq = make_shared<op::Unsqueeze>(R_a, axis_0);
        const auto B_seq = make_shared<op::Unsqueeze>(B_a, axis_0);

        const auto direction = op::RecurrentSequenceDirection::FORWARD;

        const auto sequence =
            make_shared<opset8::GRUSequence>(X_seq, Ht_seq, sequence_lengths, W_seq, R_seq, B_seq, hidden_size, direction);

        const auto squeeze_seq = std::make_shared<opset8::Squeeze>(sequence->output(1), axis_1);
        function_ref = std::make_shared<Function>(NodeVector{squeeze_seq}, gru_params);
    }
}

TEST_F(TransformationTestsF, GRUCellsToSequence_const_connected_few) {
    const size_t batch_size = 8;
    const size_t input_size = 4;
    const size_t hidden_size = 128;
    const size_t gates_count = 3;
    const size_t seq_length = 4;

    NodeVector gru_cells_x;
    ParameterVector gru_params;

    const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});
    const auto X_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    gru_cells_x.push_back(make_shared<op::Unsqueeze>(X_a, axis_1));

    const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, input_size}, std::vector<float>{1});
    const auto R_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, hidden_size}, std::vector<float>{2});
    const auto B_a = make_shared<op::Constant>(element::f32, Shape{3 * hidden_size}, std::vector<float>{3});
    gru_params.insert(gru_params.end(), {X_a, Ht_a});
    {
        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, B_a, hidden_size);
        NodeVector gru_cells{gru_cell_a};

        for (auto i = 0; i < seq_length - 1; ++i) {
            const auto X_n = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
            gru_cells.push_back(make_shared<opset8::GRUCell>(X_n, gru_cells.back(), W_a, R_a, B_a, hidden_size));
            gru_params.push_back(X_n);
            gru_cells_x.push_back(make_shared<op::Unsqueeze>(X_n, axis_1));
        }

        function = std::make_shared<Function>(gru_cells.back(), gru_params);
        manager.register_pass<pass::GRUCellsToSequence>();
        manager.register_pass<ov::pass::VisualizeTree>("after_GRUCellsToSequence_const_connected_few.svg");

        ov::pass::Manager mng;
        mng.register_pass<ov::pass::VisualizeTree>("before_GRUCellsToSequence_const_connected_few.svg");
        mng.run_passes(function);
    }
    {
        const auto axis_0 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});

        const auto X_seq = make_shared<opset8::Concat>(gru_cells_x, 1);
        const auto Ht_seq = make_shared<op::Unsqueeze>(Ht_a, axis_1);

        const auto sequence_lengths = make_shared<op::Constant>(element::i64, Shape{batch_size}, seq_length);

        const auto W_seq = make_shared<op::Unsqueeze>(W_a, axis_0);
        const auto R_seq = make_shared<op::Unsqueeze>(R_a, axis_0);
        const auto B_seq = make_shared<op::Unsqueeze>(B_a, axis_0);

        const auto direction = op::RecurrentSequenceDirection::FORWARD;

        const auto sequence =
            make_shared<opset8::GRUSequence>(X_seq, Ht_seq, sequence_lengths, W_seq, R_seq, B_seq, hidden_size, direction);

        const auto squeeze_seq = std::make_shared<opset8::Squeeze>(sequence->output(1), axis_1);
        function_ref = std::make_shared<Function>(NodeVector{squeeze_seq}, gru_params);

        ov::pass::Manager mng;
        mng.register_pass<ov::pass::VisualizeTree>("ref_GRUCellsToSequence_const_connected_few.svg");
        mng.run_passes(function);
    }
}


TEST_F(TransformationTestsF, GRUCellsToSequence_const_shared_squeeze_connected) {
    const size_t batch_size = 8;
    const size_t input_size = 4;
    const size_t hidden_size = 128;
    const size_t gates_count = 3;
    const size_t seq_length = 4;

    NodeVector gru_cells_x;
    ParameterVector gru_params;
    const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

    const auto X_param = make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto X = make_shared<opset9::Split>(X_param, axis_1, seq_length);

    const auto X_a = make_shared<opset9::Squeeze>(X->output(0), axis_1);
    gru_cells_x.push_back(make_shared<op::Unsqueeze>(X_a, axis_1));

    const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, input_size}, std::vector<float>{1});
    const auto R_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, hidden_size}, std::vector<float>{2});
    const auto B_a = make_shared<op::Constant>(element::f32, Shape{3 * hidden_size}, std::vector<float>{3});
    gru_params.insert(gru_params.end(), {X_param, Ht_a});
    {
        const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, B_a, hidden_size);
        NodeVector gru_cells{gru_cell_a};

        for (auto i = 1; i < seq_length; ++i) {
            const auto X_n = make_shared<opset9::Squeeze>(X->output(i), axis_1);
            gru_cells.push_back(make_shared<opset8::GRUCell>(X_n, gru_cells.back(), W_a, R_a, B_a, hidden_size));
            gru_cells_x.push_back(make_shared<op::Unsqueeze>(X_n, axis_1));
        }

        function = std::make_shared<Function>(gru_cells.back(), gru_params);
        manager.register_pass<pass::GRUCellsToSequence>();
        manager.register_pass<ov::pass::VisualizeTree>("after_GRUCellsToSequence_const_shared_squeeze_connected.svg");

        ov::pass::Manager mng;
        mng.register_pass<ov::pass::VisualizeTree>("before_GRUCellsToSequence_const_shared_squeeze_connected.svg");
        mng.run_passes(function);
    }
    {
        const auto axis_0 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
        const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

        const auto X_seq = make_shared<opset8::Concat>(gru_cells_x, 1);
        const auto Ht_seq = make_shared<op::Unsqueeze>(Ht_a, axis_1);

        const auto sequence_lengths = make_shared<op::Constant>(element::i64, Shape{batch_size}, seq_length);

        const auto W_seq = make_shared<op::Unsqueeze>(W_a, axis_0);
        const auto R_seq = make_shared<op::Unsqueeze>(R_a, axis_0);
        const auto B_seq = make_shared<op::Unsqueeze>(B_a, axis_0);

        const auto direction = op::RecurrentSequenceDirection::FORWARD;

        const auto sequence =
            make_shared<opset8::GRUSequence>(X_seq, Ht_seq, sequence_lengths, W_seq, R_seq, B_seq, hidden_size, direction);

        const auto squeeze_seq = std::make_shared<opset8::Squeeze>(sequence->output(1), axis_1);
        function_ref = std::make_shared<Function>(NodeVector{squeeze_seq}, gru_params);

        ov::pass::Manager mng;
        mng.register_pass<ov::pass::VisualizeTree>("ref_GRUCellsToSequence_const_shared_squeeze_connected.svg");
        mng.run_passes(function_ref);
    }
}

TEST_F(TransformationTestsF, GRUCellsToSequence_two_sequences) {
    const size_t batch_size = 8;
    const size_t input_size = 4;
    const size_t hidden_size = 128;
    const size_t gates_count = 3;
    const size_t seq_length = 4;

    NodeVector gru_cells_x_a;
    NodeVector gru_cells_x_b;

    ParameterVector gru_params;

    const auto axis_0 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
    const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

    const auto X = make_shared<op::Parameter>(element::f32, Shape{2, batch_size, input_size});
    const auto X_ab = make_shared<opset9::Split>(X, axis_0, 2);

    const auto X_a = make_shared<opset9::Squeeze>(X_ab->output(0), axis_0);
    gru_cells_x_a.push_back(make_shared<opset9::Unsqueeze>(X_a, axis_1));

    const auto Ht_a = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, input_size}, std::vector<float>{1});
    const auto R_a = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, hidden_size}, std::vector<float>{2});
    const auto B_a = make_shared<op::Constant>(element::f32, Shape{3 * hidden_size}, std::vector<float>{3});
    gru_params.insert(gru_params.end(), {X, Ht_a});

    const auto X_b = make_shared<opset9::Squeeze>(X_ab->output(1), axis_0);
    gru_cells_x_b.push_back(make_shared<op::Unsqueeze>(X_b, axis_1));

    const auto Ht_b = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W_b = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, input_size}, std::vector<float>{4});
    const auto R_b = make_shared<op::Constant>(element::f32, Shape{gates_count * hidden_size, hidden_size}, std::vector<float>{5});
    const auto B_b = make_shared<op::Constant>(element::f32, Shape{3 * hidden_size}, std::vector<float>{6});
    gru_params.insert(gru_params.end(), {Ht_b});

    {
        const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

        const auto gru_cell_a = make_shared<opset8::GRUCell>(X_a, Ht_a, W_a, R_a, B_a, hidden_size);
        NodeVector gru_cells_a{gru_cell_a};

        for (auto i = 0; i < seq_length - 1; ++i) {
            const auto X_n = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
            gru_cells_a.push_back(make_shared<opset8::GRUCell>(X_n, gru_cells_a.back(), W_a, R_a, B_a, hidden_size));
            gru_params.push_back(X_n);
            gru_cells_x_a.push_back(make_shared<op::Unsqueeze>(X_n, axis_1));
        }

        const auto gru_cell_b = make_shared<opset8::GRUCell>(X_b, Ht_b, W_b, R_b, B_b, hidden_size);
        NodeVector gru_cells_b{gru_cell_b};

        for (auto i = 0; i < seq_length - 1; ++i) {
            const auto X_n = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
            gru_cells_b.push_back(make_shared<opset8::GRUCell>(X_n, gru_cells_b.back(), W_b, R_b, B_b, hidden_size));
            gru_params.push_back(X_n);
            gru_cells_x_b.push_back(make_shared<op::Unsqueeze>(X_n, axis_1));
        }

        function = std::make_shared<Function>(NodeVector{gru_cells_a.back(), gru_cells_b.back()}, gru_params);

        ov::pass::Manager mng;
        mng.register_pass<ov::pass::VisualizeTree>("before_GRUCellsToSequence_two_sequences.svg");
        mng.run_passes(function);

        manager.register_pass<pass::GRUCellsToSequence>();
        manager.register_pass<ov::pass::VisualizeTree>("after_GRUCellsToSequence_two_sequences.svg");
    }
    {
        const auto axis_0 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
        const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

        // I Sequence
        const auto X_seq = make_shared<opset8::Concat>(gru_cells_x_a, 1);
        const auto Ht_seq = make_shared<op::Unsqueeze>(Ht_a, axis_1);

        const auto sequence_lengths = make_shared<op::Constant>(element::i64, Shape{batch_size}, seq_length);

        const auto W_seq = make_shared<op::Unsqueeze>(W_a, axis_0);
        const auto R_seq = make_shared<op::Unsqueeze>(R_a, axis_0);
        const auto B_seq = make_shared<op::Unsqueeze>(B_a, axis_0);

        const auto direction = op::RecurrentSequenceDirection::FORWARD;

        const auto sequence =
            make_shared<opset8::GRUSequence>(X_seq, Ht_seq, sequence_lengths, W_seq, R_seq, B_seq, hidden_size, direction);

        const auto squeeze_seq = std::make_shared<opset8::Squeeze>(sequence->output(1), axis_1);

        // II Sequence
        const auto axis_0_b = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
        const auto axis_1_b = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

        const auto X_seq_b = make_shared<opset8::Concat>(gru_cells_x_b, 1);
        const auto Ht_seq_b = make_shared<opset8::Unsqueeze>(Ht_b, axis_1_b);

        const auto W_seq_b = make_shared<opset8::Unsqueeze>(W_b, axis_0_b);
        const auto R_seq_b = make_shared<opset8::Unsqueeze>(R_b, axis_0_b);
        const auto B_seq_b = make_shared<opset8::Unsqueeze>(B_b, axis_0_b);

        const auto sequence_lengths_b = make_shared<op::Constant>(element::i64, Shape{batch_size}, seq_length);

        const auto sequence_b =
            make_shared<opset8::GRUSequence>(X_seq_b, Ht_seq_b, sequence_lengths_b, W_seq_b, R_seq_b, B_seq_b, hidden_size, direction);

        const auto squeeze_seq_b = std::make_shared<opset8::Squeeze>(sequence_b->output(1), axis_1_b);
        function_ref = std::make_shared<Function>(NodeVector{squeeze_seq, squeeze_seq_b}, gru_params);

        ov::pass::Manager mng;
        mng.register_pass<ov::pass::VisualizeTree>("ref_GRUCellsToSequence_two_sequences.svg");
        mng.run_passes(function);
    }
}
