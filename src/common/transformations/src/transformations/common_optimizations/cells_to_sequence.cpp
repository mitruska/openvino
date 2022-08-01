// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <transformations/common_optimizations/cells_to_sequence.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "itt.hpp"

using namespace ngraph;
using namespace std;

bool ngraph::pass::GRUCellsToSequence::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(GRUCellsToSequence);
    bool graph_rewritten = false;

    const auto X_p = ngraph::pattern::any_input();
    const auto Ht_p = ngraph::pattern::any_input();
    const auto W_p = ngraph::pattern::any_input();
    const auto R_p = ngraph::pattern::any_input();
    const auto B_p = ngraph::pattern::any_input();

    const auto gru_cell = ngraph::pattern::wrap_type<ngraph::opset9::GRUCell>({X_p, Ht_p, W_p, R_p, B_p});
    const auto m = std::make_shared<ngraph::pattern::Matcher>(gru_cell, "GRUCellsMatcher");

    const auto axis_0 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
    const auto axis_1 = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

    while (1) {
        int64_t cells_count = 0;
        NodeVector gru_cell_nodes;
        NodeVector gru_cell_x_inputs;
        std::shared_ptr<ngraph::opset9::GRUCell> first_gru_cell;
        bool has_first = false;

        std::vector<std::shared_ptr<ngraph::Node>> nodes(f->get_ordered_ops());
        std::vector<std::shared_ptr<ngraph::Node>> rev_ordered_ops(nodes.rbegin(), nodes.rend());

        for (const auto& node : nodes) {
            if (!m->match(node)) {
                continue;
            }

            if (!has_first) {
                first_gru_cell = std::dynamic_pointer_cast<ngraph::opset9::GRUCell>(node);

                gru_cell_nodes.push_back(node);
                gru_cell_x_inputs.push_back(
                    make_shared<op::Unsqueeze>(first_gru_cell->get_input_source_output(0), axis_1));
                // replace_output_update_name(first_gru_cell->get_input_source_output(0),
                // gru_cell_x_inputs.back()->input_value(0));

                cells_count++;

                if (first_gru_cell) {
                    has_first = true;
                    break;
                }
            }

            // const auto& patternMap = m->get_pattern_value_map();
        }
        if (!first_gru_cell)
            return false;

        // get_sequence_end ->
        // (https://github.com/ceciliapeng2011/openvino/blob/cecilia/cpu/augru/src/plugins/intel_cpu/src/ngraph_transformations/detect_augrucell.cpp)
        std::function<std::shared_ptr<ngraph::opset9::GRUCell>(std::shared_ptr<ngraph::opset9::GRUCell>)>
            get_sequence_end;
        get_sequence_end = [&](std::shared_ptr<ngraph::opset9::GRUCell> gru_cell) {
            // auto consumers = gru_cell->output(0).get_target_inputs();
            gru_cell_nodes.push_back(gru_cell);

            for (auto& consumer : gru_cell->output(0).get_target_inputs()) {  // consumers size required to be 1?
                const auto consumer_cell =
                    std::dynamic_pointer_cast<ngraph::opset9::GRUCell>(consumer.get_node()->shared_from_this());
                if (consumer_cell) {
                    gru_cell_nodes.push_back(consumer_cell);
                    // const auto& consumer_cell_x = consumer_cell->get_input_source_output(0).get_node_shared_ptr();
                    gru_cell_x_inputs.push_back(
                        make_shared<op::Unsqueeze>(consumer_cell->get_input_source_output(0), axis_1));

                    // ngraph::replace_node(consumer_cell->input_value(0), gru_cell_x_inputs.back());
                    // replace_source_output
                    // replace_output_update_name(consumer_cell->output(0), consumer_cell->input_value(1));
                    // replace_output_update_name(consumer_cell->get_input_source_output(0),
                    // gru_cell_x_inputs.back()->input_value(0));
                    cells_count++;
                    return get_sequence_end(consumer_cell);
                    // replace_output_update_name(consumer_cell->output(0), consumer_cell->input_value(1));
                }
                // replace_output_update_name(gru_cell->output(0), gru_cell->input_value(1));
            }
            // no consumer of grucellnode type. then itself is the last.
            return gru_cell;
        };

        auto last_gru_cell = get_sequence_end(first_gru_cell);

        // if (last_gru_cell == first_gru_cell) {
        //     continue;
        //     // return false; // continue?
        // }

        const size_t hidden_size =
            first_gru_cell->get_hidden_size();  // TODO: check params in the loop // lambda for attributes comparison
        const auto direction = op::RecurrentSequenceDirection::FORWARD;

        const auto& pattern_to_output = m->get_pattern_value_map();

        const auto& shape_node = ngraph::op::util::make_try_fold<opset9::ShapeOf>(pattern_to_output.at(X_p));
        const auto& batch_dimension =
            ngraph::op::util::make_try_fold<opset9::Gather>(shape_node,
                                                            op::Constant::create(ngraph::element::i64, {1}, {0}),
                                                            axis_0);
        auto seq_lengths_scalar = op::Constant::create(ngraph::element::i64, {}, {gru_cell_nodes.size()});
        auto sequence_lengths_in =
            ngraph::op::util::make_try_fold<opset8::Broadcast>(seq_lengths_scalar, batch_dimension);

        const auto X_in = make_shared<op::Concat>(gru_cell_x_inputs, 1);  // Concat all X at axis 1
        const auto Ht_in = make_shared<op::Unsqueeze>(pattern_to_output.at(Ht_p), axis_1);

        const auto W_in =
            make_shared<op::Unsqueeze>(pattern_to_output.at(W_p), axis_0);  // TODO: Check if the same for all nodes
        const auto R_in = make_shared<op::Unsqueeze>(pattern_to_output.at(R_p), axis_0);
        const auto B_in = make_shared<op::Unsqueeze>(pattern_to_output.at(B_p), axis_0);

        const auto gru_sequence = make_shared<opset8::GRUSequence>(X_in,
                                                                   Ht_in,
                                                                   sequence_lengths_in,
                                                                   W_in,
                                                                   R_in,
                                                                   B_in,
                                                                   hidden_size,
                                                                   direction,
                                                                   first_gru_cell->get_activations(),
                                                                   first_gru_cell->get_activations_alpha(),
                                                                   first_gru_cell->get_activations_beta(),
                                                                   first_gru_cell->get_clip(),
                                                                   first_gru_cell->get_linear_before_reset());

        auto squeeze_sequence = std::make_shared<opset8::Squeeze>(gru_sequence->output(1), axis_1);
        squeeze_sequence->set_friendly_name(last_gru_cell->get_friendly_name());

        ov::copy_runtime_info(gru_cell_nodes,
                              {squeeze_sequence,
                               gru_sequence,
                               X_in,
                               Ht_in,
                               sequence_lengths_in,
                               W_in,
                               R_in,
                               B_in,
                               shape_node,
                               batch_dimension});
        ov::copy_runtime_info(gru_cell_nodes, gru_cell_x_inputs);

        // replace_output_update_name(first_gru_cell->get_input_source_output(0), last_gru_cell->input_value(1));

        replace_node(last_gru_cell, squeeze_sequence);

        graph_rewritten = true;
    }
    return graph_rewritten;
}

bool ngraph::pass::CellsToSequence::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // RUN_ON_FUNCTION_SCOPE(CellsToSequence);
    // ngraph::pass::Manager manager;
    // // manager.set_per_pass_validation(false);
    // manager.register_pass<ngraph::pass::GRUCellsToSequence>();
    // manager.run_passes(f);
    return false;
}
