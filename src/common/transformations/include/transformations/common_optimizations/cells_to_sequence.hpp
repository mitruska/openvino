// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/util.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

// class TRANSFORMATIONS_API SimplifyShapeOfSubGraph;
// class TRANSFORMATIONS_API SharedShapeOf;

class TRANSFORMATIONS_API GRUCellsToSequence;
class TRANSFORMATIONS_API CellsToSequence;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GRUCellsToSequence transformation fuse many GRUCells ops to Sequence
 */
class ngraph::pass::GRUCellsToSequence : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("GRUCellsToSequence", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};

/**
 * @ingroup ie_transformation_common_api
 * @brief CellsToSequence transformation fuse many Cells ops to Sequence
 */
class ngraph::pass::CellsToSequence : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("CellsToSequence", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
