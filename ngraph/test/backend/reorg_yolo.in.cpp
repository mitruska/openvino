//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, reorg_yolo_match_original_output_stride_2)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 8, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, Strides{stride});
    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs(128);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  2,  4,  6,  16, 18, 20, 22, 32,  34,  36,  38,  48,  50,  52,  54,
        64, 66, 68, 70, 80, 82, 84, 86, 96,  98,  100, 102, 112, 114, 116, 118,
        1,  3,  5,  7,  17, 19, 21, 23, 33,  35,  37,  39,  49,  51,  53,  55,
        65, 67, 69, 71, 81, 83, 85, 87, 97,  99,  101, 103, 113, 115, 117, 119,
        8,  10, 12, 14, 24, 26, 28, 30, 40,  42,  44,  46,  56,  58,  60,  62,
        72, 74, 76, 78, 88, 90, 92, 94, 104, 106, 108, 110, 120, 122, 124, 126,
        9,  11, 13, 15, 25, 27, 29, 31, 41,  43,  45,  47,  57,  59,  61,  63,
        73, 75, 77, 79, 89, 91, 93, 95, 105, 107, 109, 111, 121, 123, 125, 127};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reorg_yolo_match_original_output_stride_3)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 9, 3, 3};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 3;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, Strides{stride});
    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs(81);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  3,  6,  27, 30, 33, 54, 57, 60, 1,  4,  7,  28, 31, 34, 55, 58, 61, 2,  5,  8,
        29, 32, 35, 56, 59, 62, 9,  12, 15, 36, 39, 42, 63, 66, 69, 10, 13, 16, 37, 40, 43,
        64, 67, 70, 11, 14, 17, 38, 41, 44, 65, 68, 71, 18, 21, 24, 45, 48, 51, 72, 75, 78,
        19, 22, 25, 46, 49, 52, 73, 76, 79, 20, 23, 26, 47, 50, 53, 74, 77, 80};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reorg_yolo_match_eip_output_stride_2)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 8, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, Strides{stride});
    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs(128);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  2,  8,  10, 16, 18, 24, 26, 32,  34,  40,  42,  48,  50,  56,  58,
        64, 66, 72, 74, 80, 82, 88, 90, 96,  98,  104, 106, 112, 114, 120, 122,
        1,  3,  9,  11, 17, 19, 25, 27, 33,  35,  41,  43,  49,  51,  57,  59,
        65, 67, 73, 75, 81, 83, 89, 91, 97,  99,  105, 107, 113, 115, 121, 123,
        4,  6,  12, 14, 20, 22, 28, 30, 36,  38,  44,  46,  52,  54,  60,  62,
        68, 70, 76, 78, 84, 86, 92, 94, 100, 102, 108, 110, 116, 118, 124, 126,
        5,  7,  13, 15, 21, 23, 29, 31, 37,  39,  45,  47,  53,  55,  61,  63,
        69, 71, 77, 79, 85, 87, 93, 95, 101, 103, 109, 111, 117, 119, 125, 127};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reorg_yolo_match_eip_output_stride_3)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 9, 3, 3};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 3;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, Strides{stride});
    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs(81);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  9,  18, 27, 36, 45, 54, 63, 72, 1,  10, 19, 28, 37, 46, 55, 64, 73, 2,  11, 20,
        29, 38, 47, 56, 65, 74, 3,  12, 21, 30, 39, 48, 57, 66, 75, 4,  13, 22, 31, 40, 49,
        58, 67, 76, 5,  14, 23, 32, 41, 50, 59, 68, 77, 6,  15, 24, 33, 42, 51, 60, 69, 78,
        7,  16, 25, 34, 43, 52, 61, 70, 79, 8,  17, 26, 35, 44, 53, 62, 71, 80};

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, eip_match_original_reorg_yolo_output_stride_2)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 8, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto eip = make_shared<op::v3::ExtractImagePatches>(
        p, Shape{stride, stride}, Strides{stride, stride}, Shape{1, 1}, op::PadType::VALID);

    auto fun = make_shared<Function>(OutputVector{eip}, ParameterVector{p});

    std::vector<float> inputs(128);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  2,  4,  6,  16, 18, 20, 22, 32,  34,  36,  38,  48,  50,  52,  54,
        64, 66, 68, 70, 80, 82, 84, 86, 96,  98,  100, 102, 112, 114, 116, 118,
        1,  3,  5,  7,  17, 19, 21, 23, 33,  35,  37,  39,  49,  51,  53,  55,
        65, 67, 69, 71, 81, 83, 85, 87, 97,  99,  101, 103, 113, 115, 117, 119,
        8,  10, 12, 14, 24, 26, 28, 30, 40,  42,  44,  46,  56,  58,  60,  62,
        72, 74, 76, 78, 88, 90, 92, 94, 104, 106, 108, 110, 120, 122, 124, 126,
        9,  11, 13, 15, 25, 27, 29, 31, 41,  43,  45,  47,  57,  59,  61,  63,
        73, 75, 77, 79, 89, 91, 93, 95, 105, 107, 109, 111, 121, 123, 125, 127};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, eip_match_original_reorg_yolo_output_stride_3)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 9, 3, 3};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 3;
    auto eip = make_shared<op::v3::ExtractImagePatches>(
        p, Shape{stride, stride}, Strides{stride, stride}, Shape{1, 1}, op::PadType::VALID);

    auto fun = make_shared<Function>(OutputVector{eip}, ParameterVector{p});

    std::vector<float> inputs(81);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  3,  6,  27, 30, 33, 54, 57, 60, 1,  4,  7,  28, 31, 34, 55, 58, 61, 2,  5,  8,
        29, 32, 35, 56, 59, 62, 9,  12, 15, 36, 39, 42, 63, 66, 69, 10, 13, 16, 37, 40, 43,
        64, 67, 70, 11, 14, 17, 38, 41, 44, 65, 68, 71, 18, 21, 24, 45, 48, 51, 72, 75, 78,
        19, 22, 25, 46, 49, 52, 73, 76, 79, 20, 23, 26, 47, 50, 53, 74, 77, 80};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, eip_match_custom_reorg_yolo_output_stride_2)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 8, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto eip = make_shared<op::v3::ExtractImagePatches>(
        p, Shape{stride, stride}, Strides{stride, stride}, Shape{1, 1}, op::PadType::VALID);

    auto fun = make_shared<Function>(OutputVector{eip}, ParameterVector{p});

    std::vector<float> inputs(128);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  2,  8,  10, 16, 18, 24, 26, 32,  34,  40,  42,  48,  50,  56,  58,
        64, 66, 72, 74, 80, 82, 88, 90, 96,  98,  104, 106, 112, 114, 120, 122,
        1,  3,  9,  11, 17, 19, 25, 27, 33,  35,  41,  43,  49,  51,  57,  59,
        65, 67, 73, 75, 81, 83, 89, 91, 97,  99,  105, 107, 113, 115, 121, 123,
        4,  6,  12, 14, 20, 22, 28, 30, 36,  38,  44,  46,  52,  54,  60,  62,
        68, 70, 76, 78, 84, 86, 92, 94, 100, 102, 108, 110, 116, 118, 124, 126,
        5,  7,  13, 15, 21, 23, 29, 31, 37,  39,  45,  47,  53,  55,  61,  63,
        69, 71, 77, 79, 85, 87, 93, 95, 101, 103, 109, 111, 117, 119, 125, 127};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, eip_match_custom_reorg_yolo_output_stride_3)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 9, 3, 3};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 3;
    auto eip = make_shared<op::v3::ExtractImagePatches>(
        p, Shape{stride, stride}, Strides{stride, stride}, Shape{1, 1}, op::PadType::VALID);

    auto fun = make_shared<Function>(OutputVector{eip}, ParameterVector{p});

    std::vector<float> inputs(81);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
        0,  9,  18, 27, 36, 45, 54, 63, 72, 1,  10, 19, 28, 37, 46, 55, 64, 73, 2,  11, 20,
        29, 38, 47, 56, 65, 74, 3,  12, 21, 30, 39, 48, 57, 66, 75, 4,  13, 22, 31, 40, 49,
        58, 67, 76, 5,  14, 23, 32, 41, 50, 59, 68, 77, 6,  15, 24, 33, 42, 51, 60, 69, 78,
        7,  16, 25, 34, 43, 52, 61, 70, 79, 8,  17, 26, 35, 44, 53, 62, 71, 80};
    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, reorg_yolo_catch_small_shape_stride)
{
    // Throw error test: For [N, C, H, W] input shape, C >= (stride*stride) is required.
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 1, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, Strides{stride});
    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> fake_output(16);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    try
    {
        auto test_case = test::TestCase<TestEngine>(fun);
        test_case.add_input<float>(inputs);
        test_case.add_expected_output<float>(expected_shape, fake_output);
        test_case.run();

        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible stride was not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("stride"));
    }
    catch (...)
    {
        FAIL() << "Stride size check failed for unexpected reason.";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, custom_reorg_yolo_small_shape_stride)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 1, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, Strides{stride});

    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> expected_result{0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15};

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, eip_match_custom_reorg_yolo_small_shape_stride)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 1, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    size_t stride = 2;
    auto eip = make_shared<op::v3::ExtractImagePatches>(
        p, Shape{stride, stride}, Strides{stride, stride}, Shape{1, 1}, op::PadType::VALID);

    auto fun = make_shared<Function>(OutputVector{eip}, ParameterVector{p});

    std::vector<float> inputs{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> expected_result{0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15};

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(inputs);
    test_case.add_expected_output<float>(expected_shape, expected_result);
    test_case.run();
}
