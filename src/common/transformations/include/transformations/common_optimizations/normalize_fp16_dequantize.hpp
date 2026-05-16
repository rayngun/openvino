// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NormalizeDequantizeFP16;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief NormalizeDequantizeFP16 rewrites a Dequantize subgraph that operates in
 * FP16 into an equivalent FP32 subgraph followed by a single FP16 cast:
 *
 *   Convert(int->f16) -> [Subtract(f16_zp)] -> Multiply(f16_scale)
 *
 * becomes:
 *
 *   Convert(int->f32) -> [Subtract(f32_zp)] -> Multiply(f32_scale) -> Convert(f32->f16)
 *
 * The surrounding FakeQuantize node is left unchanged.
 */
class ov::pass::NormalizeDequantizeFP16 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeDequantizeFP16");
    NormalizeDequantizeFP16();
};
