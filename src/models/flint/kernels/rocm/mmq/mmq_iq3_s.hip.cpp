// SPDX-License-Identifier: MIT
// MMQ instantiation for GGML_TYPE_IQ3_S, one translation unit per weight type.
#include "flint_mmq_prelude.h"
namespace flint_mmq {
#include "mmq.hip.hpp"
DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
}  // namespace flint_mmq
