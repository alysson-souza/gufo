// SPDX-License-Identifier: MIT
// MMQ instantiation for GGML_TYPE_IQ4_XS, one translation unit per weight type.
#include "flint_mmq_prelude.h"
namespace flint_mmq {
#include "mmq.hip.hpp"
DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
}  // namespace flint_mmq
