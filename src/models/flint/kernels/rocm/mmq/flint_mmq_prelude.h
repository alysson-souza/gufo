// Every system header the vendored tier includes, pulled in at global scope
// before the tier is wrapped in namespace flint_mmq. The tier's own includes
// then hit include guards, so its definitions land in the namespace while
// the runtime headers stay global. The wrap keeps this copy from clashing
// with the other models' copies of the same tier linked into one binary.
#pragma once
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
