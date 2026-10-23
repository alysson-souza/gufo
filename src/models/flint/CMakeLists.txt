# Recursive Flint (flint) model package. The scalar oracle is built only by
# explicit validation targets, not by production or routine checks.
add_library(gufo_flint STATIC
  config.cpp)

target_include_directories(gufo_flint PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_flint PUBLIC gufo_core Threads::Threads)

# Slice A (text): tokenizer and compiled chat template. The tool-call scanner
# is compiled into the HTTP layer and its tests, as DeepSeek's is.
target_sources(gufo_flint PRIVATE
  tokenizer.cpp
  chat_template.cpp)

# Slice B (reference): tensor binding, plus the scalar FP32 oracle. The
# oracle decodes every weight on the fly; unoptimized it is unusable even as
# an oracle, so it keeps -O2 in every configuration.
target_sources(gufo_flint PRIVATE weights.cpp)
add_library(gufo_flint_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_flint_reference PUBLIC gufo_flint)
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

# Slice C (kernels): HIP device primitives. The quantized matrix tier is
# adapted from llama.cpp, isolated in the flint_mmq namespace and keeps its
# own C++17 dialect; see kernels/rocm/mmq/VENDOR.md.
if (ENGINE_ENABLE_HIP)
  set(FLINT_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(FLINT_MMQ_DIR "${FLINT_ROCM_DIR}/mmq")
  set(FLINT_MMQ_SOURCES
    ${FLINT_MMQ_DIR}/flint_ggml_stubs.hip.cpp
    ${FLINT_MMQ_DIR}/flint_mmq.hip.cpp
    ${FLINT_MMQ_DIR}/mmid.hip.cpp
    ${FLINT_MMQ_DIR}/mmq_iq3_s.hip.cpp
    ${FLINT_MMQ_DIR}/mmq_iq4_xs.hip.cpp
    ${FLINT_MMQ_DIR}/mmq_q4_k.hip.cpp
    ${FLINT_MMQ_DIR}/mmq_q5_k.hip.cpp
    ${FLINT_MMQ_DIR}/mmq_q6_k.hip.cpp
    ${FLINT_MMQ_DIR}/mmq_q8_0.hip.cpp
    ${FLINT_MMQ_DIR}/mmvq.hip.cpp
    ${FLINT_MMQ_DIR}/quantize.hip.cpp)
  add_library(gufo_flint_mmq STATIC ${FLINT_MMQ_SOURCES})
  set_source_files_properties(${FLINT_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_flint_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_flint_mmq PRIVATE
    GGML_USE_HIP GGML_HIP_NO_VMM FLINT_ROCM_WMMA_MMQ_Y=64)
  # No fast-math or FP contraction: every template instance (row count, tile
  # width, fused or not) then evaluates a row with identical operations, so
  # batched decode replays single-row decode and split prefill replays one
  # pass bit for bit; the Q8_1 encoders fused into Flint's own kernels also
  # reproduce quantize.hip.cpp's bytes (exact division).
  target_compile_options(gufo_flint_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_flint_mmq SYSTEM PUBLIC ${FLINT_MMQ_DIR})
  target_link_libraries(gufo_flint_mmq PUBLIC hip::host roc::hipblas)

  # Flint's own device operators are compiled without fast-math so every
  # product, FMA and transcendental keeps the documented FP32 contract.
  target_sources(gufo_flint PRIVATE ${FLINT_ROCM_DIR}/kernels.hip.cpp)
  set_source_files_properties(${FLINT_ROCM_DIR}/kernels.hip.cpp PROPERTIES
    LANGUAGE HIP
    COMPILE_OPTIONS "-O3;-ffp-contract=off;-Wno-unused-command-line-argument")
  target_link_libraries(gufo_flint PUBLIC gufo_flint_mmq hip::host)
endif()

# Wave 2 (engine): weight upload, executor, sessions and snapshots.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_flint PRIVATE
    ${FLINT_ROCM_DIR}/device_model.cpp
    ${FLINT_ROCM_DIR}/executor.cpp
    engine.cpp)
endif()
