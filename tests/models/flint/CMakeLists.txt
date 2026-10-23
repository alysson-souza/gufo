# Small CPU and operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(flint_tests)

function(add_flint_test name library labels)
  set(target flint_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME flint.${name} COMMAND ${target})
  set_tests_properties(flint.${name} PROPERTIES LABELS "flint;${labels}")
  add_dependencies(flint_tests ${target})
endfunction()

add_flint_test(config gufo_flint "cpu")

# Slice A (text): tokenizer, chat template and tool-call syntax. The tokenizer
# goldens need the GGUF vocabulary and skip without GUFO_FLINT_MODEL.
add_flint_test(tokenizer gufo_flint "cpu;tokenization;external-model")
target_compile_definitions(flint_tokenizer_test PRIVATE
  GUFO_FLINT_TOKENIZER_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_goldens.json")
set_tests_properties(flint.tokenizer PROPERTIES SKIP_RETURN_CODE 77)

add_flint_test(chat_template gufo_flint "cpu;tokenization")
target_compile_definitions(flint_chat_template_test PRIVATE
  GUFO_FLINT_CHAT_TEMPLATE_REFERENCE="${PROJECT_SOURCE_DIR}/src/models/flint/reference/chat_template.jinja"
  GUFO_FLINT_CHAT_TEMPLATE_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_template_goldens.json")

add_flint_test(tool_calls gufo_flint "cpu;serving")
target_sources(flint_tool_calls_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/flint/tool_calls.cpp)

# Slice B (reference): tensor binding against the real artifact (skips
# without GUFO_FLINT_MODEL) and the scalar-oracle logit probe.
add_flint_test(weights gufo_flint "cpu;external-model")
set_tests_properties(flint.weights PROPERTIES SKIP_RETURN_CODE 77)

add_executable(flint_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(flint_reference_probe PRIVATE gufo_flint_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES
  COMPILE_OPTIONS "-O2")

# Slice C (kernels): HIP operator checks against CPU oracles (core::quant
# decoding, double accumulation).
if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS attention_ops batch_invariance_ops elementwise_ops
      moe_ops projection_ops router_ops split_invariance_ops)
    add_flint_test(${name} gufo_flint "hip;gpu;gfx1151")
  endforeach()
  # The same checks over the real artifact's tensors; skips without
  # GUFO_FLINT_MODEL.
  add_flint_test(artifact_ops gufo_flint "hip;gpu;gfx1151;external-model")
  set_tests_properties(flint.artifact_ops PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(flint_kernels_bench EXCLUDE_FROM_ALL kernels_bench.cpp)
  target_link_libraries(flint_kernels_bench PRIVATE gufo_flint)
endif()

# Serving (runner, scheduler, snapshots, disk cache) against the real model.
# Built explicitly like Flash-Next's session test; skips (77) without
# GUFO_FLINT_MODEL.
if (ENGINE_ENABLE_HIP)
  add_executable(flint_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(flint_session_test PRIVATE gufo_logging gufo_core
    gufo_flint gufo_deepseek_v4_flash gufo_qwen38_flash_next gufo_cache_hip
    hip::host)
endif()

# Wave 2 (engine): GPU logit probe for quality qualification against the CPU
# reference and llama.cpp dumps (tools/flint/compare_logits.py).
if (ENGINE_ENABLE_HIP)
  add_executable(flint_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp)
  target_link_libraries(flint_gpu_probe PRIVATE gufo_flint)
endif()
