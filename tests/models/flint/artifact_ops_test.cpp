// Projections and routed stages over the real Flint artifact's tensors
// (GUFO_FLINT_MODEL), against core::quant decoding with double
// accumulation. Skips (77) without the model.
//
// Tolerances follow projection_ops_test.cpp: Q8_1-exact activations leave
// FP32 accumulation (1e-4 of sum|w x|) plus, for Q4_K/Q5_K MMQ tiles, the
// FP16 sub-block scales (QuantTensor::Tolerance). Real routes come from the
// layer's own router on the same activations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using test::Checker;
using test::DeviceBuffer;
using test::QuantTensor;

std::vector<std::uint8_t> Bytes(const gufo::core::GgufTensorInfo& t,
                                std::size_t count) {
  const auto* p = static_cast<const std::uint8_t*>(t.data);
  return std::vector<std::uint8_t>(p, p + count);
}

/// Dense projection over the first `rows_used` rows of `info`.
void CheckDense(const gufo::core::GgufTensorInfo& info, rocm::WeightType type,
                int rows_used, std::mt19937& rng, Checker& check) {
  const int k = static_cast<int>(info.dimensions[0]);
  const std::size_t row_bytes = rocm::WeightRowBytes(type, k);
  QuantTensor w(type, 1, rows_used, k,
                Bytes(info, row_bytes * static_cast<std::size_t>(rows_used)));
  for (int rows : {1, 9, 33}) {
    const std::vector<float> x = test::ExactActivations(rows, k, rng);
    DeviceBuffer<float> dx{std::span<const float>(x)};
    DeviceBuffer<float> dy(std::size_t(rows) * rows_used);
    dy.Fill(0xff);
    const std::size_t bytes = rocm::ProjectionScratchBytes(rows, k);
    DeviceBuffer<std::uint8_t> scratch(bytes);
    test::CheckHip(rocm::Project(w.View(), dx.get(), dy.get(), rows,
                                 scratch.get(), bytes, nullptr),
                   "Project");
    test::CheckHip(hipDeviceSynchronize(), "sync");
    const std::vector<float> y = dy.Download();
    int bad = 0;
    for (int r = 0; r < rows; ++r) {
      const float* xr = &x[std::size_t(r) * k];
      for (int o = 0; o < rows_used; ++o) {
        const test::Dot ref = test::DotRow(w.Row(0, o), xr, k);
        const double tol = w.Tolerance(0, o, xr, rows > rocm::kVectorRows, ref);
        bad +=
            !(std::fabs(y[std::size_t(r) * rows_used + o] - ref.value) <= tol);
      }
    }
    std::ostringstream os;
    os << info.name << " (" << test::Name(type) << ") rows=" << rows
       << " mismatches=" << bad;
    check.Expect(bad == 0, os.str());
  }
}

/// Routed gate and down stages of one MoE layer on real routes.
void CheckRouted(const gufo::core::GgufReader& reader, int layer,
                 std::mt19937& rng, Checker& check) {
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const auto* router = reader.FindTensor(prefix + "ffn_gate_inp.weight");
  const auto* gate_info = reader.FindTensor(prefix + "ffn_gate_exps.weight");
  const auto* down_info = reader.FindTensor(prefix + "ffn_down_exps.weight");
  if (router == nullptr || gate_info == nullptr || down_info == nullptr) {
    check.Expect(false, prefix + " MoE tensors missing");
    return;
  }
  const auto gate_type =
      rocm::WeightTypeFromGgml(static_cast<std::uint32_t>(gate_info->type));
  const auto down_type =
      rocm::WeightTypeFromGgml(static_cast<std::uint32_t>(down_info->type));
  if (!gate_type || !down_type) {
    check.Expect(false, prefix + " unsupported expert type");
    return;
  }
  const int hidden = static_cast<int>(gate_info->dimensions[0]);
  const int inner = static_cast<int>(gate_info->dimensions[1]);
  const int experts = static_cast<int>(gate_info->dimensions[2]);
  constexpr int kUsed = 6;
  QuantTensor gate(*gate_type, experts, inner, hidden,
                   Bytes(*gate_info, gate_info->size_bytes));
  QuantTensor down(*down_type, experts, hidden, inner,
                   Bytes(*down_info, down_info->size_bytes));
  DeviceBuffer<std::uint16_t> dr{std::span<const std::uint16_t>(
      static_cast<const std::uint16_t*>(router->data),
      router->size_bytes / sizeof(std::uint16_t))};
  for (int tokens : {1, 3, 9, 40}) {
    const std::vector<float> x = test::ExactActivations(tokens, hidden, rng);
    DeviceBuffer<float> dx{std::span<const float>(x)};
    DeviceBuffer<std::int32_t> ids(std::size_t(tokens) * kUsed);
    DeviceBuffer<float> weights(std::size_t(tokens) * kUsed);
    const std::size_t rb = rocm::RouterScratchBytes(tokens, experts);
    DeviceBuffer<std::uint8_t> rs(rb);
    test::CheckHip(rocm::RouterTopK(dr.get(), dx.get(), nullptr, 1.0f, tokens,
                                    hidden, experts, kUsed, ids.get(),
                                    weights.get(), rs.get(), rb, nullptr),
                   "RouterTopK");
    const std::vector<std::int32_t> hids = ids.Download();
    const bool mmq = tokens > rocm::kVectorRows;
    // Gate stage on the token rows, down stage on exact per-slot rows.
    for (const bool per_slot : {false, true}) {
      QuantTensor& w = per_slot ? down : gate;
      const std::vector<float> in =
          per_slot
              ? test::ExactActivations(std::size_t(tokens) * kUsed, inner, rng)
              : x;
      DeviceBuffer<float> din{std::span<const float>(in)};
      DeviceBuffer<float> out(std::size_t(tokens) * kUsed * w.m);
      out.Fill(0xff);
      const std::size_t bytes = rocm::RoutedProjectScratchBytes(
          w.View(), experts, tokens, kUsed, per_slot);
      DeviceBuffer<std::uint8_t> scratch(bytes);
      test::CheckHip(rocm::RoutedProject(w.View(), experts, din.get(), per_slot,
                                         ids.get(), tokens, kUsed, out.get(),
                                         scratch.get(), bytes, nullptr),
                     "RoutedProject");
      test::CheckHip(hipDeviceSynchronize(), "sync");
      const std::vector<float> y = out.Download();
      int bad = 0;
      for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < kUsed; ++j) {
          const int e = hids[std::size_t(t) * kUsed + j];
          const float* xr =
              &in[(per_slot ? std::size_t(t) * kUsed + j : std::size_t(t)) *
                  w.k];
          for (int o = 0; o < w.m; ++o) {
            const test::Dot ref = test::DotRow(w.Row(e, o), xr, w.k);
            bad += !(std::fabs(y[(std::size_t(t) * kUsed + j) * w.m + o] -
                               ref.value) <= w.Tolerance(e, o, xr, mmq, ref));
          }
        }
      }
      std::ostringstream os;
      os << prefix << (per_slot ? "down_exps" : "gate_exps") << " ("
         << test::Name(w.type) << ") tokens=" << tokens
         << " mismatches=" << bad;
      check.Expect(bad == 0, os.str());
    }
  }
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_FLINT_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_FLINT_MODEL is not set\n";
    return test::kSkip;
  }
  try {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    if (reader == nullptr) {
      std::cerr << "flint.artifact_ops: " << error << '\n';
      return 1;
    }
    Checker check;
    std::mt19937 rng(2611);
    // The first tensor of every supported type, as dense rows.
    for (rocm::WeightType type : test::kAllTypes) {
      const gufo::core::GgufTensorInfo* found = nullptr;
      for (const auto& info : reader->GetTensors()) {
        if (static_cast<int>(info.type) == static_cast<int>(type) &&
            info.dimensions.size() >= 2 && info.dimensions[0] % 256 == 0) {
          found = &info;
          break;
        }
      }
      if (found == nullptr) {
        std::cout << "note: no " << test::Name(type) << " tensor\n";
        continue;
      }
      const auto rows = static_cast<int>(std::min<std::uint64_t>(
          found->dimensions[1] *
              (found->dimensions.size() > 2 ? found->dimensions[2] : 1),
          2048));
      CheckDense(*found, type, rows, rng, check);
    }
    // The vocabulary head's first rows.
    if (const auto* head = reader->FindTensor("output.weight")) {
      CheckDense(*head, rocm::WeightType::kQ8_0, 1024, rng, check);
    }
    // MoE layers with different expert types.
    for (int layer : {2, 20, 48}) {
      CheckRouted(*reader, layer, rng, check);
    }
    return check.Finish("flint.artifact_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.artifact_ops: " << e.what() << '\n';
    return 1;
  }
}
