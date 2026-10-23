// Attention preparation (per-head RMSNorm, NeoX RoPE, FP16 KV write) and
// sink attention (split-K decode, causal/sliding prefill) against double
// oracles that read the same FP16 cache values.
//
// Tolerances.
// - q after norm/RoPE: the norm is FP32 (relative < 2e-6), the rotation two
//   rounded products and a sum: 1e-5 * (|x0| + |x1|) per output.
// - K cache: FP16 round-to-nearest of the FP32 value, so within one FP16 ulp
//   of the double oracle; V cache: exactly the host FP16 conversion.
// - Attention: a score is a 128-term FP32 dot product, so its error is at
//   most delta = 130 * 2^-24 * sum|q k| / sqrt(128); a softmax-weighted mean
//   then moves by at most 2 * delta * max|v|. Online-softmax sums are
//   sequential over at most 64 positions per lane plus fixed merge trees:
//   5e-5 * max|v| covers them.
// - Prefill feeds q and p to the matrix cores as FP16 (unit roundoff 2^-11,
//   the precision of the stored K and V): scores gain 2^-11 * sum|q k| /
//   sqrt(128) of error, and rounding each weight moves the mean by at most
//   2^-11 * max|v|.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

#include "tests/models/flint/ops_test_support.hpp"

namespace {

using namespace gufo::models::flint;
using rocm::kHeadDim;
using rocm::kKvHeads;
using rocm::kQHeads;
using test::Checker;
using test::DeviceBuffer;

constexpr int kGroup = kQHeads / kKvHeads;
constexpr double kScale = 0.088388347648318440550;
constexpr float kTheta = 10000.0f;
constexpr float kEps = 1e-6f;
constexpr float kInf = std::numeric_limits<float>::infinity();

/// Deterministic FP16 K/V content of one (position, head, dim), in [-1, 1].
std::uint16_t KvValue(std::uint64_t position, int head, int d, int which) {
  std::uint64_t h =
      position * 0x9E3779B97F4A7C15ull ^
      (std::uint64_t(head * kHeadDim + d) << 1 | which) * 0xBF58476D1CE4E5B9ull;
  h ^= h >> 31;
  h *= 0x94D049BB133111EBull;
  h ^= h >> 29;
  const float v = float(h >> 40) / float(1ull << 24) * 2.0f - 1.0f;
  return test::FloatToHalfBits(v);
}

/// One sequence's cache holding, in every slot, the latest position <=
/// `last` that maps to it.
struct Cache {
  int capacity;
  DeviceBuffer<std::uint16_t> k;
  DeviceBuffer<std::uint16_t> v;

  Cache(int cap, int last) : capacity(cap) {
    const std::size_t n = std::size_t(cap) * kKvHeads * kHeadDim;
    std::vector<std::uint16_t> hk(n);
    std::vector<std::uint16_t> hv(n);
    for (int slot = 0; slot < cap; ++slot) {
      if (slot > last) {
        continue;
      }
      const int p = last - ((last - slot) % cap + cap) % cap;
      for (int h = 0; h < kKvHeads; ++h) {
        for (int d = 0; d < kHeadDim; ++d) {
          const std::size_t i =
              (std::size_t(h) * cap + slot) * kHeadDim + d;  // head-major
          hk[i] = KvValue(p, h, d, 0);
          hv[i] = KvValue(p, h, d, 1);
        }
      }
    }
    k.Resize(n);
    v.Resize(n);
    k.Upload(hk);
    v.Upload(hv);
  }

  rocm::KvSequence View() const {
    return {reinterpret_cast<__half*>(k.get()),
            reinterpret_cast<__half*>(v.get()), capacity};
  }
};

/// K and V of one KV head over positions [lo, p], decoded once.
struct Span {
  int lo = 0;
  int count = 0;
  std::vector<float> k;
  std::vector<float> v;
  double max_v = 0.0;

  Span(int kvh, int lo_pos, int p) : lo(lo_pos), count(p - lo_pos + 1) {
    k.resize(std::size_t(count) * kHeadDim);
    v.resize(std::size_t(count) * kHeadDim);
    for (int j = 0; j < count; ++j) {
      for (int d = 0; d < kHeadDim; ++d) {
        const std::size_t i = std::size_t(j) * kHeadDim + d;
        k[i] = test::HalfBitsToFloat(KvValue(lo + j, kvh, d, 0));
        v[i] = test::HalfBitsToFloat(KvValue(lo + j, kvh, d, 1));
        max_v = std::max(max_v, std::fabs(double(v[i])));
      }
    }
  }
};

/// Double attention of one query head over a decoded span.
void Oracle(const float* q, const Span& span, float sink, bool fp16_inputs,
            double* out, double* tol) {
  std::vector<double> s(span.count);
  double max_s = sink;
  double max_dot_mag = 0.0;
  for (int j = 0; j < span.count; ++j) {
    const float* kr = &span.k[std::size_t(j) * kHeadDim];
    double dot = 0.0;
    double mag = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
      dot += double(q[d]) * kr[d];
      mag += std::fabs(double(q[d]) * kr[d]);
    }
    s[j] = dot * kScale;
    max_s = std::max(max_s, s[j]);
    max_dot_mag = std::max(max_dot_mag, mag);
  }
  double denom = std::exp(double(sink) - max_s);
  for (int d = 0; d < kHeadDim; ++d) {
    out[d] = 0.0;
  }
  for (int j = 0; j < span.count; ++j) {
    const double w = std::exp(s[j] - max_s);
    denom += w;
    const float* vr = &span.v[std::size_t(j) * kHeadDim];
    for (int d = 0; d < kHeadDim; ++d) {
      out[d] += w * vr[d];
    }
  }
  for (int d = 0; d < kHeadDim; ++d) {
    out[d] /= denom;
  }
  const double rounding =
      130.0 * std::ldexp(1.0, -24) + (fp16_inputs ? std::ldexp(1.0, -11) : 0.0);
  const double delta = rounding * max_dot_mag * kScale;
  *tol = span.max_v *
         (5e-5 + 2.0 * delta + (fp16_inputs ? std::ldexp(1.0, -11) : 0.0));
}

std::vector<float> Sinks(float base, std::mt19937& rng) {
  std::vector<float> s = test::RandomFloats(kQHeads, -1.0f, 1.0f, rng);
  if (base != 0.0f) {
    for (float& v : s) {
      v += base;
    }
  }
  return s;
}

int Lo(int p, int window) {
  return window > 0 ? std::max(0, p - window + 1) : 0;
}

int CompareRows(const std::vector<float>& got, const std::vector<float>& q,
                const std::vector<int>& positions,
                const std::vector<float>& sinks, int window, bool fp16_inputs,
                const std::string& label, Checker& check) {
  int bad = 0;
  double worst = 0.0;  // largest |error| / max|v| seen
  std::vector<double> ref(kHeadDim);
  std::optional<Span> span;
  for (std::size_t r = 0; r < positions.size(); ++r) {
    for (int h = 0; h < kQHeads; ++h) {
      if (h % kGroup == 0) {
        span.emplace(h / kGroup, Lo(positions[r], window), positions[r]);
      }
      const std::size_t base = (r * kQHeads + h) * kHeadDim;
      double tol = 0.0;
      Oracle(&q[base], *span, sinks[h], fp16_inputs, ref.data(), &tol);
      for (int d = 0; d < kHeadDim; ++d) {
        const float g = got[base + d];
        worst = std::max(worst, std::fabs(g - ref[d]) / span->max_v);
        if (!(std::isfinite(g) && std::fabs(g - ref[d]) <= tol) && bad++ < 3) {
          std::ostringstream os;
          os << label << " row=" << r << " pos=" << positions[r] << " h=" << h
             << " d=" << d << " got=" << g << " ref=" << ref[d]
             << " tol=" << tol;
          check.Expect(false, os.str());
        }
      }
    }
  }
  check.Expect(bad == 0, label + " mismatches=" + std::to_string(bad));
  std::cout << label << ": max |error| / max|v| = " << worst << '\n';
  return bad;
}

void RunDecode(const std::vector<const Cache*>& caches,
               const std::vector<int>& row_seq,
               const std::vector<int>& positions,
               const std::vector<float>& sinks, int window, std::mt19937& rng,
               const std::string& label, Checker& check) {
  const int rows = static_cast<int>(positions.size());
  const std::vector<float> q = test::RandomFloats(
      std::size_t(rows) * kQHeads * kHeadDim, -3.0f, 3.0f, rng);
  std::vector<rocm::KvSequence> seqs;
  for (const Cache* c : caches) {
    seqs.push_back(c->View());
  }
  DeviceBuffer<rocm::KvSequence> dseqs{std::span<const rocm::KvSequence>(seqs)};
  DeviceBuffer<std::int32_t> dpos{std::span<const std::int32_t>(
      std::vector<std::int32_t>(positions.begin(), positions.end()))};
  DeviceBuffer<std::int32_t> drs{std::span<const std::int32_t>(
      std::vector<std::int32_t>(row_seq.begin(), row_seq.end()))};
  DeviceBuffer<float> dq{std::span<const float>(q)};
  DeviceBuffer<float> ds{std::span<const float>(sinks)};
  DeviceBuffer<float> out(q.size());
  out.Fill(0xff);
  const std::size_t bytes = rocm::AttentionDecodeScratchBytes(rows);
  DeviceBuffer<std::uint8_t> scratch(bytes);
  test::CheckHip(rocm::AttentionDecode(dq.get(), dpos.get(), drs.get(),
                                       dseqs.get(), ds.get(), window, out.get(),
                                       rows, scratch.get(), bytes, nullptr),
                 "AttentionDecode");
  test::CheckHip(hipDeviceSynchronize(), "sync");
  const std::vector<float> plain = out.Download();
  CompareRows(plain, q, positions, sinks, window, false, label, check);
  // The Q8_1-emitting overload: same floats, and its rows quantize exactly
  // like the float rows do inside a projection.
  DeviceBuffer<float> out2(q.size());
  DeviceBuffer<std::uint8_t> q8(rocm::VectorActivationBytes(rows, 4096));
  test::CheckHip(
      rocm::AttentionDecode(dq.get(), dpos.get(), drs.get(), dseqs.get(),
                            ds.get(), window, out2.get(), q8.get(), rows,
                            scratch.get(), bytes, nullptr),
      "AttentionDecode(q8)");
  check.Expect(out2.Download() == plain, label + " q8 overload floats");
  if (rows <= rocm::kVectorRows) {
    std::vector<std::uint8_t> wbytes(
        rocm::WeightRowBytes(rocm::WeightType::kQ8_0, 4096) * 64);
    for (auto& v : wbytes) {
      v = static_cast<std::uint8_t>(rng());
    }
    for (std::size_t off = 0; off < wbytes.size(); off += 34) {
      wbytes[off] = 0x00;
      wbytes[off + 1] = 0x20;  // fp16 d = 2^-7
    }
    DeviceBuffer<std::uint8_t> dwt{std::span<const std::uint8_t>(wbytes)};
    const rocm::QuantMatrix o{dwt.get(), rocm::WeightType::kQ8_0, 64, 4096};
    DeviceBuffer<float> ya(std::size_t(rows) * 64);
    DeviceBuffer<float> yb(ya.size());
    const std::size_t pb = rocm::ProjectionScratchBytes(rows, 4096);
    DeviceBuffer<std::uint8_t> ps(pb);
    test::CheckHip(
        rocm::Project(o, out2.get(), ya.get(), rows, ps.get(), pb, nullptr),
        "Project");
    test::CheckHip(rocm::ProjectQuantized(o, q8.get(), yb.get(), rows, nullptr),
                   "ProjectQuantized");
    check.Expect(ya.Download() == yb.Download(),
                 label + " q8 rows differ from the projection's quantization");
  }
}

void CheckDecode(std::mt19937& rng, Checker& check) {
  // Full layers: chunk boundaries (64), 512/513 and split-K depths.
  {
    const Cache full(262144, 262143);
    for (int p : {0, 1, 63, 64, 65, 511, 512, 513, 4095, 40000, 262143}) {
      RunDecode({&full}, {0}, {p}, Sinks(0.0f, rng), 0, rng,
                "decode full p=" + std::to_string(p), check);
    }
    // Sinks: dominant, negligible, absent.
    RunDecode({&full}, {0}, {700}, Sinks(60.0f, rng), 0, rng, "decode sink=+60",
              check);
    RunDecode({&full}, {0}, {700}, Sinks(-1e4f, rng), 0, rng,
              "decode sink=-1e4", check);
    RunDecode({&full}, {0}, {700}, std::vector<float>(kQHeads, -kInf), 0, rng,
              "decode sink=-inf", check);
  }
  // Sliding layers in a 600-slot ring, before, at and past the window.
  for (int p : {0, 100, 511, 512, 513, 514, 599, 600, 5000}) {
    const Cache ring(600, p);
    RunDecode({&ring}, {0}, {p}, Sinks(0.0f, rng), rocm::kSlidingWindow, rng,
              "decode sliding p=" + std::to_string(p), check);
  }
  // Batched independent sequences, each with its own cache and position;
  // two rows share sequence 1.
  {
    const Cache a(4096, 3000);
    const Cache b(16384, 9000);
    const Cache ring(1027, 9000);
    const Cache c(8192, 7);
    RunDecode({&a, &b, &c}, {2, 1, 0, 1}, {7, 9000, 3000, 8990},
              Sinks(0.0f, rng), 0, rng, "decode batched full", check);
    RunDecode({&a, &ring, &c}, {1, 0, 2}, {9000, 2999, 6}, Sinks(0.0f, rng),
              rocm::kSlidingWindow, rng, "decode batched sliding", check);
  }
}

void RunPrefill(int capacity, int first, int rows, int window,
                std::mt19937& rng, const std::string& label, Checker& check) {
  const Cache cache(capacity, first + rows - 1);
  const std::vector<float> q = test::RandomFloats(
      std::size_t(rows) * kQHeads * kHeadDim, -3.0f, 3.0f, rng);
  const std::vector<float> sinks = Sinks(0.0f, rng);
  DeviceBuffer<float> dq{std::span<const float>(q)};
  DeviceBuffer<float> ds{std::span<const float>(sinks)};
  DeviceBuffer<float> out(q.size());
  out.Fill(0xff);
  test::CheckHip(rocm::AttentionPrefill(dq.get(), first, cache.View(), ds.get(),
                                        window, out.get(), rows, nullptr),
                 "AttentionPrefill");
  test::CheckHip(hipDeviceSynchronize(), "sync");
  std::vector<int> positions(rows);
  for (int r = 0; r < rows; ++r) {
    positions[r] = first + r;
  }
  CompareRows(out.Download(), q, positions, sinks, window, true, label, check);
}

void CheckPrefill(std::mt19937& rng, Checker& check) {
  for (int rows : {1, 7, 8, 9, 33, 515}) {
    RunPrefill(4096, 0, rows, 0, rng,
               "prefill full first=0 rows=" + std::to_string(rows), check);
  }
  RunPrefill(4096, 1000, 100, 0, rng, "prefill full first=1000", check);
  // Sliding: inside the window, crossing it, and rings sized exactly
  // SlidingRingCapacity(chunk) that wrap during the chunk.
  RunPrefill(rocm::SlidingRingCapacity(515), 0, 300, rocm::kSlidingWindow, rng,
             "prefill sliding short", check);
  RunPrefill(rocm::SlidingRingCapacity(515), 300, 515, rocm::kSlidingWindow,
             rng, "prefill sliding crossing", check);
  RunPrefill(rocm::SlidingRingCapacity(515), 2000, 515, rocm::kSlidingWindow,
             rng, "prefill sliding ring wrap", check);
  RunPrefill(rocm::SlidingRingCapacity(33), 600, 33, rocm::kSlidingWindow, rng,
             "prefill sliding tight ring", check);
}

/// Per-head norm and NeoX rotation of one head vector, in double, from the
/// documented FP32 angle.
void PrepareOracle(const float* x, const float* gamma, int position, bool rope,
                   double* out) {
  double ss = 0.0;
  for (int d = 0; d < kHeadDim; ++d) {
    ss += double(x[d]) * x[d];
  }
  const double scale = 1.0 / std::sqrt(ss / kHeadDim + kEps);
  for (int d = 0; d < kHeadDim; ++d) {
    out[d] = x[d] * scale * gamma[d];
  }
  if (!rope) {
    return;
  }
  for (int i = 0; i < kHeadDim / 2; ++i) {
    const float freq =
        static_cast<float>(std::pow(double(kTheta), -double(i) / 64.0));
    const float angle = static_cast<float>(position) * freq;
    const double c = std::cos(double(angle));
    const double s = std::sin(double(angle));
    const double x0 = out[i];
    const double x1 = out[i + kHeadDim / 2];
    out[i] = x0 * c - x1 * s;
    out[i + kHeadDim / 2] = x0 * s + x1 * c;
  }
}

void CheckPrepare(std::mt19937& rng, Checker& check) {
  for (bool rope : {true, false}) {
    for (int rows : {1, 9, 33}) {
      const std::vector<int> base_pos = {0, 5, 513, 100000, 262138, 77};
      std::vector<std::int32_t> positions(rows);
      std::vector<std::int32_t> row_seq(rows);
      for (int r = 0; r < rows; ++r) {
        positions[r] = base_pos[r % base_pos.size()] + r / 6;
        row_seq[r] = r % 2;
      }
      const int caps[2] = {262144, 600};
      const std::size_t qn = std::size_t(rows) * kQHeads * kHeadDim;
      const std::size_t kn = std::size_t(rows) * kKvHeads * kHeadDim;
      const std::vector<float> q = test::RandomFloats(qn, -4.0f, 4.0f, rng);
      const std::vector<float> k = test::RandomFloats(kn, -4.0f, 4.0f, rng);
      const std::vector<float> v = test::RandomFloats(kn, -4.0f, 4.0f, rng);
      const std::vector<float> qw =
          test::RandomFloats(kHeadDim, 0.5f, 2.0f, rng);
      const std::vector<float> kw =
          test::RandomFloats(kHeadDim, 0.5f, 2.0f, rng);
      DeviceBuffer<float> dq{std::span<const float>(q)};
      DeviceBuffer<float> dk{std::span<const float>(k)};
      DeviceBuffer<float> dv{std::span<const float>(v)};
      DeviceBuffer<float> dqw{std::span<const float>(qw)};
      DeviceBuffer<float> dkw{std::span<const float>(kw)};
      DeviceBuffer<std::int32_t> dpos{std::span<const std::int32_t>(positions)};
      DeviceBuffer<std::int32_t> drs{std::span<const std::int32_t>(row_seq)};
      DeviceBuffer<std::uint16_t> kc[2];
      DeviceBuffer<std::uint16_t> vc[2];
      std::vector<rocm::KvSequence> seqs;
      for (int s = 0; s < 2; ++s) {
        kc[s].Resize(std::size_t(caps[s]) * kKvHeads * kHeadDim);
        vc[s].Resize(std::size_t(caps[s]) * kKvHeads * kHeadDim);
        kc[s].Fill(0);
        vc[s].Fill(0);
        seqs.push_back({reinterpret_cast<__half*>(kc[s].get()),
                        reinterpret_cast<__half*>(vc[s].get()), caps[s]});
      }
      DeviceBuffer<rocm::KvSequence> dseqs{
          std::span<const rocm::KvSequence>(seqs)};
      test::CheckHip(
          rocm::PrepareAttention(dq.get(), dk.get(), dv.get(), dqw.get(),
                                 dkw.get(), dpos.get(), drs.get(), dseqs.get(),
                                 rows, rope, kTheta, kEps, nullptr),
          "PrepareAttention");
      test::CheckHip(hipDeviceSynchronize(), "sync");
      const std::vector<float> q_out = dq.Download();
      const std::vector<std::uint16_t> kh[2] = {kc[0].Download(),
                                                kc[1].Download()};
      const std::vector<std::uint16_t> vh[2] = {vc[0].Download(),
                                                vc[1].Download()};
      int q_bad = 0;
      int k_bad = 0;
      int v_bad = 0;
      double ref[kHeadDim];
      for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < kQHeads; ++h) {
          const std::size_t base = (std::size_t(r) * kQHeads + h) * kHeadDim;
          PrepareOracle(&q[base], qw.data(), positions[r], rope, ref);
          double normed[kHeadDim];
          PrepareOracle(&q[base], qw.data(), positions[r], false, normed);
          for (int d = 0; d < kHeadDim; ++d) {
            const int pair =
                d < kHeadDim / 2 ? d + kHeadDim / 2 : d - kHeadDim / 2;
            const double tol =
                1e-5 * (std::fabs(normed[d]) + std::fabs(normed[pair])) + 1e-30;
            q_bad += !(std::fabs(q_out[base + d] - ref[d]) <= tol);
          }
        }
        const int s = row_seq[r];
        const std::size_t slot = std::size_t(positions[r] % caps[s]);
        for (int h = 0; h < kKvHeads; ++h) {
          const std::size_t base = (std::size_t(r) * kKvHeads + h) * kHeadDim;
          PrepareOracle(&k[base], kw.data(), positions[r], rope, ref);
          double knormed[kHeadDim];
          PrepareOracle(&k[base], kw.data(), positions[r], false, knormed);
          for (int d = 0; d < kHeadDim; ++d) {
            const std::size_t i =
                (std::size_t(h) * caps[s] + slot) * kHeadDim + d;
            const double got = test::HalfBitsToFloat(kh[s][i]);
            const int pair =
                d < kHeadDim / 2 ? d + kHeadDim / 2 : d - kHeadDim / 2;
            const double ulp = std::ldexp(
                1.0, std::max(-24, std::ilogb(std::fabs(ref[d]) + 1e-30) - 10));
            const double fp32 =
                1e-5 * (std::fabs(knormed[d]) + std::fabs(knormed[pair]));
            k_bad += !(std::fabs(got - ref[d]) <= ulp + fp32);
            v_bad += vh[s][i] != test::FloatToHalfBits(v[base + d]);
          }
        }
      }
      std::ostringstream os;
      os << "prepare rope=" << rope << " rows=" << rows << " q_bad=" << q_bad
         << " k_bad=" << k_bad << " v_bad=" << v_bad;
      check.Expect(q_bad == 0 && k_bad == 0 && v_bad == 0, os.str());

      // The fused-projection overload over padded qkv rows must produce the
      // same q and cache bytes.
      constexpr int kStride = (kQHeads + 2 * kKvHeads) * kHeadDim + 64;
      std::vector<float> qkv(std::size_t(rows) * kStride, 1e30f);
      for (int r = 0; r < rows; ++r) {
        std::copy_n(&q[std::size_t(r) * kQHeads * kHeadDim], kQHeads * kHeadDim,
                    &qkv[std::size_t(r) * kStride]);
        std::copy_n(&k[std::size_t(r) * kKvHeads * kHeadDim],
                    kKvHeads * kHeadDim,
                    &qkv[std::size_t(r) * kStride + kQHeads * kHeadDim]);
        std::copy_n(
            &v[std::size_t(r) * kKvHeads * kHeadDim], kKvHeads * kHeadDim,
            &qkv[std::size_t(r) * kStride + (kQHeads + kKvHeads) * kHeadDim]);
      }
      DeviceBuffer<float> dqkv{std::span<const float>(qkv)};
      DeviceBuffer<float> fused_q(qn);
      for (int s = 0; s < 2; ++s) {
        kc[s].Fill(0);
        vc[s].Fill(0);
      }
      test::CheckHip(
          rocm::PrepareAttention(dqkv.get(), kStride, fused_q.get(), dqw.get(),
                                 dkw.get(), dpos.get(), drs.get(), dseqs.get(),
                                 rows, rope, kTheta, kEps, nullptr),
          "PrepareAttention(qkv)");
      test::CheckHip(hipDeviceSynchronize(), "sync");
      check.Expect(fused_q.Download() == q_out && kc[0].Download() == kh[0] &&
                       kc[1].Download() == kh[1] && vc[0].Download() == vh[0] &&
                       vc[1].Download() == vh[1],
                   "prepare fused qkv rows=" + std::to_string(rows) +
                       " differs from separate inputs");
    }
  }
}

}  // namespace

int main() {
  try {
    Checker check;
    std::mt19937 rng(513);
    CheckPrepare(rng, check);
    CheckDecode(rng, check);
    CheckPrefill(rng, check);
    return check.Finish("flint.attention_ops");
  } catch (const std::exception& e) {
    std::cerr << "flint.attention_ops: " << e.what() << '\n';
    return 1;
  }
}
