{
  lib,
  llama-cpp,
  fetchFromGitHub,
  rocmPackages,
  runCommand,
}:
let
  # b11069 includes Qwen3.8 and DFlash2. The separate MTP pin is the
  # Flash-Next implementation from ggml-org/llama.cpp#28243. The Flint pin is
  # upstream master after Flint support (a37cc461, ggml-org/llama.cpp#30928).
  revision = "68d9053afd4f4d0752ced6187585f862355a40be";
  mtpRevision = "6fcaa16f4b360649933a54d1f91ad40ed35c0e11";
  flintRevision = "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc";
  release =
    (llama-cpp.override {
      rocmSupport = true;
      rocmGpuTargets = [ "gfx1151" ];
    }).overrideAttrs
      (oldAttrs: {
        version = "11069";
        src = fetchFromGitHub {
          owner = "ggml-org";
          repo = "llama.cpp";
          rev = revision;
          hash = "sha256-BnGWYIkVe9y4aufhS5s3Jco1j/BY0MgBlzXrRQFMW3o=";
        };
        npmDepsHash = "sha256-2Q7XhaLAArmviOLdQsNbYTfdyDE5pW9lR26cRHEVl9k=";
        cmakeFlags = (oldAttrs.cmakeFlags or [ ]) ++ [
          "-DLLAMA_HIP_UMA=ON"
          "-DLLAMA_BUILD_COMMIT:STRING=${builtins.substring 0 8 revision}"
        ];
        preConfigure = (oldAttrs.preConfigure or "") + ''
          cmakeFlagsArray+=("-DCMAKE_HIP_FLAGS=--rocm-path=${rocmPackages.clr} -mllvm --amdgpu-unroll-threshold-local=600")
        '';
      });
  mtpRuntime = release.overrideAttrs (oldAttrs: {
    pname = "llama-cpp-mtp";
    version = "11069";
    src = fetchFromGitHub {
      owner = "danielhanchen";
      repo = "llama.cpp";
      rev = mtpRevision;
      hash = "sha256-YgIkYHiV1LNA1OvTcB8SSdOeqr37+cEFAOQjEBfXcK4=";
    };
    cmakeFlags =
      builtins.filter (f: !(lib.hasPrefix "-DLLAMA_BUILD_COMMIT" f)) (oldAttrs.cmakeFlags or [ ])
      ++ [ "-DLLAMA_BUILD_COMMIT:STRING=${builtins.substring 0 8 mtpRevision}" ];
  });
  flintRuntime = release.overrideAttrs (oldAttrs: {
    pname = "llama-cpp-flint";
    version = "11827";
    src = fetchFromGitHub {
      owner = "ggml-org";
      repo = "llama.cpp";
      rev = flintRevision;
      hash = "sha256-N8BvVmwbL7IRuG1VjXIs9gXbBvAKLhw24abDMv68/ok=";
    };
    npmDepsHash = "sha256-hqYzdp1FwPhYkBP278fQXgY9NCKB6Q3P6Gko7BfP0m0=";
    cmakeFlags =
      builtins.filter (f: !(lib.hasPrefix "-DLLAMA_BUILD_COMMIT" f)) (oldAttrs.cmakeFlags or [ ])
      ++ [ "-DLLAMA_BUILD_COMMIT:STRING=${builtins.substring 0 8 flintRevision}" ];
  });
in
{
  inherit release;
  # Distinct executable names allow both reference revisions in one shell.
  mtp = runCommand "llama-cpp-mtp-bin" { } ''
    mkdir -p $out/bin
    ln -s ${mtpRuntime}/bin/llama-server $out/bin/llama-server-mtp
    ln -s ${mtpRuntime}/bin/llama-bench $out/bin/llama-bench-mtp
  '';
  flint = runCommand "llama-cpp-flint-bin" { } ''
    mkdir -p $out/bin
    ln -s ${flintRuntime}/bin/llama-server $out/bin/llama-server-flint
    ln -s ${flintRuntime}/bin/llama-bench $out/bin/llama-bench-flint
  '';
}
