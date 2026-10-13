// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"

#include "iree/base/internal/math.h"

// Each row covers one producer pipeline and pass count. Indexed bytes keep
// the full consumer matrix smaller than profile/use/pass records.
typedef struct loom_amdgpu_matrix_result_window_t {
  // VALU read, memory/export read, and nonmatrix overwrite separations.
  uint8_t ordinary[3];
  // Partial SrcC overlap separation, indexed by consumer pipeline.
  uint8_t accumulator[LOOM_AMDGPU_MATRIX_PIPELINE_COUNT_];
} loom_amdgpu_matrix_result_window_t;

// Pass columns are 2, 4, 8, and 16. Unused DGEMM two-pass cells remain zero.
// CDNA2 narrow matrix instructions share the SGEMM separation contract.
static const loom_amdgpu_matrix_result_window_t
    kResultWindows[LOOM_AMDGPU_MATRIX_WAIT_PROFILE_COUNT_ - 1]
                  [LOOM_AMDGPU_MATRIX_PIPELINE_COUNT_][4] =
                      {
                          [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A - 1] =
                              {
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_SGEMM] =
                                      {{{5, 5, 5}, {2, 2, 3}},
                                       {{7, 7, 7}, {4, 4, 5}},
                                       {{11, 11, 11}, {8, 8, 9}},
                                       {{19, 19, 19}, {16, 16, 17}}},
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_XDL] =
                                      {{{5, 5, 5}, {2, 2, 3}},
                                       {{7, 7, 7}, {4, 4, 5}},
                                       {{11, 11, 11}, {8, 8, 9}},
                                       {{19, 19, 19}, {16, 16, 17}}},
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM] =
                                      {{{0, 0, 0}, {0, 0, 0}},
                                       {{6, 9, 6}, {0, 0, 4}},
                                       {{11, 18, 11}, {0, 0, 9}},
                                       {{11, 18, 11}, {0, 0, 9}}},
                              },
                          [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX940 - 1] =
                              {
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_SGEMM] =
                                      {{{4, 4, 4}, {2, 0, 2}},
                                       {{6, 6, 6}, {4, 0, 4}},
                                       {{10, 10, 10}, {8, 0, 8}},
                                       {{18, 18, 18}, {16, 0, 16}}},
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_XDL] =
                                      {{{5, 5, 5}, {3, 3, 3}},
                                       {{7, 7, 7}, {5, 5, 5}},
                                       {{11, 11, 11}, {9, 9, 9}},
                                       {{19, 19, 19}, {17, 17, 17}}},
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM] =
                                      {{{0, 0, 0}, {0, 0, 0}},
                                       {{6, 9, 6}, {4, 0, 4}},
                                       {{11, 18, 11}, {9, 0, 9}},
                                       {{11, 18, 11}, {9, 0, 9}}},
                              },
                          [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950 - 1] =
                              {
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_SGEMM] =
                                      {{{4, 4, 4}, {2, 0, 2}},
                                       {{6, 6, 6}, {4, 0, 4}},
                                       {{10, 10, 10}, {8, 0, 8}},
                                       {{18, 18, 18}, {16, 0, 16}}},
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_XDL] =
                                      {{{5, 5, 5}, {3, 4, 3}},
                                       {{8, 8, 8}, {6, 6, 6}},
                                       {{12, 12, 12}, {10, 10, 10}},
                                       {{20, 20, 20}, {18, 18, 18}}},
                                  [LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM] =
                                      {{{0, 0, 0}, {0, 0, 0}},
                                       {{6, 9, 6}, {4, 0, 4}},
                                       {{19, 18, 11}, {17, 0, 17}},
                                       {{19, 18, 11}, {17, 0, 17}}},
                              },
};

static const uint8_t kWaitProfiles[] = {
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX940,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
    [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC] =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
};

static const iree_string_view_t kProfileNames[] = {
    IREE_SVL("unknown"),
    IREE_SVL("mfma_gfx90a"),
    IREE_SVL("mfma_gfx940"),
    IREE_SVL("mfma_gfx950"),
};

static const iree_string_view_t kResultUseNames[] = {
    IREE_SVL("unknown"),
    IREE_SVL("valu_read"),
    IREE_SVL("memory_read"),
    IREE_SVL("write"),
    IREE_SVL("matrix_src_ab"),
    IREE_SVL("matrix_srcc_exact"),
    IREE_SVL("matrix_srcc_overlap"),
};

iree_string_view_t loom_amdgpu_matrix_wait_profile_name(
    loom_amdgpu_matrix_wait_profile_t profile) {
  return kProfileNames[profile];
}

iree_string_view_t loom_amdgpu_matrix_wait_result_use_name(
    loom_amdgpu_matrix_wait_result_use_t use) {
  return kResultUseNames[use];
}

loom_amdgpu_matrix_wait_profile_t
loom_amdgpu_matrix_wait_profile_from_feature_profile(
    loom_amdgpu_matrix_feature_profile_t feature_profile) {
  return (loom_amdgpu_matrix_wait_profile_t)kWaitProfiles[feature_profile];
}

uint16_t loom_amdgpu_matrix_wait_result_cycle_count(
    loom_amdgpu_matrix_wait_profile_t profile,
    loom_amdgpu_matrix_instruction_t producer,
    loom_amdgpu_matrix_wait_result_use_t use,
    loom_amdgpu_matrix_instruction_t consumer) {
  const uint32_t pass_index =
      iree_math_count_trailing_zeros_u32(producer.pass_count) - 1;
  const loom_amdgpu_matrix_result_window_t* window =
      &kResultWindows[profile - 1][producer.pipeline][pass_index];
  if (use <= LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_WRITE) {
    return window->ordinary[use - 1];
  }
  if (use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB) {
    return window->ordinary[0];
  }
  const uint16_t overlap = window->accumulator[consumer.pipeline];
  if (use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP) {
    return overlap;
  }
  uint16_t exact = 0;
  if (producer.pipeline == LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM &&
      consumer.pipeline == LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM &&
      producer.pass_count == 4 && consumer.pass_count == 4) {
    exact = 4;
  } else if (profile != LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A &&
             producer.pass_count == 2) {
    exact = 2;
  }
  // Later pipelines forward a full accumulator only within one instruction
  // form, even when a different matrix instruction uses exactly the same
  // registers.
  if (profile != LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A &&
      producer.descriptor_ref != consumer.descriptor_ref && overlap > exact) {
    exact = overlap;
  }
  return exact;
}
