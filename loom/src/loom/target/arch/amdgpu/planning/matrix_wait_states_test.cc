// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"

#include "iree/testing/gtest.h"

namespace {

constexpr auto kGfx90a = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A;
constexpr auto kGfx940 = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX940;
constexpr auto kGfx950 = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950;
constexpr auto kSgemm = LOOM_AMDGPU_MATRIX_PIPELINE_SGEMM;
constexpr auto kXdl = LOOM_AMDGPU_MATRIX_PIPELINE_XDL;
constexpr auto kDgemm = LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM;
constexpr auto kValu = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_VALU_READ;
constexpr auto kMemory = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MEMORY_READ;
constexpr auto kWrite = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_WRITE;
constexpr auto kSrcAB = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB;
constexpr auto kExact = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT;
constexpr auto kOverlap =
    LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP;

loom_amdgpu_matrix_instruction_t Instruction(
    loom_amdgpu_matrix_pipeline_t pipeline, uint8_t passes,
    uint16_t descriptor_ref = 1) {
  return {descriptor_ref, passes, static_cast<uint8_t>(pipeline)};
}

TEST(AmdgpuMatrixWaitStatesTest, ProcessorFeaturesSelectWaitContract) {
  EXPECT_EQ(loom_amdgpu_matrix_wait_profile_from_feature_profile(
                LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A),
            kGfx90a);
  EXPECT_EQ(loom_amdgpu_matrix_wait_profile_from_feature_profile(
                LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940),
            kGfx940);
  EXPECT_EQ(loom_amdgpu_matrix_wait_profile_from_feature_profile(
                LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950),
            kGfx950);
  EXPECT_EQ(loom_amdgpu_matrix_wait_profile_from_feature_profile(
                LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC),
            kGfx950);
}

TEST(AmdgpuMatrixWaitStatesTest, DgemmSeparatesArithmeticMemoryAndOverwrite) {
  // DGEMM result access has three different forwarding windows. On CDNA4,
  // arithmetic reads wait longer than memory reads; writes retain the older
  // short window. SrcA/SrcB uses the arithmetic read window.
  for (auto profile : {kGfx90a, kGfx940, kGfx950}) {
    for (uint8_t passes : {uint8_t{4}, uint8_t(profile == kGfx950 ? 16 : 8)}) {
      const auto producer = Instruction(kDgemm, passes);
      const uint16_t arithmetic = passes == 4          ? 6
                                  : profile == kGfx950 ? 19
                                                       : 11;
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                           kValu, {}),
                arithmetic);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                    profile, producer, kSrcAB, Instruction(kXdl, 8)),
                arithmetic);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                           kMemory, {}),
                passes == 4 ? 9 : 18);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                           kWrite, {}),
                passes == 4 ? 6 : 11);
    }
  }
}

TEST(AmdgpuMatrixWaitStatesTest,
     SgemmAndXdlHaveDistinctLaterGenerationWindows) {
  for (uint8_t passes : {2, 8, 16}) {
    for (auto use : {kValu, kMemory, kWrite, kSrcAB}) {
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                    kGfx90a, Instruction(kSgemm, passes), use, {}),
                passes + 3);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                    kGfx90a, Instruction(kXdl, passes), use, {}),
                passes + 3);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                    kGfx940, Instruction(kSgemm, passes), use, {}),
                passes + 2);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                    kGfx950, Instruction(kSgemm, passes), use, {}),
                passes + 2);
    }
  }
  const uint16_t gfx940[] = {5, 7, 11, 19};
  const uint16_t gfx950[] = {5, 8, 12, 20};
  for (uint8_t index = 0; index < 4; ++index) {
    const auto producer = Instruction(kXdl, 2 << index);
    for (auto use : {kValu, kMemory, kWrite, kSrcAB}) {
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(kGfx940, producer,
                                                           use, {}),
                gfx940[index]);
      EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(kGfx950, producer,
                                                           use, {}),
                gfx950[index]);
    }
  }
}

TEST(AmdgpuMatrixWaitStatesTest,
     FullAccumulatorForwardingRequiresInstructionIdentity) {
  for (auto profile : {kGfx90a, kGfx940, kGfx950}) {
    auto producer = Instruction(kXdl, 8);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                         kExact, producer),
              0);
    auto consumer = Instruction(kXdl, 8, 2);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                         kExact, consumer),
              profile == kGfx90a   ? 0
              : profile == kGfx940 ? 9
                                   : 10);
  }
  for (auto profile : {kGfx90a, kGfx940, kGfx950}) {
    auto producer = Instruction(kDgemm, 4);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                         kExact, producer),
              4);
    producer = Instruction(kXdl, 2);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(profile, producer,
                                                         kExact, producer),
              profile == kGfx90a ? 0 : 2);
  }
}

TEST(AmdgpuMatrixWaitStatesTest, OverlapDependsOnBothPipelines) {
  for (uint8_t passes : {2, 8, 16}) {
    const auto producer = Instruction(kXdl, passes);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                  kGfx90a, producer, kOverlap, Instruction(kSgemm, 8)),
              passes);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                  kGfx90a, producer, kOverlap, Instruction(kDgemm, 8)),
              passes + 1);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                  kGfx940, producer, kOverlap, Instruction(kXdl, 8)),
              passes + 1);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                  kGfx950, producer, kOverlap, Instruction(kXdl, 8)),
              passes + 2);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_cycle_count(
                  kGfx950, producer, kOverlap, Instruction(kSgemm, 8)),
              passes == 2 ? 3 : passes + 2);
  }
  for (auto profile : {kGfx90a, kGfx940, kGfx950}) {
    EXPECT_EQ(
        loom_amdgpu_matrix_wait_result_cycle_count(
            profile, Instruction(kDgemm, 8), kOverlap, Instruction(kXdl, 8)),
        0);
  }
  for (auto profile : {kGfx940, kGfx950}) {
    EXPECT_EQ(
        loom_amdgpu_matrix_wait_result_cycle_count(
            profile, Instruction(kSgemm, 8), kOverlap, Instruction(kXdl, 8)),
        0);
    EXPECT_EQ(
        loom_amdgpu_matrix_wait_result_cycle_count(
            profile, Instruction(kDgemm, 8), kOverlap, Instruction(kSgemm, 8)),
        profile == kGfx940 ? 9 : 17);
  }
}

}  // namespace
