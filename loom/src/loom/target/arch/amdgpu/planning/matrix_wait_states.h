// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native MFMA result separation, in intervening instruction slots.
// Generated descriptor facts identify the pipeline and execution passes. The
// final packet planner adds physical overlap and consumer identity.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_WAIT_STATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_WAIT_STATES_H_

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_matrix_wait_profile_e {
  // Descriptor family does not use software MFMA result waits.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN = 0,
  // CDNA2 matrix pipeline and accumulator forwarding.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX90A = 1,
  // CDNA3 separates ordinary SGEMM and XDL execution.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX940 = 2,
  // CDNA4 extends XDL and DGEMM result windows.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950 = 3,
  // Number of matrix wait profiles.
  LOOM_AMDGPU_MATRIX_WAIT_PROFILE_COUNT_,
} loom_amdgpu_matrix_wait_profile_t;

typedef enum loom_amdgpu_matrix_pipeline_e {
  // Ordinary F32 matrix arithmetic.
  LOOM_AMDGPU_MATRIX_PIPELINE_SGEMM = 0,
  // Narrow, integer, XF32, and sparse matrix arithmetic.
  LOOM_AMDGPU_MATRIX_PIPELINE_XDL = 1,
  // F64 matrix arithmetic.
  LOOM_AMDGPU_MATRIX_PIPELINE_DGEMM = 2,
  // Number of matrix pipelines.
  LOOM_AMDGPU_MATRIX_PIPELINE_COUNT_,
} loom_amdgpu_matrix_pipeline_t;

typedef struct loom_amdgpu_matrix_instruction_t {
  // Typed descriptor identity, including input formats but not accumulator
  // bank.
  uint16_t descriptor_ref;
  // Hardware execution passes: 2, 4, 8, or 16.
  uint8_t pass_count;
  // One loom_amdgpu_matrix_pipeline_t value.
  uint8_t pipeline;
} loom_amdgpu_matrix_instruction_t;

typedef enum loom_amdgpu_matrix_wait_result_use_e {
  // No matrix result dependency.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN = 0,
  // An ordinary VALU instruction reads the result.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_VALU_READ = 1,
  // A VMEM, DS, or export instruction reads the result.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MEMORY_READ = 2,
  // A nonmatrix instruction overwrites the result.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_WRITE = 3,
  // A matrix instruction reads the result as SrcA, SrcB, index, or scale.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB = 4,
  // SrcC covers exactly the producer's physical result range.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT = 5,
  // SrcC partially overlaps the producer's physical result range.
  LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP = 6,
} loom_amdgpu_matrix_wait_result_use_t;

// Returns the stable spelling for a matrix wait profile.
iree_string_view_t loom_amdgpu_matrix_wait_profile_name(
    loom_amdgpu_matrix_wait_profile_t profile);

// Returns the stable spelling for a matrix result use.
iree_string_view_t loom_amdgpu_matrix_wait_result_use_name(
    loom_amdgpu_matrix_wait_result_use_t use);

// Selects the native wait contract from verified processor features.
loom_amdgpu_matrix_wait_profile_t
loom_amdgpu_matrix_wait_profile_from_feature_profile(
    loom_amdgpu_matrix_feature_profile_t feature_profile);

// Returns the required separation for verified instruction facts and a
// concrete use. The consumer identity is used only for matrix SrcC reads.
uint16_t loom_amdgpu_matrix_wait_result_cycle_count(
    loom_amdgpu_matrix_wait_profile_t profile,
    loom_amdgpu_matrix_instruction_t producer,
    loom_amdgpu_matrix_wait_result_use_t use,
    loom_amdgpu_matrix_instruction_t consumer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_WAIT_STATES_H_
