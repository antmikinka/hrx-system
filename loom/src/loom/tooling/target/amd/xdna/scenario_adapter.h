// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// XDNA finite-pipeline adaptation for ordinary scenario functions.

#ifndef LOOM_TOOLING_TARGET_AMD_XDNA_SCENARIO_ADAPTER_H_
#define LOOM_TOOLING_TARGET_AMD_XDNA_SCENARIO_ADAPTER_H_

#include "loom/tooling/execution/hal/testbench_actual.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds one finite AIE2P array program around a target-neutral function
// scenario subject selected for an XDNA HAL device.
iree_status_t loom_xdna_hal_testbench_adapt_scenario_function(
    const loom_testbench_compilation_t* compilation,
    const loomc_module_t* source_module,
    const loom_testbench_invocation_plan_t* source_invocation,
    const loom_target_profile_t* target_profile,
    iree_allocator_t host_allocator,
    loom_run_hal_target_function_adaptation_t* out_adaptation,
    loomc_module_t** out_module,
    loom_testbench_invocation_plan_t* out_invocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_AMD_XDNA_SCENARIO_ADAPTER_H_
