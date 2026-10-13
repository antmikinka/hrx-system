// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/testbench.h"

#include <string.h>

#include "loom/ops/check/ops.h"

enum {
  IREE_BENCHMARK_LOOM_MAX_REQUIREMENT_PROVIDERS = 8,
};

iree_status_t iree_benchmark_loom_evaluate_requirements(
    const iree_benchmark_loom_configuration_t* configuration,
    loom_run_hal_testbench_context_t* hal_context,
    const loom_testbench_module_plan_t* module_plan,
    iree_arena_allocator_t* arena, iree_benchmark_loom_work_plan_t* work_plan) {
  loom_testbench_requirement_provider_t
      requirement_providers[IREE_BENCHMARK_LOOM_MAX_REQUIREMENT_PROVIDERS] = {
          0};
  iree_host_size_t requirement_provider_count = 0;
  if (configuration->populate_requirement_providers.fn != NULL) {
    IREE_RETURN_IF_ERROR(configuration->populate_requirement_providers.fn(
        configuration->populate_requirement_providers.user_data, hal_context,
        IREE_ARRAYSIZE(requirement_providers), requirement_providers,
        &requirement_provider_count));
  }
  if (requirement_provider_count > IREE_ARRAYSIZE(requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "iree-benchmark-loom requirement provider capacity exceeded");
  }
  loom_testbench_requirement_provider_registry_t requirement_registry = {0};
  loom_testbench_requirement_provider_registry_initialize(
      requirement_providers, requirement_provider_count, &requirement_registry);
  enum { UNCHECKED = 0, RUNNABLE = 1, SKIPPED = 2 };
  const iree_host_size_t record_count =
      module_plan->case_count + module_plan->scenario_count;
  uint8_t* record_states = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, record_count, sizeof(*record_states), (void**)&record_states));
  memset(record_states, UNCHECKED, record_count * sizeof(*record_states));
  for (iree_host_size_t i = 0; i < work_plan->selected_benchmark_count; ++i) {
    iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[i];
    const iree_host_size_t record_index =
        selection->case_plan != NULL
            ? selection->case_plan - module_plan->cases
            : module_plan->case_count +
                  (selection->scenario_plan - module_plan->scenarios);
    if (record_states[record_index] == UNCHECKED) {
      const loom_region_t* body =
          selection->case_plan != NULL
              ? loom_check_case_body(selection->case_plan->op)
              : loom_check_scenario_body(selection->scenario_plan->op);
      loom_testbench_requirement_result_t result = {0};
      IREE_RETURN_IF_ERROR(loom_testbench_evaluate_requirements(
          module_plan->module, body, &requirement_registry, &result));
      record_states[record_index] = result.skipped ? SKIPPED : RUNNABLE;
    }
    selection->skipped = record_states[record_index] == SKIPPED;
  }
  return iree_ok_status();
}

void iree_benchmark_loom_configure_reference_oracles(
    loom_run_hal_testbench_context_t* context, iree_allocator_t host_allocator,
    iree_benchmark_loom_reference_oracles_t* out_oracles,
    loom_testbench_case_execution_options_t* inout_execution_options) {
  *out_oracles = (iree_benchmark_loom_reference_oracles_t){0};
  out_oracles->options = (loom_testbench_reference_matmul_oracle_options_t){
      .result_pool = context->runtime.staging_pool,
      .host_allocator = host_allocator,
  };
  loom_testbench_reference_matmul_oracle_provider_initialize(
      &out_oracles->options, &out_oracles->providers[0]);
  loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
      &out_oracles->options, &out_oracles->providers[1]);
  inout_execution_options->invocation.oracle_providers =
      loom_make_testbench_oracle_provider_list(
          out_oracles->providers,
          IREE_BENCHMARK_LOOM_REFERENCE_ORACLE_PROVIDER_COUNT);
}
