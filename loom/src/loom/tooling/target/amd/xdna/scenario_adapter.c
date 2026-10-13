// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amd/xdna/scenario_adapter.h"

#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cache.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/target/arch/amd/xdna/device/profile.h"
#include "loom/target/profile.h"
#include "loom/target/resolved_target.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

typedef struct loom_xdna_scenario_pipeline_t {
  // Builder whose insertion point moves between the pipeline and its strands.
  loom_builder_t* builder;
  // Original target-neutral function called by the compute strand.
  loom_symbol_ref_t function_ref;
  // Source location retained from the original function.
  loom_location_id_t location;
  // External HAL binding refined to the subject's element alignment.
  loom_value_id_t external_buffer;
  // Compute-tile allocation carrying the staged input record.
  loom_value_id_t input_buffer;
  // Compute-tile allocation carrying the staged output record.
  loom_value_id_t output_buffer;
  // Input channel shared by the DMA and compute strands.
  loom_value_id_t input_channel;
  // Output channel shared by the compute and DMA strands.
  loom_value_id_t output_channel;
  // Zero byte offset used by all whole-record views and copies.
  loom_value_id_t byte_origin;
  // Exact byte length of one staged record.
  loom_value_id_t byte_length;
  // Typed view of one complete record.
  loom_type_t record_view_type;
  // Immutable channel read obligation for one record.
  loom_type_t read_type;
  // Channel write obligation for one record.
  loom_type_t write_type;
  // Asynchronous transfer token type.
  loom_type_t async_token_type;
  // Committed asynchronous transfer group type.
  loom_type_t async_group_type;
} loom_xdna_scenario_pipeline_t;

static iree_status_t loom_xdna_scenario_add_symbol(loom_module_t* module,
                                                   iree_string_view_t stem,
                                                   loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  for (iree_host_size_t ordinal = module->symbols.count;; ++ordinal) {
    char name_buffer[96];
    const int name_length =
        iree_snprintf(name_buffer, sizeof(name_buffer), "%.*s_%zu",
                      (int)stem.size, stem.data, ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "XDNA scenario adapter symbol is too long");
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module,
        iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
        &name_id));
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
    return iree_ok_status();
  }
}

static iree_status_t loom_xdna_scenario_build_constant(
    loom_builder_t* builder, int64_t value, loom_type_t type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, loom_attr_i64(value),
                                                 type, location, &op));
  *out_value = loom_index_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_xdna_scenario_make_opaque_type(
    loom_builder_t* builder, iree_string_view_t name, loom_type_t* out_type) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_type = loom_type_dialect_opaque(name_id);
  return loom_module_intern_type(builder->module, *out_type, out_type);
}

static iree_status_t loom_xdna_scenario_build_async_transfer(
    loom_xdna_scenario_pipeline_t* pipeline, loom_value_id_t source,
    loom_value_id_t target, loom_kernel_direction_t direction) {
  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_async_copy_build(
      pipeline->builder, source, target, LOOM_CACHE_SCOPE_WORKGROUP,
      LOOM_CACHE_TEMPORAL_REGULAR, direction, pipeline->async_token_type,
      pipeline->location, &copy_op));
  const loom_value_id_t token = loom_kernel_async_copy_token(copy_op);
  loom_op_t* group_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_async_group_build(
      pipeline->builder, &token, 1, pipeline->async_group_type,
      pipeline->location, &group_op));
  loom_op_t* wait_op = NULL;
  return loom_kernel_async_wait_build(
      pipeline->builder, loom_kernel_async_group_group(group_op),
      /*newer_groups=*/0, pipeline->location, &wait_op);
}

static iree_status_t loom_xdna_scenario_build_dma_strand(
    loom_xdna_scenario_pipeline_t* pipeline, uint16_t row) {
  const int64_t origins[] = {0, row};
  const int64_t counts[] = {1, 1};
  const int64_t strides[] = {1, 1};
  loom_op_t* strand_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_strand_build(
      pipeline->builder, /*build_flags=*/0, loom_symbol_ref_null(),
      /*origins=*/NULL, /*origins_count=*/0, origins, IREE_ARRAYSIZE(origins),
      /*counts=*/NULL, /*counts_count=*/0, counts, IREE_ARRAYSIZE(counts),
      /*strides=*/NULL, /*strides_count=*/0, strides, IREE_ARRAYSIZE(strides),
      pipeline->location, &strand_op));
  const loom_builder_ip_t strand_ip = loom_builder_enter_region(
      pipeline->builder, strand_op, loom_pipeline_strand_body(strand_op));

  loom_op_t* external_view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      pipeline->builder, pipeline->external_buffer, pipeline->byte_origin,
      pipeline->record_view_type, pipeline->location, &external_view_op));
  const loom_value_id_t external_view =
      loom_buffer_view_result(external_view_op);

  loom_op_t* input_reserve_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_reserve_build(
      pipeline->builder, pipeline->input_channel, pipeline->write_type,
      pipeline->record_view_type, pipeline->location, &input_reserve_op));
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_build_async_transfer(
      pipeline, external_view, loom_channel_reserve_view(input_reserve_op),
      LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP));
  loom_op_t* publish_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_publish_build(
      pipeline->builder, loom_channel_reserve_write(input_reserve_op),
      pipeline->location, &publish_op));

  loom_op_t* output_acquire_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_acquire_build(
      pipeline->builder, /*build_flags=*/0, /*mode=*/0,
      pipeline->output_channel, pipeline->read_type, pipeline->record_view_type,
      pipeline->location, &output_acquire_op));
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_build_async_transfer(
      pipeline, loom_channel_acquire_view(output_acquire_op), external_view,
      LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL));
  loom_op_t* release_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_release_build(
      pipeline->builder, loom_channel_acquire_read(output_acquire_op),
      pipeline->location, &release_op));
  loom_op_t* end_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_end_build(pipeline->builder, pipeline->location, &end_op));
  loom_builder_restore(pipeline->builder, strand_ip);
  return iree_ok_status();
}

static iree_status_t loom_xdna_scenario_build_compute_strand(
    loom_xdna_scenario_pipeline_t* pipeline, uint16_t row) {
  const int64_t origins[] = {0, row};
  const int64_t counts[] = {1, 1};
  const int64_t strides[] = {1, 1};
  loom_op_t* strand_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_strand_build(
      pipeline->builder, /*build_flags=*/0, loom_symbol_ref_null(),
      /*origins=*/NULL, /*origins_count=*/0, origins, IREE_ARRAYSIZE(origins),
      /*counts=*/NULL, /*counts_count=*/0, counts, IREE_ARRAYSIZE(counts),
      /*strides=*/NULL, /*strides_count=*/0, strides, IREE_ARRAYSIZE(strides),
      pipeline->location, &strand_op));
  const loom_builder_ip_t strand_ip = loom_builder_enter_region(
      pipeline->builder, strand_op, loom_pipeline_strand_body(strand_op));

  loom_op_t* input_acquire_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_acquire_build(
      pipeline->builder, /*build_flags=*/0, /*mode=*/0, pipeline->input_channel,
      pipeline->read_type, pipeline->record_view_type, pipeline->location,
      &input_acquire_op));
  loom_op_t* output_reserve_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_reserve_build(
      pipeline->builder, pipeline->output_channel, pipeline->write_type,
      pipeline->record_view_type, pipeline->location, &output_reserve_op));

  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_copy_build(
      pipeline->builder, pipeline->input_buffer, pipeline->byte_origin,
      pipeline->output_buffer, pipeline->byte_origin, pipeline->byte_length,
      pipeline->location, &copy_op));
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_call_build(
      pipeline->builder, LOOM_FUNC_CALL_BUILD_FLAG_HAS_INLINE_POLICY,
      /*purity=*/0, /*temperature=*/0, LOOM_INLINE_POLICY_INLINE,
      pipeline->function_ref, &pipeline->output_buffer, 1,
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, pipeline->location, &call_op));
  loom_op_t* publish_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_publish_build(
      pipeline->builder, loom_channel_reserve_write(output_reserve_op),
      pipeline->location, &publish_op));
  loom_op_t* release_op = NULL;
  IREE_RETURN_IF_ERROR(loom_channel_release_build(
      pipeline->builder, loom_channel_acquire_read(input_acquire_op),
      pipeline->location, &release_op));
  loom_op_t* end_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_end_build(pipeline->builder, pipeline->location, &end_op));
  loom_builder_restore(pipeline->builder, strand_ip);
  return iree_ok_status();
}

static iree_status_t loom_xdna_scenario_build_pipeline(
    loom_module_t* module, loom_symbol_ref_t function_ref,
    loom_op_t* function_op, loom_type_t input_type, uint64_t element_count,
    int64_t byte_length, int64_t byte_alignment,
    const loom_aie2p_target_profile_t* target_profile,
    loom_testbench_invocation_plan_t* out_invocation) {
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_add_symbol(
      module, IREE_SV("__loom_xdna_scenario_target"), &target_ref));
  loom_target_facts_t* target_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_profile_project_facts(
      &target_profile->base, &module->arena, &target_facts));
  const loom_resolved_target_t resolved_target = {
      .provider = &loom_aie2p_target_provider,
      .facts = target_facts,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_target_provider.materialize_definition(
      &builder, &resolved_target, target_ref, function_op->location));

  loom_symbol_ref_t pipeline_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_add_symbol(
      module, IREE_SV("__loom_xdna_scenario_adapter"), &pipeline_ref));
  const loom_type_t binding_type = loom_type_buffer();
  loom_op_t* pipeline_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_def_build(
      &builder,
      LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_SCOPE |
          LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_VISIBILITY |
          LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_RETAIN |
          LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_TARGET,
      LOOM_PIPELINE_DEF_SCOPE_KERNEL, LOOM_PIPELINE_DEF_VISIBILITY_PUBLIC,
      LOOM_PIPELINE_DEF_RETAIN_RETAIN, target_ref, pipeline_ref,
      /*specializations_types=*/NULL, /*specializations_types_count=*/0,
      &binding_type, /*bindings_types_count=*/1, /*predicates=*/NULL,
      /*predicates_count=*/0, function_op->location, &pipeline_op));

  loom_region_t* body = loom_pipeline_def_body(pipeline_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  IREE_ASSERT(entry_block->arg_count == 1);
  const loom_value_id_t external_binding = entry_block->arg_ids[0];
  loom_builder_enter_region(&builder, pipeline_op, body);

  loom_op_t* alignment_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_assume_alignment_build(
      &builder, &external_binding, 1, byte_alignment, &binding_type, 1,
      function_op->location, &alignment_op));
  const loom_value_id_t external_buffer = loom_op_results(alignment_op)[0];

  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(target_profile->device_profile);
  const loom_xdna_tile_facts_t* compute_facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  const uint16_t dma_row = compute_facts->first_row;
  const uint16_t compute_row = dma_row + 1;
  const int64_t memory_coordinates[] = {0, compute_row};
  loom_op_t* memory_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_memory_build(
      &builder, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      /*coordinates=*/NULL, /*coordinates_count=*/0, memory_coordinates,
      IREE_ARRAYSIZE(memory_coordinates), loom_type_pool(),
      function_op->location, &memory_op));
  const loom_value_id_t memory = loom_pipeline_memory_result(memory_op);

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(module, index_type, &index_type));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(module, offset_type, &offset_type));
  loom_value_id_t byte_origin = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_build_constant(
      &builder, 0, offset_type, function_op->location, &byte_origin));
  loom_value_id_t byte_length_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_build_constant(
      &builder, byte_length, offset_type, function_op->location,
      &byte_length_value));
  loom_value_id_t capacity = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_build_constant(
      &builder, 1, index_type, function_op->location, &capacity));

  loom_type_t record_view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, loom_type_element_type(input_type),
                          loom_type_dim(input_type, 0), /*encoding_id=*/0);
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(module, record_view_type, &record_view_type));
  loom_type_t slot_view_type = loom_type_shaped_2d(
      LOOM_TYPE_VIEW, loom_type_element_type(input_type),
      loom_dim_pack_static(1), loom_dim_pack_static((int64_t)element_count),
      /*encoding_id=*/0);
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(module, slot_view_type, &slot_view_type));
  loom_type_t payload_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, loom_type_element_type(input_type),
                          loom_type_dim(input_type, 0), /*encoding_id=*/0);
  loom_type_id_t payload_type_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type_id(module, payload_type, &payload_type_id));
  loom_type_t channel_type = loom_type_none();
  loom_type_t read_type = loom_type_none();
  loom_type_t write_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_channel_type_make(module, payload_type_id, &channel_type));
  IREE_RETURN_IF_ERROR(loom_read_type_make(
      module, /*build_flags=*/0, payload_type_id, /*mode=*/0, &read_type));
  IREE_RETURN_IF_ERROR(
      loom_write_type_make(module, payload_type_id, &write_type));

  loom_value_id_t local_buffers[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  loom_value_id_t channels[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(local_buffers); ++i) {
    loom_op_t* allocation_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
        &builder, LOOM_BUFFER_ALLOCA_BUILD_FLAG_HAS_POOL,
        LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, memory, byte_alignment,
        byte_length_value, loom_type_buffer(), function_op->location,
        &allocation_op));
    local_buffers[i] = loom_buffer_alloca_result(allocation_op);
    loom_op_t* slots_op = NULL;
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        &builder, local_buffers[i], byte_origin, slot_view_type,
        function_op->location, &slots_op));
    loom_op_t* channel_op = NULL;
    IREE_RETURN_IF_ERROR(loom_channel_bind_build(
        &builder, LOOM_CHANNEL_BIND_DISCIPLINE_FIFO,
        loom_buffer_view_result(slots_op), capacity, channel_type,
        function_op->location, &channel_op));
    channels[i] = loom_channel_bind_result(channel_op);
  }

  loom_type_t async_token_type = loom_type_none();
  loom_type_t async_group_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_make_opaque_type(
      &builder, IREE_SV("kernel.async.token"), &async_token_type));
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_make_opaque_type(
      &builder, IREE_SV("kernel.async.group"), &async_group_type));
  loom_xdna_scenario_pipeline_t pipeline = {
      .builder = &builder,
      .function_ref = function_ref,
      .location = function_op->location,
      .external_buffer = external_buffer,
      .input_buffer = local_buffers[0],
      .output_buffer = local_buffers[1],
      .input_channel = channels[0],
      .output_channel = channels[1],
      .byte_origin = byte_origin,
      .byte_length = byte_length_value,
      .record_view_type = record_view_type,
      .read_type = read_type,
      .write_type = write_type,
      .async_token_type = async_token_type,
      .async_group_type = async_group_type,
  };
  IREE_RETURN_IF_ERROR(loom_xdna_scenario_build_dma_strand(&pipeline, dma_row));
  IREE_RETURN_IF_ERROR(
      loom_xdna_scenario_build_compute_strand(&pipeline, compute_row));
  loom_op_t* finish_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_finish_build(&builder, function_op->location, &finish_op));

  *out_invocation = (loom_testbench_invocation_plan_t){
      .kind = LOOM_TESTBENCH_INVOCATION_PIPELINE,
      .module = module,
      .op = pipeline_op,
      .callee_ref = pipeline_ref,
      .provider_id = LOOM_STRING_ID_INVALID,
      .provider = iree_string_view_empty(),
      .attrs = loom_named_attr_slice_empty(),
      .execution_epoch = 0,
      .launch_schedule_depth = 0,
      .workload_value_ids = NULL,
      .workload_count = 0,
      .input_value_ids = entry_block->arg_ids,
      .input_count = entry_block->arg_count,
      .result_value_ids = NULL,
      .result_count = 0,
  };
  return iree_ok_status();
}

iree_status_t loom_xdna_hal_testbench_adapt_scenario_function(
    const loom_testbench_compilation_t* compilation,
    const loomc_module_t* source_module,
    const loom_testbench_invocation_plan_t* source_invocation,
    const loom_target_profile_t* target_profile,
    iree_allocator_t host_allocator,
    loom_run_hal_target_function_adaptation_t* out_adaptation,
    loomc_module_t** out_module,
    loom_testbench_invocation_plan_t* out_invocation) {
  if (compilation == NULL || source_module == NULL ||
      source_invocation == NULL || target_profile == NULL ||
      out_adaptation == NULL || out_module == NULL || out_invocation == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA scenario adapter requires compilation, "
                            "module, invocation, target, and result state");
  }
  *out_adaptation = LOOM_RUN_HAL_TARGET_FUNCTION_ADAPTATION_UNSUPPORTED;
  *out_module = NULL;
  *out_invocation = (loom_testbench_invocation_plan_t){0};
  const loom_aie2p_target_profile_t* aie2p_profile =
      loom_aie2p_target_profile_cast(target_profile);
  if (aie2p_profile == NULL ||
      aie2p_profile->kind != LOOM_AIE2P_TARGET_KIND_ARRAY ||
      aie2p_profile->device_profile == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA scenario adapter requires an exact AIE2P array target");
  }
  if (source_invocation->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL ||
      source_invocation->workload_count != 0 ||
      source_invocation->input_count != 1 ||
      source_invocation->result_count != 0) {
    return iree_ok_status();
  }

  const loom_type_t input_type = loom_module_value_type(
      source_invocation->module, source_invocation->input_value_ids[0]);
  uint64_t element_count = 0;
  const uint32_t element_bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  if (!loom_type_is_tensor(input_type) || loom_type_rank(input_type) != 1 ||
      !loom_type_static_element_count(input_type, &element_count) ||
      element_count == 0 || input_type.encoding_id != 0 ||
      input_type.encoding_flags != 0 || element_bitwidth < 8 ||
      element_bitwidth % 8 != 0) {
    return iree_ok_status();
  }
  const uint64_t element_byte_length = element_bitwidth / 8;
  if (element_count > (uint64_t)INT64_MAX / element_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA scenario tensor byte length overflows i64");
  }
  const int64_t byte_length = (int64_t)(element_count * element_byte_length);

  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(aie2p_profile->device_profile);
  const loom_xdna_tile_facts_t* compute_facts =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  if (family->column_count == 0 || compute_facts->row_count < 2) {
    return iree_ok_status();
  }
  if ((uint64_t)byte_length * 2 > compute_facts->memory.local_capacity) {
    return iree_ok_status();
  }

  const loom_module_t* source_native_module = source_invocation->module;
  const loom_symbol_ref_t function_ref = source_invocation->callee_ref;
  if (source_native_module == NULL || !loom_symbol_ref_is_valid(function_ref) ||
      function_ref.module_id != 0 ||
      function_ref.symbol_id >= source_native_module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA scenario invocation does not identify a source function");
  }
  const loom_op_t* source_function_op =
      source_native_module->symbols.entries[function_ref.symbol_id].defining_op;
  if (!loom_func_def_isa(source_function_op)) {
    return iree_ok_status();
  }
  const loom_func_like_t source_function =
      loom_func_like_const_cast(source_native_module, source_function_op);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_function, &source_argument_count);
  if (source_argument_count != 1 ||
      !loom_type_is_buffer(
          loom_module_value_type(source_native_module, source_arguments[0])) ||
      source_function_op->result_count != 0) {
    return iree_ok_status();
  }

  loomc_module_t* module = NULL;
  iree_status_t status = iree_status_from_loomc(
      loomc_module_clone(source_module, compilation->workspace,
                         loomc_allocator_from_iree(host_allocator), &module));
  loom_module_t* native_module = NULL;
  loom_op_t* function_op = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_module_mutable_interop_view_t view =
        loomc_module_get_mutable_interop_view(module);
    native_module = view.module;
    if (native_module == NULL) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "XDNA scenario adapter clone has no exact-version native IR");
    } else {
      IREE_ASSERT(function_ref.symbol_id < native_module->symbols.count);
      function_op =
          native_module->symbols.entries[function_ref.symbol_id].defining_op;
      IREE_ASSERT(loom_func_def_isa(function_op));
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_xdna_scenario_build_pipeline(
        native_module, function_ref, function_op, input_type, element_count,
        byte_length, (int64_t)element_byte_length, aie2p_profile,
        out_invocation);
  }
  if (iree_status_is_ok(status)) {
    *out_module = module;
    *out_adaptation = LOOM_RUN_HAL_TARGET_FUNCTION_ADAPTATION_ADAPTED;
    module = NULL;
  }
  loomc_module_release(module);
  return status;
}
