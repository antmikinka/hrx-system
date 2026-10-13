// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/store_data_wait.h"

#include <string.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/cfg_dataflow.h"
#include "loom/util/cfg_graph.h"

// The two-slot horizon bounds both live local sources and relevant prefix
// writes. Bit planes carry joined obligations without retaining producer lists
// whose size would grow with CFG fan-in. Plane 1 is a subset of plane 0.
enum {
  LOOM_AMDGPU_STORE_DATA_WAIT_VALU = 0,
  LOOM_AMDGPU_STORE_DATA_WAIT_OTHER = 1,
  LOOM_AMDGPU_STORE_DATA_WAIT_PREFIX_PLANES = 4,
  LOOM_AMDGPU_STORE_DATA_WAIT_BLOCK_PLANES = 6,
};

typedef struct loom_amdgpu_store_data_source_t {
  // Issue position immediately after the source instruction.
  uint64_t end_position;
  // Schedule node issuing the source.
  uint32_t node_index;
  // First physical VGPR in the payload.
  uint32_t base;
  // Number of physical VGPRs in the payload.
  uint16_t count;
  // Intervening slots required before VALU reuse, or zero for an empty row.
  uint8_t cycles;
} loom_amdgpu_store_data_source_t;

typedef struct loom_amdgpu_store_data_prefix_t {
  // Scheduled packet containing this prefix write.
  uint32_t packet_index;
  // Position at which a delay before that packet is inserted.
  uint8_t packet_position;
} loom_amdgpu_store_data_prefix_t;

typedef struct loom_amdgpu_store_data_block_t {
  // Writes in the first two issue slots, corresponding to the prefix bit
  // planes.
  loom_amdgpu_store_data_prefix_t prefix[2];
  // Sources remaining live at block exit; cycles holds their residual window.
  loom_amdgpu_store_data_source_t sources[2];
  // Block issue length saturated at the retention horizon.
  uint8_t cycles;
} loom_amdgpu_store_data_block_t;

struct loom_amdgpu_store_data_wait_state_t {
  // Schedule owning packet order and the already-built CFG.
  const loom_low_schedule_table_t* schedule;
  // Final allocation supplying physical payloads and writes.
  const loom_low_allocation_table_t* allocation;
  // Arena owning transient summaries and propagation worklists.
  iree_arena_allocator_t* arena;
  // Block summaries in region order.
  loom_amdgpu_store_data_block_t* blocks;
  // Four prefix-write and two outgoing-retention bit planes per block.
  uint64_t* bits;
  // Four write bit planes for the currently inspected packet.
  uint64_t* writes;
  // Number of words per physical VGPR bit plane.
  iree_host_size_t word_count;
  // The two most recently issued local payloads.
  loom_amdgpu_store_data_source_t sources[2];
  // Source published by the currently inspected packet, or cycles zero.
  loom_amdgpu_store_data_source_t pending_source;
  // Conservative executed issue length of the inspected packet.
  uint64_t pending_cycles;
  // Current issue position within the active block.
  uint64_t position;
  // Active region block index.
  uint16_t block_index;
  // Next local source slot, alternating between the two retained rows.
  uint8_t next_source;
};

static uint64_t* loom_amdgpu_store_data_block_bits(
    const loom_amdgpu_store_data_wait_state_t* state, uint16_t block_index,
    unsigned plane) {
  return state->bits +
         (block_index * LOOM_AMDGPU_STORE_DATA_WAIT_BLOCK_PLANES + plane) *
             state->word_count;
}

static void loom_amdgpu_store_data_set_range(uint64_t* bits, uint32_t base,
                                             uint32_t count) {
  for (uint32_t i = base; i < base + count; ++i) {
    bits[i / 64] |= UINT64_C(1) << (i % 64);
  }
}

iree_status_t loom_amdgpu_store_data_wait_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena,
    loom_amdgpu_store_data_wait_state_t** out_state) {
  *out_state = NULL;
  const loom_amdgpu_descriptor_set_info_t* info =
      loom_amdgpu_target_info_descriptor_set_at(
          schedule->target.descriptor_set->descriptor_set_ordinal);
  const uint32_t vgpr_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  if (!iree_any_bit_set(
          info->flags,
          LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_STORE_DATA_WAIT_STATES) ||
      vgpr_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_store_data_wait_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->schedule = schedule;
  state->allocation = allocation;
  state->arena = arena;
  state->word_count = (vgpr_count + 63) / 64;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->block_count,
                                                 sizeof(*state->blocks),
                                                 (void**)&state->blocks));
  memset(state->blocks, 0, schedule->block_count * sizeof(*state->blocks));
  const iree_host_size_t bit_count = schedule->block_count *
                                     LOOM_AMDGPU_STORE_DATA_WAIT_BLOCK_PLANES *
                                     state->word_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bit_count, sizeof(*state->bits), (void**)&state->bits));
  memset(state->bits, 0, bit_count * sizeof(*state->bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, LOOM_AMDGPU_STORE_DATA_WAIT_PREFIX_PLANES * state->word_count,
      sizeof(*state->writes), (void**)&state->writes));
  *out_state = state;
  return iree_ok_status();
}

void loom_amdgpu_store_data_wait_begin_block(
    loom_amdgpu_store_data_wait_state_t* state, uint16_t block_index) {
  state->block_index = block_index;
  state->position = 0;
  state->next_source = 0;
  memset(state->sources, 0, sizeof(state->sources));
}

static void loom_amdgpu_store_data_record_write(
    loom_amdgpu_store_data_wait_state_t* state, uint32_t base, uint32_t count,
    unsigned offset, unsigned kind) {
  loom_amdgpu_store_data_set_range(
      state->writes + (offset * 2 + kind) * state->word_count, base, count);
}

loom_amdgpu_store_data_wait_match_t loom_amdgpu_store_data_wait_inspect(
    loom_amdgpu_store_data_wait_state_t* state,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits) {
  state->pending_source = (loom_amdgpu_store_data_source_t){0};
  state->pending_cycles =
      packet->descriptor != NULL ? 1 : structural->instruction_count;
  // With two emitted branches, the taken conditional edge executes only the
  // first. Both successors must receive that conservative progress bound.
  if (packet->descriptor == NULL && loom_low_cond_br_isa(packet->node->op) &&
      state->pending_cycles == 2) {
    state->pending_cycles = 1;
  }
  if (iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
    const loom_low_descriptor_set_t* descriptor_set =
        state->schedule->target.descriptor_set;
    const loom_amdgpu_store_data_wait_t binding =
        loom_amdgpu_descriptor_store_data_wait(descriptor_set,
                                               packet->descriptor);
    if (binding.cycles != 0) {
      const loom_low_allocation_assignment_t* payload =
          loom_low_packet_descriptor_operand_assignment(
              state->allocation, packet, binding.operand_index);
      state->pending_source = (loom_amdgpu_store_data_source_t){
          .node_index = packet->node_index,
          .base = payload->location_base,
          .count = payload->location_count,
          .cycles = binding.cycles,
      };
    }
  }
  // Outside the two-slot prefix, only a still-live local payload can need a
  // write footprint. Ordinary arithmetic stretches skip all bitset work.
  if (state->position >= 2 &&
      state->position >=
          state->sources[0].end_position + state->sources[0].cycles &&
      state->position >=
          state->sources[1].end_position + state->sources[1].cycles) {
    return (loom_amdgpu_store_data_wait_match_t){0};
  }
  memset(state->writes, 0, 4 * state->word_count * sizeof(*state->writes));
  if (structural->moves.count != 0) {
    const iree_host_size_t count =
        structural->moves.count < 2 ? structural->moves.count : 2;
    for (iree_host_size_t i = 0; i < count; ++i) {
      const loom_low_move_t* move =
          &state->allocation->moves[structural->moves.start + i];
      if (move->destination.descriptor_reg_class_id ==
          LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
        loom_amdgpu_store_data_record_write(state, move->destination.location,
                                            1, (unsigned)i,
                                            LOOM_AMDGPU_STORE_DATA_WAIT_VALU);
      }
    }
  } else if (state->pending_cycles != 0) {
    const unsigned kind =
        iree_any_bit_set(traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU |
                                     LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX) ||
                structural->vector_alu_instruction_count != 0
            ? LOOM_AMDGPU_STORE_DATA_WAIT_VALU
            : LOOM_AMDGPU_STORE_DATA_WAIT_OTHER;
    for (uint16_t i = 0; i < packet->node->result_count; ++i) {
      const loom_low_allocation_assignment_t* result =
          loom_low_packet_result_assignment(state->allocation, packet, i);
      if (loom_low_allocation_assignment_is_physical_register_class(
              result, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
        loom_amdgpu_store_data_record_write(state, result->location_base,
                                            result->location_count, 0, kind);
      }
    }
  }
  loom_amdgpu_store_data_wait_match_t match = {0};
  for (unsigned i = 0; i < 2; ++i) {
    const loom_amdgpu_store_data_source_t* source = &state->sources[i];
    for (unsigned offset = 0; offset < 2; ++offset) {
      const uint64_t observed = state->position + offset - source->end_position;
      for (unsigned kind = 0; kind < 2; ++kind) {
        const int required = source->cycles - kind;
        if (required <= 0 || observed >= (uint64_t)required ||
            required - observed <= match.cycles) {
          continue;
        }
        const uint64_t* writes =
            state->writes + (offset * 2 + kind) * state->word_count;
        for (uint32_t reg = source->base; reg < source->base + source->count;
             ++reg) {
          if (writes[reg / 64] & (UINT64_C(1) << (reg % 64))) {
            match = (loom_amdgpu_store_data_wait_match_t){
                .producer_node = source->node_index,
                .required_cycles = (uint16_t)required,
                .observed_cycles = (uint16_t)observed,
                .cycles = (uint16_t)(required - observed),
            };
            break;
          }
        }
      }
    }
  }
  return match;
}

void loom_amdgpu_store_data_wait_advance(
    loom_amdgpu_store_data_wait_state_t* state, uint64_t cycles) {
  state->position += cycles;
}

void loom_amdgpu_store_data_wait_commit(
    loom_amdgpu_store_data_wait_state_t* state,
    const loom_low_packet_view_t* packet) {
  loom_amdgpu_store_data_block_t* block = &state->blocks[state->block_index];
  for (unsigned offset = 0; offset < 2 && state->position + offset < 2;
       ++offset) {
    const unsigned position = (unsigned)state->position + offset;
    uint64_t* prefix = loom_amdgpu_store_data_block_bits(
        state, state->block_index, position * 2);
    const uint64_t* writes = state->writes + offset * 2 * state->word_count;
    bool has_writes = false;
    for (iree_host_size_t i = 0; i < 2 * state->word_count; ++i) {
      prefix[i] |= writes[i];
      has_writes |= writes[i] != 0;
    }
    if (has_writes) {
      block->prefix[position] = (loom_amdgpu_store_data_prefix_t){
          .packet_index = (uint32_t)packet->packet_index,
          .packet_position = (uint8_t)state->position,
      };
    }
  }
  state->position += state->pending_cycles;
  if (state->pending_source.cycles != 0) {
    state->pending_source.end_position = state->position;
    state->sources[state->next_source] = state->pending_source;
    state->next_source ^= 1;
  }
}

void loom_amdgpu_store_data_wait_end_block(
    loom_amdgpu_store_data_wait_state_t* state) {
  loom_amdgpu_store_data_block_t* block = &state->blocks[state->block_index];
  block->cycles = (uint8_t)(state->position < 2 ? state->position : 2);
  for (unsigned i = 0; i < 2; ++i) {
    const loom_amdgpu_store_data_source_t* source = &state->sources[i];
    const uint64_t elapsed = state->position - source->end_position;
    if (elapsed >= source->cycles) {
      continue;
    }
    block->sources[i] = *source;
    block->sources[i].cycles -= (uint8_t)elapsed;
    for (unsigned plane = 0; plane < block->sources[i].cycles; ++plane) {
      loom_amdgpu_store_data_set_range(
          loom_amdgpu_store_data_block_bits(state, state->block_index,
                                            4 + plane),
          source->base, source->count);
    }
  }
}

// Joins retained predecessor frontiers, advancing them through this block.
static uint64_t loom_amdgpu_store_data_incoming_word(
    const loom_amdgpu_store_data_wait_state_t* state,
    loom_cfg_block_index_span_t predecessors, unsigned plane,
    iree_host_size_t word) {
  uint64_t bits = 0;
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    if (!loom_cfg_graph_block_is_reachable(&state->schedule->cfg_graph,
                                           predecessors.values[i])) {
      continue;
    }
    bits |= loom_amdgpu_store_data_block_bits(state, predecessors.values[i],
                                              4 + plane)[word];
  }
  return bits;
}

static iree_status_t loom_amdgpu_store_data_transfer(
    void* user_data, uint16_t source_block_index, uint16_t target_block_index,
    bool* out_changed) {
  loom_amdgpu_store_data_wait_state_t* state =
      (loom_amdgpu_store_data_wait_state_t*)user_data;
  const loom_amdgpu_store_data_block_t* target =
      &state->blocks[target_block_index];
  *out_changed = false;
  for (unsigned plane = 0; plane + target->cycles < 2; ++plane) {
    uint64_t* outgoing =
        loom_amdgpu_store_data_block_bits(state, target_block_index, 4 + plane);
    const uint64_t* incoming = loom_amdgpu_store_data_block_bits(
        state, source_block_index, 4 + plane + target->cycles);
    for (iree_host_size_t word = 0; word < state->word_count; ++word) {
      *out_changed |= (incoming[word] & ~outgoing[word]) != 0;
      outgoing[word] |= incoming[word];
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_store_data_wait_resolve(
    loom_amdgpu_store_data_wait_state_t* state,
    loom_amdgpu_store_data_wait_emit_fn_t emit, void* emit_user_data) {
  if (state->schedule->cfg_graph.edge_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_dataflow_solve(&state->schedule->cfg_graph,
                                               loom_amdgpu_store_data_transfer,
                                               state, state->arena));
  loom_amdgpu_wait_state_t pending = {0};
  bool has_pending = false;
  iree_status_t status = iree_ok_status();
  // Reuse packet-write scratch for one incoming frontier. Resolved predecessor
  // outputs replace static bounds as we proceed; backedges retain safe bounds.
  uint64_t* incoming = state->writes;
  for (uint16_t index = 0;
       index < state->schedule->block_count && iree_status_is_ok(status);
       ++index) {
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(&state->schedule->cfg_graph, index);
    for (unsigned plane = 0; plane < 2; ++plane) {
      for (iree_host_size_t word = 0; word < state->word_count; ++word) {
        incoming[plane * state->word_count + word] =
            loom_amdgpu_store_data_incoming_word(state, predecessors, plane,
                                                 word);
      }
    }
    const loom_amdgpu_store_data_block_t* block = &state->blocks[index];
    uint8_t inserted[2] = {0};
    unsigned delay = 0;
    for (unsigned position = 0;
         position + delay < 2 && iree_status_is_ok(status); ++position) {
      unsigned residual = 0;
      unsigned required = 0;
      for (unsigned kind = 0; kind < 2; ++kind) {
        const uint64_t* writes = loom_amdgpu_store_data_block_bits(
            state, index, position * 2 + kind);
        for (unsigned plane = position + delay + kind; plane < 2; ++plane) {
          for (iree_host_size_t word = 0; word < state->word_count; ++word) {
            if (writes[word] & incoming[plane * state->word_count + word]) {
              const unsigned candidate = plane + 1 - kind - position - delay;
              if (candidate > residual) {
                required = plane + 1 - kind;
                residual = candidate;
              }
            }
          }
        }
      }
      if (residual == 0) {
        continue;
      }
      const loom_amdgpu_store_data_prefix_t* prefix = &block->prefix[position];
      const loom_low_packet_view_t packet =
          loom_low_packet_at(state->schedule, prefix->packet_index);
      if (has_pending && pending.node_index == packet.node_index) {
        pending.cycle_count += (uint16_t)residual;
        pending.required_cycle_count += (uint16_t)residual;
      } else {
        if (has_pending) {
          status = emit(emit_user_data, &pending);
        }
        if (!iree_status_is_ok(status)) {
          break;
        }
        pending = (loom_amdgpu_wait_state_t){
            .reason = LOOM_AMDGPU_WAIT_STATE_REASON_STORE_DATA_REUSE,
            .action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP,
            .block_index = index,
            .node_index = packet.node_index,
            .scheduled_ordinal = packet.node->scheduled_ordinal,
            .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
            .consumer_node = packet.node->source_ordinal,
            .required_cycle_count = (uint16_t)required,
            .observed_cycle_count = (uint16_t)(position + delay),
            .cycle_count = (uint16_t)residual,
        };
        has_pending = true;
      }
      inserted[prefix->packet_position] += (uint8_t)residual;
      delay += residual;
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    // Incoming windows see all issue progress. Locally opened windows only
    // see inserted waits after their own source instruction.
    for (unsigned plane = 0; plane < 2; ++plane) {
      uint64_t* outgoing =
          loom_amdgpu_store_data_block_bits(state, index, 4 + plane);
      const unsigned shifted = plane + block->cycles + delay;
      for (iree_host_size_t word = 0; word < state->word_count; ++word) {
        outgoing[word] =
            shifted < 2 ? incoming[shifted * state->word_count + word] : 0;
      }
    }
    for (unsigned i = 0; i < 2; ++i) {
      const loom_amdgpu_store_data_source_t* source = &block->sources[i];
      unsigned elapsed = 0;
      for (unsigned position = 0; position < 2; ++position) {
        if (position >= source->end_position) {
          elapsed += inserted[position];
        }
      }
      for (unsigned plane = elapsed; plane < source->cycles; ++plane) {
        loom_amdgpu_store_data_set_range(loom_amdgpu_store_data_block_bits(
                                             state, index, 4 + plane - elapsed),
                                         source->base, source->count);
      }
    }
  }
  if (iree_status_is_ok(status) && has_pending) {
    status = emit(emit_user_data, &pending);
  }
  return status;
}
