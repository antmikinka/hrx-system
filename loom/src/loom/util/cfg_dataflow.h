// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_CFG_DATAFLOW_H_
#define LOOM_UTIL_CFG_DATAFLOW_H_

#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects one source's outgoing facts through the destination's retained
// transfer and joins that contribution into its output. Facts and summaries
// belong to the caller; no IR is walked. The transfer is monotone over a
// finite-height domain and sets |out_changed| when destination output grows.
// Allocation needed to retain facts may fail.
typedef iree_status_t (*loom_cfg_dataflow_transfer_fn_t)(
    void* user_data, uint16_t source_block_index, uint16_t target_block_index,
    bool* out_changed);

// Solves a forward dataflow problem over an immutable, verified CFG. The
// caller publishes local seeds before the solve. Reachable blocks initially
// propagate in reverse postorder; subsequent visits follow changed outputs.
// Each visit propagates directly through outgoing edges. Unreachable blocks
// contribute no facts. The transfer owns the domain-specific join and storage.
//
// Worklist storage is bounded by block count and allocated once in |arena|.
// The graph and caller-owned facts must remain valid throughout the solve.
iree_status_t loom_cfg_dataflow_solve(const loom_cfg_graph_t* graph,
                                      loom_cfg_dataflow_transfer_fn_t transfer,
                                      void* user_data,
                                      iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_CFG_DATAFLOW_H_
