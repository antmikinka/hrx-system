// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_dataflow.h"

#include <string.h>

iree_status_t loom_cfg_dataflow_solve(const loom_cfg_graph_t* graph,
                                      loom_cfg_dataflow_transfer_fn_t transfer,
                                      void* user_data,
                                      iree_arena_allocator_t* arena) {
  const iree_host_size_t capacity = graph->block_count;
  iree_host_size_t count = graph->reverse_postorder.count;
  if (count == 0) {
    return iree_ok_status();
  }
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, capacity * (sizeof(*worklist) + sizeof(uint8_t)),
      (void**)&worklist));
  uint8_t* queued = (uint8_t*)(worklist + capacity);
  memset(queued, 0, capacity);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const uint16_t block_index = graph->reverse_postorder.values[i];
    worklist[i] = block_index;
    queued[block_index] = 1;
  }
  iree_host_size_t head = 0;
  iree_host_size_t tail = count == capacity ? 0 : count;
  iree_status_t status = iree_ok_status();
  while (count != 0 && iree_status_is_ok(status)) {
    const uint16_t block_index = worklist[head];
    if (++head == capacity) {
      head = 0;
    }
    --count;
    queued[block_index] = 0;
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0;
         i < successors.count && iree_status_is_ok(status); ++i) {
      const uint16_t successor = successors.values[i];
      bool changed = false;
      status = transfer(user_data, block_index, successor, &changed);
      if (!iree_status_is_ok(status) || !changed || queued[successor]) {
        continue;
      }
      worklist[tail] = successor;
      if (++tail == capacity) {
        tail = 0;
      }
      ++count;
      queued[successor] = 1;
    }
  }
  return status;
}
