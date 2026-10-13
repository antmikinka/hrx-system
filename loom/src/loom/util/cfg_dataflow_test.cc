// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_dataflow.h"

#include <algorithm>
#include <random>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_graph_test_util.h"

namespace loom {
namespace {

// A finite release lattice: local windows join incoming windows after the
// block's guaranteed progress. Zero-progress cycles preserve pending releases.
struct ReleaseFlow {
  // Guaranteed progress through each block.
  std::vector<unsigned> progress;
  // Solved outgoing release bounds.
  std::vector<unsigned> outgoing;
  // Number of incoming edge contributions for each block.
  std::vector<unsigned> visits;
};

iree_status_t TransferRelease(void* user_data, uint16_t source_block_index,
                              uint16_t target_block_index, bool* out_changed) {
  auto& flow = *static_cast<ReleaseFlow*>(user_data);
  ++flow.visits[target_block_index];
  const unsigned incoming = flow.outgoing[source_block_index];
  const unsigned progress = flow.progress[target_block_index];
  const unsigned remaining = incoming > progress ? incoming - progress : 0;
  *out_changed = remaining > flow.outgoing[target_block_index];
  flow.outgoing[target_block_index] =
      std::max(flow.outgoing[target_block_index], remaining);
  return iree_ok_status();
}

class CfgDataflowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Check(const std::vector<std::vector<uint16_t>>& successors,
             const std::vector<unsigned>& generated,
             const std::vector<unsigned>& progress) {
    testing::CfgGraph fixture(successors);
    const auto* graph = fixture.get();
    // Seed local facts before propagation, as packet planners do.
    ReleaseFlow flow{progress, generated,
                     std::vector<unsigned>(successors.size())};
    std::vector<unsigned> reference = generated;
    bool changed;
    do {
      changed = false;
      for (uint16_t source = 0; source < successors.size(); ++source) {
        if (!graph->blocks[source].reachable) {
          continue;
        }
        for (uint16_t target : successors[source]) {
          const unsigned remaining = reference[source] > progress[target]
                                         ? reference[source] - progress[target]
                                         : 0;
          if (remaining > reference[target]) {
            reference[target] = remaining;
            changed = true;
          }
        }
      }
    } while (changed);
    IREE_ASSERT_OK(
        loom_cfg_dataflow_solve(graph, TransferRelease, &flow, &arena_));
    EXPECT_EQ(flow.outgoing, reference);
    for (size_t i = 0; i < successors.size(); ++i) {
      if (!graph->blocks[i].reachable) {
        EXPECT_EQ(flow.visits[i], 0);
      }
    }
    iree_arena_reset(&arena_);
  }

  // Reusable backing storage for transient solves.
  iree_arena_block_pool_t pool_;
  // Worklist lifetime, reset between graphs.
  iree_arena_allocator_t arena_;
};

TEST_F(CfgDataflowTest, EmptyJoinCyclesAndUnreachablePredecessors) {
  Check({}, {}, {});
  Check({{}}, {7}, {0});
  Check({{1, 2}, {3}, {3}, {}}, {8, 0, 0, 0}, {0, 1, 5, 1});
  Check({{1}, {2, 3}, {1}, {}}, {0, 0, 8, 0}, {0, 0, 0, 0});
  Check({{1, 1}, {1}}, {8, 0}, {0, 1});
  Check({{1}, {}, {1}}, {2, 0, 20}, {0, 1, 0});
}

TEST_F(CfgDataflowTest, ExhaustiveThreeBlockGraphs) {
  for (uint32_t mask = 0; mask < 512; ++mask) {
    SCOPED_TRACE(mask);
    std::vector<std::vector<uint16_t>> successors(3);
    for (uint16_t source = 0; source < 3; ++source) {
      for (uint16_t target = 0; target < 3; ++target) {
        if (mask & (1u << (3 * source + target))) {
          successors[source].push_back(target);
        }
      }
    }
    Check(successors, {2, 9, 5}, {0, 1, 2});
    Check(successors, {0, 0, 8}, {0, 0, 0});
  }
}

TEST_F(CfgDataflowTest, RandomReorderedGraphsMatchFullIteration) {
  std::mt19937 random(925718);
  for (unsigned trial = 0; trial < 1000; ++trial) {
    SCOPED_TRACE(trial);
    const unsigned count = 1 + random() % 32;
    std::vector<std::vector<uint16_t>> successors(count);
    std::vector<unsigned> generated(count), progress(count);
    for (unsigned i = 0; i < count; ++i) {
      for (unsigned edges = random() % 5; edges != 0; --edges) {
        successors[i].push_back(random() % count);
      }
      generated[i] = random() % 21;
      progress[i] = random() % 21;
    }
    Check(successors, generated, progress);
  }
}

TEST_F(CfgDataflowTest, AcyclicFlowTransfersEachEdgeOnce) {
  constexpr unsigned kCount = 4096;
  std::vector<std::vector<uint16_t>> successors(kCount);
  for (uint16_t i = 0; i + 1 < kCount; ++i) {
    successors[i].push_back(i + 1);
  }
  testing::CfgGraph fixture(successors);
  ReleaseFlow flow{std::vector<unsigned>(kCount), std::vector<unsigned>(kCount),
                   std::vector<unsigned>(kCount)};
  flow.outgoing[0] = 20;
  IREE_ASSERT_OK(
      loom_cfg_dataflow_solve(fixture.get(), TransferRelease, &flow, &arena_));
  EXPECT_EQ(flow.outgoing, std::vector<unsigned>(kCount, 20));
  std::vector<unsigned> expected_visits(kCount, 1);
  expected_visits[0] = 0;
  EXPECT_EQ(flow.visits, expected_visits);
}

}  // namespace
}  // namespace loom
