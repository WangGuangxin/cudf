/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/filling.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <string>
#include <vector>

namespace {

// Payload table shapes. Column 0 is always INT64 so hash_partition can hash on it.
std::vector<cudf::type_id> payload_types(std::string const& payload)
{
  using cudf::type_id;
  if (payload == "int64x1") { return {type_id::INT64}; }
  if (payload == "int64x8") { return std::vector<type_id>(8, type_id::INT64); }
  if (payload == "string") { return {type_id::INT64, type_id::STRING}; }
  if (payload == "mixed") {
    return {type_id::INT64, type_id::INT32, type_id::FLOAT64, type_id::STRING, type_id::LIST};
  }
  CUDF_FAIL("unknown payload " + payload);
}

std::unique_ptr<cudf::table> make_payload(std::string const& payload, cudf::size_type num_rows)
{
  auto builder = data_profile_builder()
                   .cardinality(0)
                   .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, 32)
                   .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, 8)
                   .list_depth(1)
                   .list_type(cudf::type_id::INT32);
  if (payload == "mixed") {
    builder.null_probability(0.1);
  } else {
    builder.no_validity();
  }
  return create_random_table(payload_types(payload), row_count{num_rows}, builder);
}

std::unique_ptr<cudf::column> make_partition_map(std::string const& dist,
                                                 cudf::size_type num_rows,
                                                 cudf::size_type num_partitions)
{
  if (dist == "single") {
    // Every row maps to the same partition: worst case for per-partition atomic counters
    return cudf::sequence(
      num_rows, cudf::numeric_scalar<int32_t>(0), cudf::numeric_scalar<int32_t>(0));
  }
  auto const id = dist == "skewed" ? distribution_id::GEOMETRIC : distribution_id::UNIFORM;
  auto map      = create_random_column(cudf::type_id::INT32,
                                  row_count{num_rows},
                                  data_profile_builder().cardinality(0).no_validity().distribution(
                                    cudf::type_id::INT32, id, 0, num_partitions - 1),
                                  7);
  if (dist == "sorted") {
    auto sorted = cudf::sort(cudf::table_view{{map->view()}});
    return std::move(sorted->release().front());
  }
  return map;
}

}  // namespace

static void bench_partition(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto const dist           = state.get_string("map_dist");
  auto const payload        = state.get_string("payload");

  if (dist == "skewed" && num_partitions < 16) {
    state.skip("geometric skew needs a wider partition range");
    return;
  }

  auto const input = make_payload(payload, num_rows);
  auto const map   = make_partition_map(dist, num_rows, num_partitions);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::partition(input->view(), map->view(), num_partitions);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_hash_partition_paths(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto const payload        = state.get_string("payload");

  auto const input = make_payload(payload, num_rows);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::hash_partition(input->view(), {0}, num_partitions);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_partition)
  .set_name("partition")
  .add_int64_axis("num_rows", {1 << 20, 1 << 24})
  .add_int64_axis("num_partitions", {2, 16, 128, 1024, 8192, 65536})
  .add_string_axis("map_dist", {"uniform", "skewed", "single", "sorted"})
  .add_string_axis("payload", {"int64x1", "int64x8", "string", "mixed"});

// Partition counts chosen to hit each hash_partition path: <= 1024 takes the shared-memory copy
// path, 1025 up to the shared-memory histogram limit (~12K) takes the scatter-map fallback, and
// beyond that the global-memory atomic path runs.
NVBENCH_BENCH(bench_hash_partition_paths)
  .set_name("hash_partition_paths")
  .add_int64_axis("num_rows", {1 << 20, 1 << 24})
  .add_int64_axis("num_partitions", {256, 2048, 8192, 16384, 65536})
  .add_string_axis("payload", {"int64x1", "int64x8", "string", "mixed"});
