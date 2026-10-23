/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cstdlib>
#include <optional>

namespace cudf {
namespace detail {
namespace {

// Below this many rows the setup kernels and syncs cost more than the comparison sort.
constexpr size_type min_lsd_rows = 1 << 16;
// Each key column costs one radix sort pass over all rows.
constexpr size_type max_lsd_columns = 8;
// A string key is ranked through its sorted distinct values, which pays off only when they are
// few relative to the rows.
constexpr int64_t max_distinct_string_fraction = 4;

template <typename T>
constexpr bool is_lsd_value_type()
{
  return (cuda::std::is_integral_v<T> or std::is_same_v<T, __int128_t> or cudf::is_chrono<T>()) and
         not cudf::is_floating_point<T>();
}

template <typename T>
__device__ __int128_t value_as_int128(column_device_view const& col, size_type row)
{
  if constexpr (cudf::is_timestamp<T>()) {
    return static_cast<__int128_t>(col.element<T>(row).time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return static_cast<__int128_t>(col.element<T>(row).count());
  } else {
    return static_cast<__int128_t>(col.element<T>(row));
  }
}

struct value_range {
  __int128_t lo;
  __int128_t hi;
};

struct value_range_op {
  __device__ value_range operator()(value_range const& a, value_range const& b) const
  {
    return {a.lo < b.lo ? a.lo : b.lo, a.hi > b.hi ? a.hi : b.hi};
  }
};

template <typename T>
struct row_range_fn {
  column_device_view values;
  column_device_view validity;
  __device__ value_range operator()(size_type row) const
  {
    if (validity.is_null(row)) {
      return {cuda::std::numeric_limits<__int128_t>::max(),
              cuda::std::numeric_limits<__int128_t>::min()};
    }
    auto const v = value_as_int128<T>(values, row);
    return {v, v};
  }
};

/**
 * Builds the ascending radix key of each row in `perm` order: values mapped to [0, hi - lo] in
 * the requested direction, with nulls folded in at the smallest or largest key.
 */
template <typename T, typename Key>
struct row_key_fn {
  column_device_view values;
  column_device_view validity;
  size_type const* perm;
  __int128_t lo;
  __int128_t hi;
  bool ascending;
  Key value_offset;  // 1 when nulls take key 0
  Key null_key;
  __device__ Key operator()(size_type i) const
  {
    auto const row = perm[i];
    if (validity.is_null(row)) { return null_key; }
    auto const v = value_as_int128<T>(values, row);
    return static_cast<Key>(ascending ? v - lo : hi - v) + value_offset;
  }
};

struct column_range_dispatch {
  template <typename T>
  std::optional<value_range> operator()(column_view const& values,
                                        column_view const& validity,
                                        rmm::cuda_stream_view stream)
    requires(is_lsd_value_type<T>())
  {
    auto const d_values   = column_device_view::create(values, stream);
    auto const d_validity = column_device_view::create(validity, stream);
    return thrust::transform_reduce(
      rmm::exec_policy(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{values.size()},
      row_range_fn<T>{*d_values, *d_validity},
      value_range{cuda::std::numeric_limits<__int128_t>::max(),
                  cuda::std::numeric_limits<__int128_t>::min()},
      value_range_op{});
  }

  template <typename T>
  std::optional<value_range> operator()(column_view const&, column_view const&, rmm::cuda_stream_view)
    requires(not is_lsd_value_type<T>())
  {
    return std::nullopt;
  }
};

struct sort_pass_dispatch {
  template <typename Key>
  static void radix_pass(Key const* keys,
                         size_type const* perm_in,
                         size_type* perm_out,
                         size_type num_rows,
                         int end_bit,
                         rmm::cuda_stream_view stream)
  {
    rmm::device_uvector<Key> sorted_keys(num_rows, stream);
    std::size_t temp_bytes{};
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    temp_bytes,
                                    keys,
                                    sorted_keys.data(),
                                    perm_in,
                                    perm_out,
                                    num_rows,
                                    0,
                                    end_bit,
                                    stream.value());
    rmm::device_buffer temp(temp_bytes, stream);
    cub::DeviceRadixSort::SortPairs(temp.data(),
                                    temp_bytes,
                                    keys,
                                    sorted_keys.data(),
                                    perm_in,
                                    perm_out,
                                    num_rows,
                                    0,
                                    end_bit,
                                    stream.value());
  }

  template <typename T, typename Key>
  static void keyed_pass(column_device_view const& values,
                         column_device_view const& validity,
                         value_range range,
                         bool ascending,
                         bool nulls_first,
                         int end_bit,
                         size_type const* perm_in,
                         size_type* perm_out,
                         size_type num_rows,
                         rmm::cuda_stream_view stream)
  {
    auto const span       = static_cast<Key>(range.hi - range.lo);
    auto const has_nulls  = validity.nullable();
    auto const offset     = static_cast<Key>(has_nulls and nulls_first ? 1 : 0);
    auto const null_key   = static_cast<Key>(nulls_first ? 0 : span + 1);
    rmm::device_uvector<Key> keys(num_rows, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{num_rows},
                      keys.begin(),
                      row_key_fn<T, Key>{
                        values, validity, perm_in, range.lo, range.hi, ascending, offset, null_key});
    radix_pass(keys.data(), perm_in, perm_out, num_rows, end_bit, stream);
  }

  template <typename T>
  void operator()(column_view const& values,
                  column_view const& validity,
                  value_range range,
                  bool ascending,
                  bool nulls_first,
                  size_type const* perm_in,
                  size_type* perm_out,
                  rmm::cuda_stream_view stream)
    requires(is_lsd_value_type<T>())
  {
    auto const d_values   = column_device_view::create(values, stream);
    auto const d_validity = column_device_view::create(validity, stream);
    // Largest key: the value span, plus one slot for nulls
    auto const max_key = static_cast<unsigned __int128>(range.hi - range.lo) +
                         static_cast<unsigned __int128>(validity.nullable() ? 1 : 0);
    auto const end_bit =
      std::max(1, static_cast<int>(cuda::std::bit_width(static_cast<uint64_t>(max_key))));
    auto const num_rows = values.size();
    if (end_bit <= 32) {
      keyed_pass<T, uint32_t>(*d_values,
                              *d_validity,
                              range,
                              ascending,
                              nulls_first,
                              end_bit,
                              perm_in,
                              perm_out,
                              num_rows,
                              stream);
    } else {
      keyed_pass<T, uint64_t>(*d_values,
                              *d_validity,
                              range,
                              ascending,
                              nulls_first,
                              end_bit,
                              perm_in,
                              perm_out,
                              num_rows,
                              stream);
    }
  }

  template <typename T>
  void operator()(column_view const&,
                  column_view const&,
                  value_range,
                  bool,
                  bool,
                  size_type const*,
                  size_type*,
                  rmm::cuda_stream_view)
    requires(not is_lsd_value_type<T>())
  {
    CUDF_FAIL("Unsupported type for LSD sort");
  }
};

bool is_lsd_key_column(column_view const& col)
{
  auto const id = col.type().id();
  return id == type_id::STRING or
         (cudf::is_fixed_width(col.type()) and not cudf::is_floating_point(col.type()) and
          not cudf::is_dictionary(col.type()));
}

/**
 * Ranks each row of a strings column by its value among the sorted distinct values, or returns
 * nullptr when there are too many distinct values for ranking to pay off.
 */
std::unique_ptr<column> string_ranks(column_view const& col,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const keys     = table_view{{col}};
  auto const distinct = cudf::detail::distinct(keys,
                                               {0},
                                               duplicate_keep_option::KEEP_ANY,
                                               null_equality::EQUAL,
                                               nan_equality::ALL_EQUAL,
                                               stream,
                                               mr);
  if (static_cast<int64_t>(distinct->num_rows()) * max_distinct_string_fraction >
      static_cast<int64_t>(col.size())) {
    return nullptr;
  }
  auto const sorted =
    cudf::detail::sort(distinct->view(), {order::ASCENDING}, {null_order::AFTER}, stream, mr);
  return cudf::detail::lower_bound(
    sorted->view(), keys, {order::ASCENDING}, {null_order::AFTER}, stream, mr);
}

bool lsd_disabled()
{
  static bool const disabled = std::getenv("CUDF_SORT_LSD_DISABLE") != nullptr;
  return disabled;
}

}  // namespace

std::unique_ptr<column> sorted_order_lsd(table_view const& input,
                                         std::vector<order> const& column_order,
                                         std::vector<null_order> const& null_precedence,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.num_rows();
  auto const num_cols = input.num_columns();
  if (lsd_disabled() or num_cols < 2 or num_cols > max_lsd_columns or num_rows < min_lsd_rows) {
    return nullptr;
  }
  for (auto const& col : input) {
    if (not is_lsd_key_column(col)) { return nullptr; }
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  // Strings are replaced by their dense ranks; every key is then an integer with a known range.
  std::vector<std::unique_ptr<column>> ranks(num_cols);
  std::vector<column_view> values(num_cols);
  std::vector<value_range> ranges(num_cols);
  for (size_type c = 0; c < num_cols; ++c) {
    auto const col = input.column(c);
    if (col.type().id() == type_id::STRING) {
      ranks[c] = string_ranks(col, stream, temp_mr);
      if (not ranks[c]) { return nullptr; }
      values[c] = ranks[c]->view();
    } else {
      values[c] = col;
    }
    auto const range = cudf::type_dispatcher<dispatch_storage_type>(
      values[c].type(), column_range_dispatch{}, values[c], col, stream);
    if (not range.has_value()) { return nullptr; }
    // An all-null column sorts like a constant one
    ranges[c] = range->lo > range->hi ? value_range{0, 0} : *range;
    auto const span = static_cast<unsigned __int128>(ranges[c].hi - ranges[c].lo);
    // Keys must fit in 64 bits including the null slot
    if (span >= static_cast<unsigned __int128>(cuda::std::numeric_limits<uint64_t>::max())) {
      return nullptr;
    }
  }

  // Least-significant key first: each stable pass keeps the order of the passes before it.
  rmm::device_uvector<size_type> perm_a(num_rows, stream, temp_mr);
  rmm::device_uvector<size_type> perm_b(num_rows, stream, temp_mr);
  thrust::sequence(rmm::exec_policy_nosync(stream, temp_mr), perm_a.begin(), perm_a.end(), 0);
  auto* perm_in  = perm_a.data();
  auto* perm_out = perm_b.data();
  for (size_type c = num_cols - 1; c >= 0; --c) {
    auto const ascending =
      column_order.empty() or column_order[c] == order::ASCENDING;
    auto const nulls_before =
      null_precedence.empty() or null_precedence[c] == null_order::BEFORE;
    // A null compares as the smallest value under BEFORE; descending order then reverses that
    auto const nulls_first = nulls_before == ascending;
    cudf::type_dispatcher<dispatch_storage_type>(values[c].type(),
                                                 sort_pass_dispatch{},
                                                 values[c],
                                                 input.column(c),
                                                 ranges[c],
                                                 ascending,
                                                 nulls_first,
                                                 perm_in,
                                                 perm_out,
                                                 stream);
    std::swap(perm_in, perm_out);
  }

  auto result = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), num_rows, mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(result->mutable_view().data<size_type>(),
                                perm_in,
                                num_rows * sizeof(size_type),
                                cudaMemcpyDefault,
                                stream.value()));
  return result;
}

}  // namespace detail
}  // namespace cudf
