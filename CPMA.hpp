#ifndef CPMA_HPP
#define CPMA_HPP
#include "ParallelTools/ParallelTools/concurrent_hash_map.hpp"
#include "ParallelTools/ParallelTools/parallel.h"
#include "ParallelTools/concurrent_hash_map.hpp"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "ParallelTools/sort.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "ParallelTools/integerSort/blockRadixSort.h"

#if VQSORT == 1
#include <hwy/contrib/sort/vqsort.h>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"
#include "parlaylib/include/parlay/primitives.h"
#pragma clang diagnostic pop

// #include "AlignedAllocator2.hpp"
#include "helpers.h"
#include "leaf.hpp"
#include "timers.hpp"

enum HeadForm { InPlace, Linear, Eytzinger, BNary };
// BNAry has B pointers, and B-1 elements in each block

class empty_type {};

template <typename leaf, HeadForm head_form, uint64_t B_size = 0,
          bool store_density = false>
class CPMA {
  using T = typename leaf::value_type;
  static_assert(std::is_trivial<T>::value, "T must be a trivial type");
  static_assert(B_size == 0 || head_form == BNary,
                "B_size should only be used if we are using head_form = BNary");

private:
  static constexpr double growing_factor = 1.1; // 1.5;
  static constexpr int leaf_blow_up_factor = 16;
  static constexpr uint64_t min_leaf_size = 64;
  static_assert(min_leaf_size >= 64, "min_leaf_size must be at least 64 bytes");
  T *data_array;

  [[no_unique_address]]
  typename std::conditional<head_form == InPlace, empty_type, T *>::type
      head_array;

  // stored the density of each leaf to speed up merges and get_density_count
  // only use a uint16_t to save on space
  // this might not be enough to store out of place leaves after batch merge, so
  // in the event of an overflow just store max which we then just go count the
  // density as usual, which is cheap since it is stored out of place and the
  // size is just written
  [[no_unique_address]]
  typename std::conditional<store_density, uint16_t *, empty_type>::type
      density_array = {};

  [[nodiscard]] uint8_t *byte_array() const { return (uint8_t *)data_array; }

#if VQSORT == 1
  hwy::Sorter sorter;
#endif

  T &index_to_head(uint64_t index) const {
    if constexpr (head_form == InPlace) {
      return data_array[index * elts_per_leaf()];
    }
    // linear order
    if constexpr (head_form == Linear) {
      return head_array[index];
    }
    //
    // Eytzinger order
    if constexpr (head_form == Eytzinger) {
      return head_array[e_index(index, total_leaves())];
    }
    // BNary order
    if constexpr (head_form == BNary) {
      uint64_t in = bnary_index<B_size>(index, total_leaves_rounded_up());
      return head_array[in];
    }
  }

  T *index_to_data(uint64_t index) const {
    if constexpr (head_form == InPlace) {
      return &data_array[index * elts_per_leaf()] + 1;
    } else {
      return &data_array[index * elts_per_leaf()];
    }
  }

  // how big will the leaf be not counting the head
  [[nodiscard]] uint64_t leaf_size_in_bytes() const {
    if constexpr (head_form == InPlace) {
      return logN() - sizeof(T);
    } else {
      return logN();
    }
  }

  [[nodiscard]] uint64_t head_array_size() const {
    if constexpr (head_form == InPlace) {
      return get_size();
    }
    // linear order
    if constexpr (head_form == Linear) {
      return total_leaves() * sizeof(T);
    }
    // make next power of 2
    if constexpr (head_form == Eytzinger) {
      if (nextPowerOf2(total_leaves()) > total_leaves()) {
        uint64_t space =
            ((nextPowerOf2(total_leaves()) - 1) + total_leaves() + 1) / 2;
        return space * sizeof(T);
      }
      return ((total_leaves() * 2) - 1) * sizeof(T);
    }
    // BNary order
    if constexpr (head_form == BNary) {
      uint64_t size = B_size;
      while (size <= total_leaves()) {
        size *= B_size;
      }
      uint64_t check_size =
          ((size / B_size + total_leaves() + B_size) / B_size) * B_size;

      return std::min(size, check_size) * sizeof(T);
    }
  }
  [[nodiscard]] uint64_t calculate_num_leaves_rounded_up() const {
    static_assert(head_form == Eytzinger || head_form == BNary,
                  "you should only be rounding the head array size of you are "
                  "in either Eytzinger or BNary form");
    // Eytzinger and Bnary sometimes need to know the rounded number of leaves
    // linear order
    // make next power of 2
    if constexpr (head_form == Eytzinger) {
      if (nextPowerOf2(total_leaves()) > total_leaves()) {
        return (nextPowerOf2(total_leaves()) - 1);
      }
      return ((total_leaves() * 2) - 1);
    }
    // BNary order
    if constexpr (head_form == BNary) {
      uint64_t size = B_size;
      while (size <= total_leaves()) {
        size *= B_size;
      }
      return size;
    }
  }

  uint64_t n;

  uint64_t count_elements = 0;

  [[nodiscard]] std::pair<float, float> density_bound(uint64_t depth) const;

  uint8_t loglogN_;
  [[nodiscard]] uint8_t loglogN() const { return loglogN_; }
  uint32_t logN_;
  [[nodiscard]] uint32_t logN() const { return logN_; }
  uint64_t mask_for_leaf_;
  [[nodiscard]] uint64_t mask_for_leaf() const { return mask_for_leaf_; }
  uint32_t H_;
  [[nodiscard]] uint32_t H() const { return H_; }
  uint64_t total_leaves_;
  [[nodiscard]] uint64_t total_leaves() const { return total_leaves_; }

  [[no_unique_address]]
  typename std::conditional<head_form == InPlace || head_form == Linear,
                            empty_type, uint64_t>::type
      total_leaves_rounded_up_;
  [[nodiscard]] uint64_t total_leaves_rounded_up() const {
    static_assert(head_form == Eytzinger || head_form == BNary,
                  "you should only be rounding the head array size of you are "
                  "in either Eytzinger or BNary form");
    return total_leaves_rounded_up_;
  }
  [[nodiscard]] uint64_t elts_per_leaf() const { return logN() / sizeof(T); }

  bool has_0 = false;

  std::array<std::pair<float, float>, sizeof(T) * 8> density_bound_;

  [[nodiscard]] float lower_density_bound(uint64_t depth) const {
    ASSERT(depth < 10000000,
           "depth shouldn't be higher than log(n) it is %lu\n", depth);
    return density_bound_[depth].first;
  }
  [[nodiscard]] float upper_density_bound(uint64_t depth) const {
    ASSERT(density_bound_[depth].second <= density_limit(),
           "density_bound_[%lu].second = %f > density_limit() = %f\n", depth,
           density_bound_[depth].second, density_limit());
    // making sure we don't pass in a negative number
    ASSERT(depth < 100000000UL, "depth = %lu\n", depth);
    return density_bound_[depth].second;
  }

  [[nodiscard]] double density_limit() const {
    // we need enough space on both sides regardless of how elements are split
    return static_cast<double>(logN() - (3 * leaf::max_element_size)) / logN();
  }

  void grow_list(double factor = 2.0);
  void shrink_list(double factor = 2.0);

  [[nodiscard]] uint64_t get_density_count(uint64_t index, uint64_t len) const;
  [[nodiscard]] uint64_t get_density_count_no_overflow(uint64_t index,
                                                       uint64_t len) const;
  bool check_leaf_heads(uint64_t start_idx = 0,
                        uint64_t end_idx = std::numeric_limits<T>::max());
  [[nodiscard]] uint64_t get_depth(uint64_t len) const {
    return bsr_long(N() / len);
  }

  [[nodiscard]] uint64_t find_leaf(uint64_t index) const {
    return index & mask_for_leaf();
  }
  [[nodiscard]] uint64_t find_node(uint64_t index, uint64_t len) const {
    return (index / len) * len;
  }

  [[nodiscard]] uint64_t find_containing_leaf_index(
      T e, uint64_t start = 0,
      uint64_t end = std::numeric_limits<T>::max()) const;

  [[nodiscard]] uint64_t find_containing_leaf_index_debug(
      T e, uint64_t start = 0,
      uint64_t end = std::numeric_limits<T>::max()) const;

  template <class F>
  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute(
      const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>
          &leaves_to_check,
      uint64_t num_elts_merged, F bounds_check) const;

  template <class F>
  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute_serial(
      const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>
          &leaves_to_check,
      uint64_t num_elts_merged, F bounds_check) const;

  [[nodiscard]] uint64_t get_ranges_to_redistibute_lookup_sibling_count(
      const std::vector<ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
          &ranges_check,
      uint64_t start, uint64_t length, uint64_t level,
      uint64_t depth = 0) const;
  [[nodiscard]] uint64_t get_ranges_to_redistibute_lookup_sibling_count_serial(
      const std::vector<std::unordered_map<uint64_t, uint64_t>> &ranges_check,
      uint64_t start, uint64_t length, uint64_t level) const;

  [[nodiscard]] std::pair<std::vector<std::tuple<uint64_t, uint64_t>>,
                          std::optional<uint64_t>>
  get_ranges_to_redistibute_debug(
      const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>
          &leaves_to_check,
      uint64_t num_elts_merged) const;

  [[nodiscard]] std::map<uint64_t, std::pair<uint64_t, uint64_t>>
  get_ranges_to_redistibute_internal(std::pair<uint64_t, uint64_t> *begin,
                                     std::pair<uint64_t, uint64_t> *end) const;

  [[nodiscard]] uint64_t sum_serial(uint64_t start, uint64_t end) const;
  [[nodiscard]] uint64_t sum_parallel(uint64_t start, uint64_t end) const;
  void print_array_region(uint64_t start_leaf, uint64_t end_leaf) const;

public:
  // TODO(AUTHOR) make private
  bool check_nothing_full();
  static constexpr bool compressed = leaf::compressed;
  explicit CPMA();
  CPMA(const CPMA &source);
  ~CPMA() {
    if constexpr (head_form != InPlace) {
      free(head_array);
    }
    free(data_array);
    if constexpr (store_density) {
      free(density_array);
    }
  }
  void print_pma() const;
  void print_array() const;
  bool has(T e) const;
  bool insert(T e);
  uint64_t insert_batch(T *e, uint64_t batch_size);
  uint64_t remove_batch(T *e, uint64_t batch_size);
  // split num is the index of which partition you are
  uint64_t
  insert_batch_internal(T *e, uint64_t batch_size,
                        std::vector<std::pair<uint64_t, uint64_t>> &queue,
                        uint64_t start_leaf_idx, uint64_t end_leaf_idx);
  uint64_t
  remove_batch_internal(T *e, uint64_t batch_size,
                        std::vector<std::pair<uint64_t, uint64_t>> &queue,
                        uint64_t start_leaf_idx, uint64_t end_leaf_idx);

  bool remove(T e);
  [[nodiscard]] uint64_t get_size() const;
  [[nodiscard]] uint64_t get_size_no_allocator() const;

  [[nodiscard]] uint64_t get_element_count() const { return count_elements; }

  [[nodiscard]] uint64_t N() const { return n; }
  [[nodiscard]] uint64_t sum() const;
  [[nodiscard]] uint64_t sum_serial() const;
  [[nodiscard]] T max() const;
  [[nodiscard]] uint32_t num_nodes() const;
  [[nodiscard]] uint64_t get_head_structure_size() const {
    return head_array_size();
  }

  template <bool no_early_exit, class F> bool map(F f) const;

  template <bool no_early_exit, class F>
  void serial_map(F f, T start, T end, uint64_t start_hint = 0,
                  uint64_t end_hint = std::numeric_limits<T>::max()) const;

  template <bool no_early_exit, class F>
  void serial_map_with_hint(F f, T end,
                            const std::pair<uint8_t *, T> &hint) const;

  template <bool no_early_exit, class F>
  void serial_map_with_hint_par(F f, T end, const std::pair<uint8_t *, T> &hint,
                                const std::pair<uint8_t *, T> &end_hint) const;

  // returns a vector
  // for each node in the graph has a pointer to the start and the most recent
  // element so we know what the difference is from if we are starting from a
  // head, returns nullptr and the number leaf we are starting with
  std::vector<std::pair<uint8_t *, T>> getExtraData(bool skip = false) const {
    if (skip) {
      return {};
    }

    auto *hints = (std::pair<uint8_t *, uint64_t> *)::operator new(
        sizeof(std::pair<uint8_t *, uint64_t>) * (num_nodes() + 1));
    ParallelTools::parallel_for(0, num_nodes(), 512, [&](uint64_t i_) {
      uint64_t end = std::min(i_ + 512, (uint64_t)num_nodes());
      uint64_t i = i_;
      uint64_t start_hint = find_containing_leaf_index(i << 32U);
      uint64_t end_hint = find_containing_leaf_index(end << 32U, start_hint) +
                          (elts_per_leaf());
      if (i < end) {
        uint64_t leaf_idx =
            find_containing_leaf_index(i << 32U, start_hint, end_hint);
        // the element in question is the head of a leaf
        if (index_to_head(leaf_idx / elts_per_leaf()) >= (i << 32U)) {
          hints[i] = {nullptr, leaf_idx / elts_per_leaf()};
        } else {
          leaf l(index_to_head(leaf_idx / elts_per_leaf()),
                 index_to_data(leaf_idx / elts_per_leaf()),
                 leaf_size_in_bytes());
          // TODO(AUTHOR) be carefull here with new head structure,
          // find_loc-and_difference doesn't deal with heads
          hints[i] = l.find_loc_and_difference(i << 32U);
        }
        i += 1;
        for (; i < end; i++) {
          if (leaf_idx > 0 && leaf_idx + elts_per_leaf() < N() / sizeof(T) &&
              index_to_head((leaf_idx + elts_per_leaf()) / elts_per_leaf()) >
                  (i << 32U)) {
            // in the same leaf as the previous one so use the hint to speed
            // up the search
            // TODO(AUTHOR) be carefull here with new head structure,
            // find_loc-and_difference doesn't deal with heads
            if (hints[i - 1].first != nullptr) {
              hints[i] = leaf::find_loc_and_difference_with_hint(
                  i << 32U, hints[i - 1].first, hints[i - 1].second);
            } else {
              // the last hint was just the leaf head so we do a normal search
              // in the same leaf as the last element
              leaf l(index_to_head(leaf_idx / elts_per_leaf()),
                     index_to_data(leaf_idx / elts_per_leaf()),
                     leaf_size_in_bytes());
              hints[i] = l.find_loc_and_difference(i << 32U);
            }
            // printf("thought it was %p, and %lu\n", hints[i].first,
            //        hints[i].second);

          } else {
            leaf_idx =
                find_containing_leaf_index(i << 32U, start_hint, end_hint);
            // the element in question is the head of a leaf
            if (index_to_head(leaf_idx / elts_per_leaf()) >= (i << 32U)) {
              hints[i] = {nullptr, leaf_idx / elts_per_leaf()};
            } else {
              leaf l(index_to_head(leaf_idx / elts_per_leaf()),
                     index_to_data(leaf_idx / elts_per_leaf()),
                     leaf_size_in_bytes());
              // TODO(AUTHOR) be carefull here with new head structure,
              // find_loc-and_difference doesn't deal with heads
              hints[i] = l.find_loc_and_difference(i << 32U);
            }
            // printf("was %p, and %lu\n", hints[i].first, hints[i].second);
          }
        }
      }
    });
    hints[num_nodes()] = {byte_array() + N(), std::numeric_limits<T>::max()};
    std::vector<std::pair<uint8_t *, T>> ret;
    wrapArrayInVector(hints, num_nodes() + 1, ret);
    return ret;
  }

  template <class F>
  void map_neighbors(
      uint64_t i, F f,
      [[maybe_unused]] const std::vector<std::pair<uint8_t *, T>> &hints,
      bool parallel) const {

    if (parallel) {
      serial_map_with_hint_par<F::no_early_exit>(
          [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
          (i + 1) << 32U, hints[i], hints[i + 1]);
    } else {
      serial_map_with_hint<F::no_early_exit>(
          [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
          (i + 1) << 32U, hints[i]);
    }
  }
  template <class F>
  void map_range(
      F f, uint64_t start_node, uint64_t end_node,
      [[maybe_unused]] const std::vector<std::pair<uint8_t *, T>> &d) const {
    serial_map<true>(
        [&](uint64_t el) { return f(el >> 32UL, el & 0xFFFFFFFFUL); },
        start_node << 32U, end_node << 32U);
  }
};

// when adjusting the list size, make sure you're still in the
// density bound

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
std::pair<float, float>
CPMA<leaf, head_form, B_size, store_density>::density_bound(
    uint64_t depth) const {
  std::pair<double, double> pair;

  // between 1/4 and 1/2
  // pair.x = 1.0/2.0 - (( .25*depth)/list->H);
  // between 1/8 and 1/4
  if (H() == 0) {
    pair.first = std::max(((double)sizeof(T)) / logN(), 1.0 / 4.0);
    pair.second = 1.0 / 2.0;
    if (pair.second >= density_limit()) {
      pair.second = density_limit() - .001;
    }
    assert(pair.first < pair.second);
    return pair;
  }
  pair.first = std::max(((double)sizeof(T)) / logN(),
                        1.0 / 4.0 - ((.125 * depth) / H()));

  pair.second = 3.0 / 4.0 + (((1.0 / 4.0) * depth) / H());

  // // TODO(AUTHOR) not sure why I need this
  // if (H() < 12) {
  //   pair.second = 3.0 / 4.0 + (((1.0 / 4.0) * depth) / H());
  // } else {
  //   pair.second = 15.0 / 16.0 + (((1.0 / 16.0) * depth) / H());
  // }

  if (pair.second >= density_limit()) {
    pair.second = density_limit() - .001;
  }
  return pair;
}

// doubles the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
void CPMA<leaf, head_form, B_size, store_density>::grow_list(double factor) {
  // printf("grow list\n");
  // print_pma();
  // static_timer merge_timer("merge_in_double");
  // merge_timer.start();
  std::pair<leaf, uint64_t> merged_data =
      leaf::template merge<head_form == InPlace, store_density>(
          data_array, total_leaves(), logN(), 0,
          [this](uint64_t index) -> T & { return index_to_head(index); },
          density_array);
  // merge_timer.stop();

  /*
  printf("\nMERGED DATA IN GROW LIST\n");
  merged_data.first.print();
  */

  uint64_t min_new_size = n + logN();
  uint64_t desired_new_size = n * factor;
  if (desired_new_size < min_new_size) {
    n = min_new_size;
  } else {
    n = desired_new_size;
  }

  loglogN_ = bsr_long(bsr_long(N()));
  logN_ = leaf_blow_up_factor * (1U << loglogN());

  while (n % logN() != 0) {
    n += logN() - (n % logN());
    loglogN_ = bsr_long(bsr_long(N()));
    logN_ = leaf_blow_up_factor * (1U << loglogN());
  }
  mask_for_leaf_ = ~(logN() - 1U);

  H_ = bsr_long(total_leaves());
  total_leaves_ = N() / logN();
  if constexpr (head_form == Eytzinger || head_form == BNary) {
    total_leaves_rounded_up_ = calculate_num_leaves_rounded_up();
  }

  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
    assert(density_bound_[i].second <= density_limit());
  }

  free(data_array);
  // steal an extra few bytes to ensure we never read off the end
  data_array = (T *)aligned_alloc(32, N() + 32);
  if constexpr (head_form != InPlace) {
    free(head_array);
    head_array = (T *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(T)), 0);
  }

  if constexpr (store_density) {
    free(density_array);
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }

  // static_timer split_timer("split_in_double");
  // split_timer.start();
  merged_data.first.template split<head_form == InPlace, store_density>(
      total_leaves(), merged_data.second, logN(), data_array, 0,
      [this](uint64_t index) -> T & { return index_to_head(index); },
      density_array);
  // split_timer.stop();
  // -1 is since in this case the head is just being stored before the data
  free(reinterpret_cast<uint8_t *>(merged_data.first.array) - sizeof(T));
  for (uint64_t i = 0; i < N(); i += logN()) {
    ASSERT(get_density_count(i, logN()) <= logN() - leaf::max_element_size,
           "%lu > %lu\n tried to split %lu bytes into %lu leaves\n i = %lu\n",
           get_density_count(i, logN()), logN() - leaf::max_element_size,
           merged_data.second, total_leaves(), i / logN());
  }
}

// halves the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
void CPMA<leaf, head_form, B_size, store_density>::shrink_list(double factor) {
  if (n == min_leaf_size) {
    return;
  }
  std::pair<leaf, uint64_t> merged_data =
      leaf::template merge<head_form == InPlace, store_density>(
          data_array, total_leaves(), logN(), 0,
          [this](uint64_t index) -> T & { return index_to_head(index); },
          density_array);
  uint64_t max_new_size = n - logN();
  uint64_t desired_new_size = n / factor;
  if (desired_new_size > max_new_size) {
    n = max_new_size;
  } else {
    n = desired_new_size;
  }
  if (n < min_leaf_size) {
    n = min_leaf_size;
  }

  loglogN_ = bsr_long(bsr_long(N()));
  logN_ = leaf_blow_up_factor * (1U << loglogN());

  while (n % logN() != 0) {
    n += logN() - (n % logN());
    loglogN_ = bsr_long(bsr_long(N()));
    logN_ = leaf_blow_up_factor * (1U << loglogN());
  }
  total_leaves_ = N() / logN();
  if constexpr (head_form == Eytzinger || head_form == BNary) {
    total_leaves_rounded_up_ = calculate_num_leaves_rounded_up();
  }
  mask_for_leaf_ = ~(logN() - 1U);

  H_ = bsr_long(total_leaves());

  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
    assert(density_bound_[i].second <= density_limit());
  }

  free(data_array);
  data_array = (T *)aligned_alloc(32, N() + 32);
  if constexpr (head_form != InPlace) {
    free(head_array);
    head_array = (T *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(T)), 0);
  }
  if constexpr (store_density) {
    free(density_array);
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }

  merged_data.first.template split<head_form == InPlace, store_density>(
      total_leaves(), merged_data.second, logN(), data_array, 0,
      [this](uint64_t index) -> T & { return index_to_head(index); },
      density_array);
  // -1 is since in this case the head is just being stored before the data
  free(reinterpret_cast<uint8_t *>(merged_data.first.array) - sizeof(T));
}

// asumes the region is already locked
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::get_density_count(
    uint64_t byte_index, uint64_t len) const {

  uint64_t total = 0;
  int64_t num_leaves = len / logN();
  if constexpr (CILK == 1) {
    if (num_leaves > ParallelTools::getWorkers() * 1024) {
      ParallelTools::Reducer_sum<uint64_t> total_red;
      ParallelTools::parallel_for(0, num_leaves, [&](int64_t i) {
        if constexpr (store_density) {
          uint64_t val = density_array[byte_index / logN() + i];
          if (val == std::numeric_limits<uint16_t>::max()) {
            leaf l(index_to_head(byte_index / logN() + i),
                   index_to_data(byte_index / logN() + i),
                   leaf_size_in_bytes());
            val = l.template used_size<head_form == InPlace>();
          }
          total_red.add(val);
        } else {
          leaf l(index_to_head(byte_index / logN() + i),
                 index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
          total_red.add(l.template used_size<head_form == InPlace>());
        }
      });
      return total_red.get();
    }
  }
  for (int64_t i = 0; i < num_leaves; i++) {
    if constexpr (store_density) {
      uint64_t val = density_array[byte_index / logN() + i];
      if (val == std::numeric_limits<uint16_t>::max()) {
        leaf l(index_to_head(byte_index / logN() + i),
               index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
        val = l.template used_size<head_form == InPlace>();
      }
#if DEBUG == 1
      leaf l(index_to_head(byte_index / logN() + i),
             index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
      ASSERT(val == l.template used_size<head_form == InPlace>(),
             "got %lu, expected %lu, leaf_number = %lu\n", val,
             l.template used_size<head_form == InPlace>(),
             byte_index / logN() + i);
#endif
      total += val;
    } else {
      leaf l(index_to_head(byte_index / logN() + i),
             index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
      total += l.template used_size<head_form == InPlace>();
    }
  }
  return total;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t
CPMA<leaf, head_form, B_size, store_density>::get_density_count_no_overflow(
    uint64_t byte_index, uint64_t len) const {

  uint64_t total = 0;
  int64_t num_leaves = len / logN();
  if constexpr (CILK == 1) {
    if (num_leaves > ParallelTools::getWorkers() * 1024) {
      ParallelTools::Reducer_sum<uint64_t> total_red;
      ParallelTools::parallel_for(0, num_leaves, [&](int64_t i) {
        if constexpr (store_density) {
          total_red.add(density_array[byte_index / logN() + i]);
        } else {
          leaf l(index_to_head(byte_index / logN() + i),
                 index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
          total_red.add(
              l.template used_size_no_overflow<head_form == InPlace>());
        }
      });
      return total_red.get();
    }
  }
  for (int64_t i = 0; i < num_leaves; i++) {
    if constexpr (store_density) {
#if DEBUG == 1
      leaf l(index_to_head(byte_index / logN() + i),
             index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
      ASSERT(density_array[byte_index / logN() + i] ==
                 l.template used_size_no_overflow<head_form == InPlace>(),
             "got %lu, expected %lu\n", density_array[byte_index / logN() + i],
             l.template used_size_no_overflow<head_form == InPlace>());
#endif
      total += density_array[byte_index / logN() + i];
    } else {
      leaf l(index_to_head(byte_index / logN() + i),
             index_to_data(byte_index / logN() + i), leaf_size_in_bytes());
      total += l.template used_size_no_overflow<head_form == InPlace>();
    }
  }
  return total;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t
CPMA<leaf, head_form, B_size, store_density>::find_containing_leaf_index_debug(
    T e, uint64_t start, uint64_t end) const {

  if (N() == logN()) {
    return 0;
  }
  if (end > N() / sizeof(T)) {
    end = N() / sizeof(T);
  }
  assert((start * sizeof(T)) % logN() == 0);
  uint64_t size = (end - start) / elts_per_leaf();
  uint64_t logstep = bsr_long(size);
  uint64_t first_step = (size - (1UL << logstep));

  uint64_t step = (1UL << logstep);
  uint64_t idx = start / elts_per_leaf();
  assert(index_to_head(idx + first_step) != 0);
  idx = (index_to_head(idx + first_step) <= e) ? idx + first_step : idx;
  static constexpr uint64_t linear_cutoff = 128;
  while (step > linear_cutoff) {
    step >>= 1U;
    idx = (index_to_head(idx + step) <= e) ? idx + step : idx;
  }
  uint64_t end_linear = std::min(linear_cutoff, step);
  for (uint64_t i = 1; i < end_linear; i++) {
    if (index_to_head(idx + i) > e) {
      return (idx + i - 1) * (elts_per_leaf());
    }
  }
  return (idx + end_linear - 1) * (elts_per_leaf());
}

// searches in the unlocked array and only looks at leaf heads
// start is in PMA index, start the binary search after that point
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t
CPMA<leaf, head_form, B_size, store_density>::find_containing_leaf_index(
    T e, uint64_t start, uint64_t end) const {

  if (N() == logN()) {
    return 0;
  }
  if constexpr (head_form == Eytzinger) {
    if (start == 0 && end == std::numeric_limits<T>::max()) {
      uint64_t length = total_leaves_rounded_up();
      uint64_t value_to_check = length / 2;
      uint64_t length_to_add = length / 4 + 1;
      uint64_t e_index = 0;
      // printf("looking for %lu\n", (uint64_t)e);
      while (length_to_add > 0) {
        // printf("e_index = %lu, value_to_check = %lu, length_to_add = %lu, "
        //        "head_array[e_index] = %lu\n",
        //        e_index, value_to_check, length_to_add,
        //        (uint64_t)head_array[e_index]);
        if (head_array[e_index] == e) {
          ASSERT(value_to_check * elts_per_leaf() ==
                     find_containing_leaf_index_debug(e, start, end),
                 "got %lu, expected %lu", value_to_check * elts_per_leaf(),
                 find_containing_leaf_index_debug(e, start, end));
          __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
          return value_to_check * elts_per_leaf();
        }
        if (e < head_array[e_index] || head_array[e_index] == 0) {
          e_index = 2 * e_index + 1;
          value_to_check -= length_to_add;
        } else {
          e_index = 2 * e_index + 2;
          value_to_check += length_to_add;
        }
        length_to_add /= 2;
      }
      // printf("after: e_index = %lu, value_to_check = %lu, length_to_add =
      // %lu, "
      //        "head_array[e_index] = %lu\n",
      //        e_index, value_to_check, length_to_add,
      //        (uint64_t)head_array[e_index]);
      if (value_to_check >= total_leaves()) {
        value_to_check = total_leaves() - 1;
      }
      if (e < head_array[e_index] && value_to_check > 0) {
        value_to_check -= 1;
      }
      ASSERT(value_to_check * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu\n", value_to_check * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end));
      __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
      return value_to_check * elts_per_leaf();
    } else {
      uint64_t length = total_leaves_rounded_up();
      uint64_t value_to_check = length / 2;
      uint64_t length_to_add = length / 4 + 1;
      uint64_t e_index = 0;
      // printf("looking for %lu, start = %lu, end = %lu\n", (uint64_t)e, start,
      //        end);
      while (length_to_add > 0) {
        // printf("e_index = %lu, value_to_check = %lu, length_to_add = %lu\n",
        //        e_index, value_to_check, length_to_add);

        // if we are outside the searching range before start then we are a
        // right child
        if (value_to_check * elts_per_leaf() < start) {
          e_index = 2 * e_index + 2;
          value_to_check += length_to_add;
          length_to_add /= 2;
          continue;
        }
        if (value_to_check * elts_per_leaf() >= end) {
          e_index = 2 * e_index + 1;
          value_to_check -= length_to_add;
          length_to_add /= 2;
          continue;
        }
        // printf("head_array[e_index] = %lu\n", (uint64_t)head_array[e_index]);

        if (head_array[e_index] == e) {
          ASSERT(value_to_check * elts_per_leaf() ==
                     find_containing_leaf_index_debug(e, start, end),
                 "got %lu, expected %lu\n", value_to_check * elts_per_leaf(),
                 find_containing_leaf_index_debug(e, start, end));
          __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
          return value_to_check * elts_per_leaf();
        }
        if (e < head_array[e_index] || head_array[e_index] == 0) {
          e_index = 2 * e_index + 1;
          value_to_check -= length_to_add;
        } else {
          e_index = 2 * e_index + 2;
          value_to_check += length_to_add;
        }
        length_to_add /= 2;
      }
      // printf("after: e_index = %lu, value_to_check = %lu, length_to_add =
      // %lu, "
      //        "head_array[e_index] = %lu\n",
      //        e_index, value_to_check, length_to_add,
      //        (uint64_t)head_array[e_index]);
      if (value_to_check >= total_leaves()) {
        value_to_check = total_leaves() - 1;
      }
      if (value_to_check * elts_per_leaf() >= end) {
        value_to_check = (end / elts_per_leaf()) - 1;
      } else if (value_to_check * elts_per_leaf() <= start) {
        value_to_check = start / elts_per_leaf();
      } else if (e < head_array[e_index] && value_to_check > 0) {
        value_to_check -= 1;
      }
      ASSERT(value_to_check * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu, elts_per_leaf = %lu, start = %lu, end = "
             "%lu\n",
             value_to_check * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end), elts_per_leaf(),
             start, end);
      __builtin_prefetch(&data_array[value_to_check * (elts_per_leaf())]);
      return value_to_check * elts_per_leaf();
    }
  }
  if constexpr (head_form == BNary) {
    if (start == 0 && end == std::numeric_limits<T>::max()) {

      // print_pma();
      // std::cout << "search for " << e << "\n";

      uint64_t block_number = 0;
      uint64_t leaf_index = 0;
      uint64_t amount_to_add = total_leaves_rounded_up() / B_size;
      while (amount_to_add >= 1) {
        uint64_t number_in_block_greater_item = 0;
#ifdef __AVX2NO__
        if constexpr ((B_size - 1) % 4 == 0 && sizeof(T) == 8) {
          const __m256i e_vec = _mm256_set1_epi64x(e);
          const __m256i zero_vec = _mm256_setzero_si256();
          if constexpr (B_size - 1 == 4) {
            __m256i data = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1)]);
            __m256i equal_zero = _mm256_cmpeq_epi64(data, zero_vec);
            __m256i greater = _mm256_cmpgt_epi64(data, e_vec);
            __m256i cells_to_count = _mm256_or_si256(equal_zero, greater);
            number_in_block_greater_item +=
                __builtin_popcount(_mm256_movemask_epi8(cells_to_count)) / 8;
          }
          if constexpr (B_size - 1 == 8) {
            __m256i data1 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1)]);
            __m256i data2 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 4]);
            __m256i equal_zero1 = _mm256_cmpeq_epi64(data1, zero_vec);
            __m256i equal_zero2 = _mm256_cmpeq_epi64(data2, zero_vec);
            __m256i greater1 = _mm256_cmpgt_epi64(data1, e_vec);
            __m256i greater2 = _mm256_cmpgt_epi64(data2, e_vec);
            __m256i cells_to_count1 = _mm256_or_si256(equal_zero1, greater1);
            __m256i cells_to_count2 = _mm256_or_si256(equal_zero2, greater2);
            __m256i cells_to_count =
                _mm256_blend_epi32(cells_to_count1, cells_to_count2, 0x55);
            number_in_block_greater_item +=
                __builtin_popcount(_mm256_movemask_epi8(cells_to_count)) / 4;
          }
          if constexpr (B_size - 1 == 16) {
            __m256i data1 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1)]);
            __m256i data2 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 4]);
            __m256i data3 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 8]);
            __m256i data4 = _mm256_loadu_si256(
                (__m256i *)&head_array[block_number * (B_size - 1) + 12]);
            __m256i equal_zero1 = _mm256_cmpeq_epi64(data1, zero_vec);
            __m256i equal_zero2 = _mm256_cmpeq_epi64(data2, zero_vec);
            __m256i equal_zero3 = _mm256_cmpeq_epi64(data3, zero_vec);
            __m256i equal_zero4 = _mm256_cmpeq_epi64(data4, zero_vec);
            __m256i greater1 = _mm256_cmpgt_epi64(data1, e_vec);
            __m256i greater2 = _mm256_cmpgt_epi64(data2, e_vec);
            __m256i greater3 = _mm256_cmpgt_epi64(data3, e_vec);
            __m256i greater4 = _mm256_cmpgt_epi64(data4, e_vec);
            __m256i cells_to_count1 = _mm256_or_si256(equal_zero1, greater1);
            __m256i cells_to_count2 = _mm256_or_si256(equal_zero2, greater2);
            __m256i cells_to_count3 = _mm256_or_si256(equal_zero3, greater3);
            __m256i cells_to_count4 = _mm256_or_si256(equal_zero4, greater4);
            __m256i cells_to_counta =
                _mm256_blend_epi32(cells_to_count1, cells_to_count2, 0x55);
            __m256i cells_to_countb =
                _mm256_blend_epi32(cells_to_count3, cells_to_count4, 0x55);
            __m256i cells_to_count =
                _mm256_blend_epi16(cells_to_counta, cells_to_countb, 0x55);
            number_in_block_greater_item +=
                __builtin_popcount(_mm256_movemask_epi8(cells_to_count)) / 2;
          }
        } else
#endif
        {
          for (uint64_t i = 0; i < B_size - 1; i++) {
            number_in_block_greater_item +=
                head_array[block_number * (B_size - 1) + i] > e;
          }

          for (uint64_t i = 0; i < B_size - 1; i++) {
            number_in_block_greater_item +=
                head_array[block_number * (B_size - 1) + i] == 0;
          }
        }
        uint64_t child_number = B_size - number_in_block_greater_item - 1;
        leaf_index += amount_to_add * child_number;
        amount_to_add /= B_size;
        block_number = block_number * B_size + child_number + 1;
      }

      if (leaf_index > 0 && index_to_head(leaf_index) > e) {
        leaf_index -= 1;
      }

      if (leaf_index >= total_leaves()) {
        leaf_index = total_leaves() - 1;
      }

      ASSERT(leaf_index * elts_per_leaf() ==
                 find_containing_leaf_index_debug(e, start, end),
             "got %lu, expected %lu\n", leaf_index * elts_per_leaf(),
             find_containing_leaf_index_debug(e, start, end));
      __builtin_prefetch(&data_array[leaf_index * (elts_per_leaf())]);
      return leaf_index * elts_per_leaf();
    }

    // print_array_region(start / elts_per_leaf(), end / elts_per_leaf());
    // std::cout << "search for " << e << " start = " << start / elts_per_leaf()
    //           << " end = " << end / elts_per_leaf()
    //           << " total leaves = " << total_leaves() << "\n";

    uint64_t block_number = 0;
    uint64_t leaf_index = 0;
    uint64_t amount_to_add = total_leaves_rounded_up() / B_size;
    while (amount_to_add >= 1) {
      uint64_t number_in_block_greater_item = 0;
      for (uint64_t i = 0; i < B_size - 1; i++) {
        // printf("%lu\n", leaf_index + (amount_to_add * (i + 1)) - 1);
        if (leaf_index + (amount_to_add * (i + 1)) - 1 <
            start / elts_per_leaf()) {
          continue;
        } else if (leaf_index + (amount_to_add * (i + 1)) - 1 >=
                   end / elts_per_leaf()) {
          number_in_block_greater_item += 1;
        } else {
          ASSERT(head_array[block_number * (B_size - 1) + i] != 0,
                 "looked at a zero entry, shouldn't have happened, "
                 "block_number = %lu, i = %lu, amount_to_add = %lu, start = "
                 "%lu, end = %lu, leaf_index = %lu\n",
                 block_number, i, amount_to_add, start, end, leaf_index);
          number_in_block_greater_item +=
              head_array[block_number * (B_size - 1) + i] > e;
        }
      }
      uint64_t child_number = B_size - number_in_block_greater_item - 1;
      leaf_index += amount_to_add * child_number;
      amount_to_add /= B_size;
      block_number = block_number * B_size + child_number + 1;
    }
    bool in_range = true;
    if (leaf_index < start / elts_per_leaf()) {
      leaf_index = start / elts_per_leaf();
      in_range = false;
    }

    if (leaf_index >= end / elts_per_leaf()) {
      leaf_index = (end / elts_per_leaf()) - 1;
      in_range = false;
    }

    if (in_range && leaf_index > 0 && leaf_index > start / elts_per_leaf() &&
        index_to_head(leaf_index) > e) {
      leaf_index -= 1;
    }

    if (leaf_index >= total_leaves()) {
      leaf_index = total_leaves() - 1;
    }

    ASSERT(leaf_index * elts_per_leaf() ==
               find_containing_leaf_index_debug(e, start, end),
           "got %lu, expected %lu, start = "
           "%lu, end = %lu, leaf_index = %lu, elts_per_leaf = %lu, "
           "total_leaves = %lu\n",
           leaf_index * elts_per_leaf(),
           find_containing_leaf_index_debug(e, start, end), start, end,
           leaf_index, elts_per_leaf(), total_leaves());
    __builtin_prefetch(&data_array[leaf_index * (elts_per_leaf())]);
    return leaf_index * elts_per_leaf();
  }
  assert(end > start);
  // print_pma();
  if (end > N() / sizeof(T)) {
    end = N() / sizeof(T);
  }
  assert((start * sizeof(T)) % logN() == 0);
  uint64_t size = (end - start) / elts_per_leaf();
  uint64_t logstep = bsr_long(size);
  uint64_t first_step = (size - (1UL << logstep));

  uint64_t step = (1UL << logstep);
  uint64_t idx = start / elts_per_leaf();
  assert(index_to_head(idx + first_step) != 0);
  idx = (index_to_head(idx + first_step) <= e) ? idx + first_step : idx;
  static constexpr uint64_t linear_cutoff = (head_form == InPlace) ? 1 : 128;
  while (step > linear_cutoff) {
    step >>= 1U;
    assert(idx < total_leaves());
    assert(index_to_head(idx + step) != 0);
    idx = (index_to_head(idx + step) <= e) ? idx + step : idx;
  }
  uint64_t end_linear = std::min(linear_cutoff, step);
  for (uint64_t i = 1; i < end_linear; i++) {
    if (index_to_head(idx + i) > e) {
      __builtin_prefetch(&data_array[(idx + i - 1) * (elts_per_leaf())]);
      return (idx + i - 1) * (elts_per_leaf());
    }
  }
  __builtin_prefetch(&data_array[(idx + end_linear - 1) * (elts_per_leaf())]);
  return (idx + end_linear - 1) * (elts_per_leaf());
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
void CPMA<leaf, head_form, B_size, store_density>::print_array_region(
    uint64_t start_leaf, uint64_t end_leaf) const {
  for (uint64_t i = start_leaf; i < end_leaf; i++) {
    leaf l(index_to_head(i), index_to_data(i), leaf_size_in_bytes());
    printf("LEAF NUMBER %lu, STARTING IDX %lu, BYTE IDX %lu", i,
           i * elts_per_leaf(), i * logN());
    if constexpr (store_density) {
      printf(", density_array says: %d", +density_array[i]);
    }
    printf("\n");
    l.print();
  }
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
void CPMA<leaf, head_form, B_size, store_density>::print_array() const {
  print_array_region(0, total_leaves());
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
void CPMA<leaf, head_form, B_size, store_density>::print_pma() const {
  printf("N = %lu, logN = %u, loglogN = %u, H = %u\n", N(), logN(), loglogN(),
         H());
  printf("count_elements %lu\n", count_elements);
  if (has_0) {
    printf("has 0\n");
  }
  if (count_elements) {
    print_array();
  } else {
    printf("The PMA is empty\n");
  }
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
CPMA<leaf, head_form, B_size, store_density>::CPMA()
    : n(min_leaf_size), loglogN_(bsr_long(bsr_long(N()))),
      logN_(leaf_blow_up_factor * (1U << loglogN())) {
  assert(logN() <= n);
  assert(logN() >= min_leaf_size);

  while (n % logN() != 0) {
    n += logN() - (n % logN());
    loglogN_ = bsr_long(bsr_long(N()));
    logN_ = leaf_blow_up_factor * (1U << loglogN());
  }
  total_leaves_ = N() / logN();
  if constexpr (head_form == Eytzinger || head_form == BNary) {
    total_leaves_rounded_up_ = calculate_num_leaves_rounded_up();
  }
  mask_for_leaf_ = ~(logN() - 1U);

  H_ = bsr_long(total_leaves());

  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
    assert(density_bound_[i].second <= density_limit());
  }
  data_array = (T *)aligned_alloc(32, N() + 32);
  std::fill(byte_array(), byte_array() + N(), 0);
  if constexpr (head_form != InPlace) {
    head_array = (T *)malloc(head_array_size());
    std::fill(head_array, head_array + (head_array_size() / sizeof(T)), 0);
  }
  if constexpr (store_density) {
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
  }
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
CPMA<leaf, head_form, B_size, store_density>::CPMA(
    const CPMA<leaf, head_form, B_size, store_density> &source)
    : n(source.n), count_elements(source.count_elements),
      loglogN_(source.loglogN_), logN_(source.logN_),
      mask_for_leaf_(source.mask_for_leaf_), H_(source.H_), has_0(source.has_0),
      total_leaves_(source.total_leaves_),
      total_leaves_rounded_up_(source.total_leaves_rounded_up_) {

  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
    assert(density_bound_[i].second <= density_limit());
  }
  data_array = (T *)aligned_alloc(32, N() + 32);
  std::copy(source.data_array, &source.data_array[N() / sizeof(T)], data_array);
  if constexpr (head_form != InPlace) {
    head_array = (T *)malloc(head_array_size());
    std::copy(source.head_array,
              &source.head_array[head_array_size() / sizeof(T)], head_array);
  }
  if constexpr (store_density) {
    density_array = (uint16_t *)malloc(total_leaves() * sizeof(uint16_t));
    std::copy(source.density_array, &source.density_array[total_leaves()],
              density_array);
  }
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
bool CPMA<leaf, head_form, B_size, store_density>::has(T e) const {
  if (count_elements == 0) {
    return false;
  }
  if (e == 0) {
    return has_0;
  }
  uint64_t leaf_start = find_containing_leaf_index(e);
  // std::cout << "has(" << e << ") leaf_start = " << leaf_start << std::endl;
  leaf l(index_to_head(leaf_start / elts_per_leaf()),
         index_to_data(leaf_start / elts_per_leaf()), leaf_size_in_bytes());
  return l.contains(e);
}

// input: ***sorted*** batch, number of elts in a batch
// return true if the element was inserted, false if it was already there
// return number of things inserted (not already there)
// can only merge at least one leaf at a time (not multiple threads on the
// same leaf) - can merge multiple leaves at a time optional arguments in
// terms of leaf idx
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
bool CPMA<leaf, head_form, B_size, store_density>::check_leaf_heads(
    uint64_t start_idx, uint64_t end_idx) {
  if (get_element_count() == 0) {
    return true;
  }
  uint64_t end = std::min(end_idx * sizeof(T), N());
  for (uint64_t idx = start_idx * sizeof(T); idx < end; idx += logN()) {
    if (index_to_head(idx / logN()) == 0) {
      printf("\n\nLEAF %lu HEAD IS 0\n", idx / logN());
      printf("idx is %lu, byte_idx = %lu, logN = %u, N = %lu, count_elements = "
             "%lu\n",
             idx / sizeof(T), idx, logN(), N(), count_elements);
      // print_pma();
      return false;
    }
  }
  return true;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
bool CPMA<leaf, head_form, B_size, store_density>::check_nothing_full() {
  for (uint64_t i = 0; i < N(); i += logN()) {
    if constexpr (compressed) {
      if (get_density_count(i, logN()) >= logN() - leaf::max_element_size) {
        printf("%lu >= %lu, i = %lu\n", get_density_count(i, logN()),
               logN() - leaf::max_element_size, i / logN());
        return false;
      }
    } else {
      if (get_density_count(i, logN()) > logN() - leaf::max_element_size) {
        printf("%lu >= %lu, i = %lu\n", get_density_count(i, logN()),
               logN() - leaf::max_element_size, i / logN());
        return false;
      }
    }
  }
  return true;
}

template <typename leaf, typename T>
bool everything_from_batch_added(leaf l, T *start, T *end) {
  bool have_everything = true;
  if (end - start < 10000000) {
    ParallelTools::parallel_for(0, end - start, [&](size_t i) {
      if (!l.debug_contains(start[i])) {
        std::cout << "missing " << start[i] << std::endl;
        // l.print();
        have_everything = false;
      }
    });
  }
  return have_everything;
}

// mark all leaves that exceed their density bound
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::insert_batch_internal(
    T *e, uint64_t batch_size,
    std::vector<std::pair<uint64_t, uint64_t>> &queue, uint64_t start_leaf_idx,
    uint64_t end_leaf_idx) {
  std::vector<std::pair<uint64_t, uint64_t>> local_queue;
  // current_elt_ptr is ptr into the batch
  T *current_elt_ptr = e;
  uint64_t num_elts_merged = 0;

  uint64_t prev_leaf_start = start_leaf_idx;
  while (current_elt_ptr < e + batch_size) {
    // print_pma();
    assert(check_leaf_heads(prev_leaf_start, end_leaf_idx));
    // find the leaf that the next batch element goes into
    // NOTE: so we are guaranteed that a merge will always take at least one
    // thing from the batch

    // uint64_t leaf_idx = prev_leaf_start;
    // uint64_t i;
    // bool found = false;
    // for (i = leaf_idx + elts_per_leaf();
    //      i < std::min(end_leaf_idx, prev_leaf_start + 50);
    //      i += elts_per_leaf()) {
    //   if (index_to_head(i / elts_per_leaf()) > *current_elt_ptr) {
    //     leaf_idx = i - elts_per_leaf();
    //     found = true;
    //   }
    // }
    // if (!found) {
    uint64_t leaf_idx = find_containing_leaf_index(
        *current_elt_ptr, prev_leaf_start, end_leaf_idx);
    // }
    assert(leaf_idx % (elts_per_leaf()) == 0);
    prev_leaf_start = leaf_idx + elts_per_leaf();
    assert(prev_leaf_start % (elts_per_leaf()) == 0);

    // merge as much of this batch as you can
    // N(), logN() are in terms of bytes
    // find_containing_leaf_index gives you in terms of elt idx
    T next_head = std::numeric_limits<T>::max(); // max int
    if (leaf_idx + elts_per_leaf() < end_leaf_idx) {
      next_head = index_to_head((leaf_idx / elts_per_leaf()) + 1);
    }

    // merge into leaf returns the pointer in the batch
    // takes in pointer to start of merge in batch, remaining size in batch,
    // idx of start leaf in merge, and head of the next leaf

    // returns pointer to new start in batch, number of (distinct) elements
    // merged in
    leaf l(index_to_head(leaf_idx / elts_per_leaf()),
           index_to_data(leaf_idx / elts_per_leaf()), leaf_size_in_bytes());
    auto result = l.template merge_into_leaf<head_form == InPlace>(
        current_elt_ptr, e + batch_size, next_head);

    assert(
        everything_from_batch_added(l, current_elt_ptr, std::get<0>(result)));
    current_elt_ptr = std::get<0>(result);
    // number of elements merged is the number of distinct elts merged into
    // this leaf
    num_elts_merged += std::get<1>(result);
    // number of bytes used in this leaf (if exceeds logN(), merge_into_leaf
    // will have written some auxiliary memory
    auto bytes_used = std::get<2>(result);
    if (std::get<1>(result)) {
      if constexpr (store_density) {
        density_array[leaf_idx / elts_per_leaf()] = std::min(
            bytes_used,
            static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()));
      }
      ASSERT(bytes_used == get_density_count(leaf_idx * sizeof(T), logN()),
             "got %lu, expected %lu\n", bytes_used,
             get_density_count(leaf_idx * sizeof(T), logN()));
      // if exceeded leaf density bound, add self to per-worker queue for
      // leaves to rebalance
      if (bytes_used > logN() * upper_density_bound(H())) {
        local_queue.push_back({leaf_idx, bytes_used});
      }
    }
  }
  queue = std::move(local_queue);
  return num_elts_merged;
}

// mark all leaves that exceed their density bound
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::remove_batch_internal(
    T *e, uint64_t batch_size,
    std::vector<std::pair<uint64_t, uint64_t>> &queue, uint64_t start_leaf_idx,
    uint64_t end_leaf_idx) {
  // current_elt_ptr is ptr into the batch
  T *current_elt_ptr = e;
  uint64_t num_elts_removed = 0;

  uint64_t prev_leaf_start = start_leaf_idx;
  while (current_elt_ptr < e + batch_size && prev_leaf_start < end_leaf_idx) {
    // print_pma();
    assert(get_element_count() == 0 ||
           check_leaf_heads(prev_leaf_start, end_leaf_idx));
    // find the leaf that the next batch element goes into
    // NOTE: so we are guaranteed that a merge will always take at least one
    // thing from the batch
    // std::cout << REPORT3(*current_elt_ptr, prev_leaf_start, end_leaf_idx)
    //           << std::endl;

    uint64_t leaf_idx = find_containing_leaf_index(
        *current_elt_ptr, prev_leaf_start, end_leaf_idx);
    assert(leaf_idx % (elts_per_leaf()) == 0);
    prev_leaf_start = leaf_idx + elts_per_leaf();
    assert(prev_leaf_start % (elts_per_leaf()) == 0);

    // merge as much of this batch as you can
    // N(), logN() are in terms of bytes
    // find_containing_leaf_index gives you in terms of elt idx
    T next_head = std::numeric_limits<T>::max(); // max int
    if (leaf_idx + elts_per_leaf() < end_leaf_idx) {
      next_head = index_to_head((leaf_idx / elts_per_leaf()) + 1);
    }

    // merge into leaf returns the pointer in the batch
    // takes in pointer to start of merge in batch, remaining size in batch,
    // idx of start leaf in merge, and head of the next leaf

    // returns pointer to new start in batch, number of (distinct) elements
    // merged in
    leaf l(index_to_head(leaf_idx / elts_per_leaf()),
           index_to_data(leaf_idx / elts_per_leaf()), leaf_size_in_bytes());
    auto result = l.template strip_from_leaf<head_form == InPlace>(
        current_elt_ptr, e + batch_size, next_head);
    assert(l.head < next_head);
    // l.print();
    current_elt_ptr = std::get<0>(result);
    // number of elements merged is the number of distinct elts merged into
    // this leaf
    num_elts_removed += std::get<1>(result);
    // number of bytes used in this leaf (if exceeds logN(), merge_into_leaf
    // will have written some auxiliary memory
    auto bytes_used = std::get<2>(result);

    // if exceeded leaf density bound, add self to per-worker queue for leaves
    // to rebalance
    if (std::get<1>(result)) {
      if constexpr (store_density) {
        density_array[leaf_idx / elts_per_leaf()] = bytes_used;
      }
      ASSERT(bytes_used == get_density_count(leaf_idx * sizeof(T), logN()),
             "got %lu, expected %lu, removed %lu elements, used_size = %lu\n",
             bytes_used, get_density_count(leaf_idx * sizeof(T), logN()),
             std::get<1>(result), l.template used_size<head_form == InPlace>());
      if (bytes_used < logN() * lower_density_bound(H()) || bytes_used == 0) {
        queue.push_back({leaf_idx, bytes_used});
      }
    }
  }
  return num_elts_removed;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
std::map<uint64_t, std::pair<uint64_t, uint64_t>>
CPMA<leaf, head_form, B_size, store_density>::
    get_ranges_to_redistibute_internal(
        std::pair<uint64_t, uint64_t> *begin,
        std::pair<uint64_t, uint64_t> *end) const {
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> ranges_to_redistribute_2;

  for (auto it = begin; it < end; ++it) {
    auto p = *it;
    // printf("LEAF TO REDISTRIBUTE: [%lu, %lu)\n", p.first, p.second);
    uint64_t current_bytes_filled = p.second;
    uint64_t leaf_idx = p.first; // pma index
    uint64_t index_len = elts_per_leaf();

    uint64_t parent = find_node(leaf_idx, (elts_per_leaf()));

    double current_density = (double)current_bytes_filled / logN();
    uint64_t byte_len = logN();
    uint64_t level = H();
    // while exceeding density bound
    while (current_density > upper_density_bound(level)) {
      byte_len *= 2;
      index_len = byte_len / sizeof(T);
      // printf("byte len %lu, index len %lu\n", byte_len, index_len);

      // start, end in index
      uint64_t old_parent = parent;
      parent = find_node(leaf_idx, index_len);
      // printf("parent = %lu, old_parent = %lu\n", parent, old_parent);
      bool left_child = parent == old_parent;

      // printf("parent %lu, index_len %lu\n", parent, index_len);
      // if the beginning of this range is already in the map, use it
      // TODO(AUTHOR) better map
      if (ranges_to_redistribute_2.contains(parent)) {
        if (ranges_to_redistribute_2[parent].first >= byte_len) {
          byte_len = ranges_to_redistribute_2[parent].first;
          current_bytes_filled = ranges_to_redistribute_2[parent].second;
          break;
        }
      }

      if (level > 0) {
        level--;
      }

      // if its the whole thing
      if (parent == 0 && byte_len >= N()) {
        current_bytes_filled = get_density_count(0, N());
        byte_len = N();
        break;
      }

      uint64_t end = parent + index_len;

      // if you go off the end, count this range and exit the while
      // number of leaves doesn't have to be a power of 2
      if (end > N() / sizeof(T)) {
        // printf("end %lu, N %lu, N idx %lu\n", end, N(), N() / sizeof(T));
        end = N() / sizeof(T);

        current_bytes_filled =
            get_density_count(parent * sizeof(T), (end - parent) * sizeof(T));
        current_density =
            (double)current_bytes_filled / ((end - parent) * sizeof(T));
        continue;
      }
      // current_bytes_filled =
      //     get_density_count(parent * sizeof(T), index_len *
      //     sizeof(T));
      // printf("%d, %lu, %lu, %lu, %lu, %lu\n", left_child, parent,
      //        current_bytes_filled,
      //        get_density_count(parent * sizeof(T), index_len * sizeof(T) /
      //        2), get_density_count((parent + (index_len / 2)) * sizeof(T),
      //                          index_len * sizeof(T) / 2),
      //        get_density_count(parent * sizeof(T), index_len * sizeof(T)));
      if (left_child) {
        current_bytes_filled += get_density_count(
            (parent + (index_len / 2)) * sizeof(T), index_len * sizeof(T) / 2);
      } else {
        current_bytes_filled +=
            get_density_count(parent * sizeof(T), index_len * sizeof(T) / 2);
      }
      // printf("%lu\n", current_bytes_filled);
      ASSERT(current_bytes_filled ==
                 get_density_count(parent * sizeof(T), index_len * sizeof(T)),
             "got %lu expected %lu\n", current_bytes_filled,
             get_density_count(parent * sizeof(T), index_len * sizeof(T)));

      current_density = (double)current_bytes_filled / byte_len;
    }

    if (parent + index_len > N() / sizeof(T)) {
      byte_len = N() - (parent * sizeof(T));
    }

    assert(current_bytes_filled <= density_limit() * byte_len ||
           byte_len >= N());
    ranges_to_redistribute_2[parent] = {byte_len, current_bytes_filled};
    if (parent == 0 && byte_len >= N()) {
      ranges_to_redistribute_2.clear();
      ranges_to_redistribute_2[0] = {N(), current_bytes_filled};

      break;
    }
  }
  return ranges_to_redistribute_2;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
[[nodiscard]] uint64_t CPMA<leaf, head_form, B_size, store_density>::
    get_ranges_to_redistibute_lookup_sibling_count(
        const std::vector<
            ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
            &ranges_check,
        uint64_t start, uint64_t length, uint64_t level, uint64_t depth) const {
  if (start * sizeof(T) >= N()) {
    // we are a sibling off the end
    return 0;
  }
  if (level == 0) {
    return get_density_count(start * sizeof(T), logN());
  }
  uint64_t value = ranges_check[level].unlocked_value(
      start, std::numeric_limits<uint64_t>::max());
  if (value == std::numeric_limits<uint64_t>::max()) {
    // look up the children
    if (level < 3) {
      if (start + length > N() / sizeof(T)) {
        length = (N() / sizeof(T)) - start;
      }
      return get_density_count(start * sizeof(T), length * sizeof(T));
    }
    if (depth <= 5) {

      uint64_t left = cilk_spawn get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start, length / 2, level - 1, depth + 1);
      uint64_t right = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start + length / 2, length / 2, level - 1, depth + 1);
      cilk_sync;
      return left + right;
    } else {
      uint64_t left = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start, length / 2, level - 1, depth + 1);
      uint64_t right = get_ranges_to_redistibute_lookup_sibling_count(
          ranges_check, start + length / 2, length / 2, level - 1, depth + 1);
      return left + right;
    }
  }
  return value;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
[[nodiscard]] uint64_t CPMA<leaf, head_form, B_size, store_density>::
    get_ranges_to_redistibute_lookup_sibling_count_serial(
        const std::vector<std::unordered_map<uint64_t, uint64_t>> &ranges_check,
        uint64_t start, uint64_t length, uint64_t level) const {
  if (start * sizeof(T) >= N()) {
    // we are a sibling off the end
    return 0;
  }
  if (level == 0) {
    return get_density_count(start * sizeof(T), logN());
  }
  auto it = ranges_check[level].find(start);
  if (it == ranges_check[level].end()) {
    // look up the children
    if (level < 3) {
      if (start + length > N() / sizeof(T)) {
        length = (N() / sizeof(T)) - start;
      }
      return get_density_count(start * sizeof(T), length * sizeof(T));
    }

    uint64_t left = get_ranges_to_redistibute_lookup_sibling_count_serial(
        ranges_check, start, length / 2, level - 1);
    uint64_t right = get_ranges_to_redistibute_lookup_sibling_count_serial(
        ranges_check, start + length / 2, length / 2, level - 1);
    return left + right;
  }
  return it->second;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<leaf, head_form, B_size, store_density>::get_ranges_to_redistibute_debug(
    const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>
        &leaves_to_check,
    uint64_t num_elts_merged) const {
  // post-process leaves_to_check into ranges
  // make one big list of leaves to redistribute

  std::vector<uint64_t> sizes;
  uint64_t total_size = 0;
  for (const auto &item : leaves_to_check) {
    sizes.push_back(total_size);
    total_size += item.size();
  }

  // ranges to redistribute
  // start -> (length to redistribute, bytes)
  // needs to be sorted for deduplication
  // TODO(AUTHOR) better map
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> ranges_to_redistribute_2;

  if (total_size * 2 >= total_leaves() ||
      num_elts_merged * 10 >= count_elements) {
    uint64_t current_bytes_filled = get_density_count(0, N());
    double current_density = (double)current_bytes_filled / N();
    if (current_density > upper_density_bound(0)) {
      return {{}, current_bytes_filled};
    }
  }
  if (ranges_to_redistribute_2.empty()) {
    // copy into big list
    std::vector<std::pair<uint64_t, uint64_t>> leaves_to_redistribute(
        total_size);
    ParallelTools::parallel_for(0, leaves_to_check.size(), [&](uint64_t i) {
      std::copy(leaves_to_check[i].begin(), leaves_to_check[i].end(),
                leaves_to_redistribute.begin() + sizes[i]);
      // leaves_to_check[i].clear();
    });
    ranges_to_redistribute_2 = get_ranges_to_redistibute_internal(
        leaves_to_redistribute.data(),
        leaves_to_redistribute.data() + leaves_to_redistribute.size());
  }

  // deduplicate ranges_to_redistribute_2 into ranges_to_redistribute_3
  std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;

  for (auto const &[key, value] : ranges_to_redistribute_2) {
    if (ranges_to_redistribute_3.empty()) {
      if (value.first == N() && value.second > upper_density_bound(0) * N()) {
        return {{}, value.second};
      }
      ranges_to_redistribute_3.emplace_back(key, value.first);
    } else {
      const auto &last =
          ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
      auto end_of_last_range =
          std::get<0>(last) + (std::get<1>(last) / sizeof(T));
      if (key >= end_of_last_range) {
        if (value.first == N() && value.second > upper_density_bound(0) * N()) {
          return {{}, value.second};
        }
        ranges_to_redistribute_3.emplace_back(key, value.first);
      }
    }
  }
  return {ranges_to_redistribute_3, {}};
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
template <class F>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<leaf, head_form, B_size, store_density>::get_ranges_to_redistibute_serial(
    const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>
        &leaves_to_check,
    uint64_t num_elts_merged, F bounds_check) const {
  // post-process leaves_to_check into ranges
  // make one big list of leaves to redistribute

  std::vector<uint64_t> sizes;
  uint64_t total_size = 0;
  for (const auto &item : leaves_to_check) {
    sizes.push_back(total_size);
    total_size += item.size();
  }

  if (total_size * 2 >= total_leaves() ||
      num_elts_merged * 10 >= count_elements) {
    uint64_t current_bytes_filled = get_density_count(0, N());
    double current_density = (double)current_bytes_filled / N();
    if (bounds_check(0, current_density)) {
      return {{}, current_bytes_filled};
    }
  }

  // ranges to redistribute
  // start -> (length to redistribute)
  // needs to be sorted for deduplication
  std::map<uint64_t, uint64_t> ranges_to_redistribute_2;
  uint64_t full_opt = std::numeric_limits<uint64_t>::max();
  // (height) -> (start) -> (bytes_used)
  std::vector<std::unordered_map<uint64_t, uint64_t>> ranges_check(2);

  uint64_t length_in_index = 2 * (elts_per_leaf());
  {
    uint64_t level = 1;
    uint64_t child_length_in_index = length_in_index / 2;

    for (const auto &leaves : leaves_to_check) {
      for (const auto &[child_range_start, child_byte_count] : leaves) {
        uint64_t parent_range_start =
            find_node(child_range_start, length_in_index);
        uint64_t length_in_index_local = length_in_index;
        if (parent_range_start + length_in_index > N() / sizeof(T)) {
          length_in_index_local = (N() / sizeof(T)) - parent_range_start;
        }
        bool left_child = parent_range_start == child_range_start;

        // get sibling byte count
        uint64_t sibling_range_start =
            (left_child) ? parent_range_start + child_length_in_index
                         : parent_range_start;
        uint64_t sibling_byte_count =
            get_ranges_to_redistibute_lookup_sibling_count_serial(
                ranges_check, sibling_range_start, child_length_in_index,
                level - 1);
        uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
        double density =
            ((double)parent_byte_count) / (length_in_index_local * sizeof(T));
        // printf("level = %lu, start = %lu, length = %lu, density = %f, "
        //        "density_limit = %f, N() = %lu\n",
        //        level, parent_range_start, length_in_index_local, density,
        //        upper_density_bound(H() - level), N() / sizeof(T));
        if (length_in_index_local >= N() / sizeof(T) ||
            !bounds_check(H() - level, density)) {
          if (length_in_index_local == N() / sizeof(T) &&
              bounds_check(0, density)) {
            full_opt = parent_byte_count;
          }

          ranges_to_redistribute_2[parent_range_start] =
              length_in_index_local * sizeof(T);

        } else {
          ranges_check[level][parent_range_start] = parent_byte_count;
        }
      }
    }
  }

  for (uint64_t level = 2; level <= get_depth(logN()) + 1; level++) {
    if (ranges_check[level - 1].empty()) {
      break;
    }
    length_in_index *= 2;
    uint64_t child_length_in_index = length_in_index / 2;

    ranges_check.emplace_back();

    uint64_t level_for_density = H() - level;
    if (level > H()) {
      level_for_density = 0;
    }
    // double density_bound_for_level =
    // upper_density_bound(level_for_density);

    for (const auto &p : ranges_check[level - 1]) {

      uint64_t child_range_start = p.first;
      uint64_t parent_range_start =
          find_node(child_range_start, length_in_index);
      uint64_t length_in_index_local = length_in_index;
      if (parent_range_start + length_in_index > N() / sizeof(T)) {
        length_in_index_local = (N() / sizeof(T)) - parent_range_start;
      }
      bool left_child = parent_range_start == child_range_start;
      uint64_t child_byte_count = p.second;

      // get sibling byte count
      uint64_t sibling_range_start =
          (left_child) ? parent_range_start + child_length_in_index
                       : parent_range_start;
      uint64_t sibling_byte_count =
          get_ranges_to_redistibute_lookup_sibling_count_serial(
              ranges_check, sibling_range_start, child_length_in_index,
              level - 1);
      uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
      double density =
          ((double)parent_byte_count) / (length_in_index_local * sizeof(T));
      // printf("level = %lu, start = %lu, length = %lu, density = %f, "
      //        "density_limit = %f, N() = %lu\n",
      //        level, parent_range_start, length_in_index_local, density,
      //        upper_density_bound(H() - level), N() / sizeof(T));
      if (length_in_index_local >= N() / sizeof(T) ||
          !bounds_check(level_for_density, density)) {
        if (length_in_index_local == N() / sizeof(T) &&
            bounds_check(0, density)) {
          full_opt = parent_byte_count;
        }

        ranges_to_redistribute_2[parent_range_start] =
            length_in_index_local * sizeof(T);

      } else {
        ranges_check[level][parent_range_start] = parent_byte_count;
      }
    }
  }

  if (full_opt != std::numeric_limits<uint64_t>::max()) {
    return {{}, full_opt};
  }

  std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;
  for (auto const &[key, value] : ranges_to_redistribute_2) {
    if (ranges_to_redistribute_3.empty()) {
      ranges_to_redistribute_3.emplace_back(key, value);
    } else {
      const auto &last =
          ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
      auto end_of_last_range =
          std::get<0>(last) + (std::get<1>(last) / sizeof(T));
      if (key >= end_of_last_range) {
        ranges_to_redistribute_3.emplace_back(key, value);
      }
    }
  }

  return {ranges_to_redistribute_3, {}};
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
template <class F>
std::pair<std::vector<std::tuple<uint64_t, uint64_t>>, std::optional<uint64_t>>
CPMA<leaf, head_form, B_size, store_density>::get_ranges_to_redistibute(
    const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>
        &leaves_to_check,
    uint64_t num_elts_merged, F bounds_check) const {
  // post-process leaves_to_check into ranges
  // make one big list of leaves to redistribute

  std::vector<uint64_t> sizes;
  uint64_t total_size = 0;
  for (const auto &item : leaves_to_check) {
    sizes.push_back(total_size);
    total_size += item.size();
  }

  if (total_size * 2 >= total_leaves() ||
      num_elts_merged * 10 >= count_elements) {
    uint64_t current_bytes_filled = get_density_count(0, N());
    double current_density = (double)current_bytes_filled / N();
    if (bounds_check(0, current_density)) {
      return {{}, current_bytes_filled};
    }
  }

  // leaves_to_check[i].clear();

  if ((ParallelTools::getWorkers() == 1 ||
       total_size < ParallelTools::getWorkers() * 100U)) {
    /*
    static_timer timer_7("timer_7");
    timer_7.start();

    // copy into big list
    std::vector<std::pair<uint64_t, uint64_t>> leaves_to_redistribute(
        total_size);

    p_for(uint64_t i = 0; i < leaves_to_check.size(); i++) {
      std::copy(leaves_to_check[i].begin(), leaves_to_check[i].end(),
                leaves_to_redistribute.begin() + sizes[i]);
    }
    // ranges to redistribute
    // start -> (length to redistribute, bytes)
    // needs to be sorted for deduplication
    std::map<uint64_t, std::pair<uint64_t, uint64_t>>
    ranges_to_redistribute_2;

    ranges_to_redistribute_2 = get_ranges_to_redistibute_internal(
        leaves_to_redistribute.data(),
        leaves_to_redistribute.data() + leaves_to_redistribute.size());
    std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;

    for (auto const &[key, value] : ranges_to_redistribute_2) {
      if (ranges_to_redistribute_3.empty()) {
        if (value.first == N() && value.second > upper_density_bound(0) * N())
    { return {{}, value.second};
        }
        ranges_to_redistribute_3.emplace_back(key, value.first);
      } else {
        const auto &last =
            ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
        auto end_of_last_range =
            std::get<0>(last) + (std::get<1>(last) / sizeof(T));
        if (key >= end_of_last_range) {
          if (value.first == N() &&
              value.second > upper_density_bound(0) * N()) {
            return {{}, value.second};
          }
          ranges_to_redistribute_3.emplace_back(key, value.first);
        }
      }
    }
    timer_7.stop();
    return {ranges_to_redistribute_3, {}};
    */

    // ranges to redistribute
    // start -> (length to redistribute)
    // needs to be sorted for deduplication
    std::map<uint64_t, uint64_t> ranges_to_redistribute_2;
    uint64_t full_opt = std::numeric_limits<uint64_t>::max();
    // (height) -> (start) -> (bytes_used)
    std::vector<std::unordered_map<uint64_t, uint64_t>> ranges_check(2);

    uint64_t length_in_index = 2 * (elts_per_leaf());
    {
      uint64_t level = 1;
      uint64_t child_length_in_index = length_in_index / 2;

      for (const auto &leaves : leaves_to_check) {
        for (const auto &[child_range_start, child_byte_count] : leaves) {
          uint64_t parent_range_start =
              find_node(child_range_start, length_in_index);
          uint64_t length_in_index_local = length_in_index;
          if (parent_range_start + length_in_index > N() / sizeof(T)) {
            length_in_index_local = (N() / sizeof(T)) - parent_range_start;
          }
          bool left_child = parent_range_start == child_range_start;

          // get sibling byte count
          uint64_t sibling_range_start =
              (left_child) ? parent_range_start + child_length_in_index
                           : parent_range_start;
          uint64_t sibling_byte_count =
              get_ranges_to_redistibute_lookup_sibling_count_serial(
                  ranges_check, sibling_range_start, child_length_in_index,
                  level - 1);
          uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
          double density =
              ((double)parent_byte_count) / (length_in_index_local * sizeof(T));
          // printf("level = %lu, start = %lu, length = %lu, density = %f, "
          //        "density_limit = %f, N() = %lu\n",
          //        level, parent_range_start, length_in_index_local, density,
          //        upper_density_bound(H() - level), N() / sizeof(T));
          if (length_in_index_local >= N() / sizeof(T) ||
              !bounds_check(H() - level, density)) {
            if (length_in_index_local == N() / sizeof(T) &&
                bounds_check(0, density)) {
              full_opt = parent_byte_count;
            }

            ranges_to_redistribute_2[parent_range_start] =
                length_in_index_local * sizeof(T);

          } else {
            ranges_check[level][parent_range_start] = parent_byte_count;
          }
        }
      }
    }

    for (uint64_t level = 2; level <= get_depth(logN()) + 1; level++) {
      if (ranges_check[level - 1].empty()) {
        break;
      }
      length_in_index *= 2;
      uint64_t child_length_in_index = length_in_index / 2;

      ranges_check.emplace_back();

      uint64_t level_for_density = H() - level;
      if (level > H()) {
        level_for_density = 0;
      }
      // double density_bound_for_level =
      // upper_density_bound(level_for_density);

      for (const auto &p : ranges_check[level - 1]) {

        uint64_t child_range_start = p.first;
        uint64_t parent_range_start =
            find_node(child_range_start, length_in_index);
        uint64_t length_in_index_local = length_in_index;
        if (parent_range_start + length_in_index > N() / sizeof(T)) {
          length_in_index_local = (N() / sizeof(T)) - parent_range_start;
        }
        bool left_child = parent_range_start == child_range_start;
        uint64_t child_byte_count = p.second;

        // get sibling byte count
        uint64_t sibling_range_start =
            (left_child) ? parent_range_start + child_length_in_index
                         : parent_range_start;
        uint64_t sibling_byte_count =
            get_ranges_to_redistibute_lookup_sibling_count_serial(
                ranges_check, sibling_range_start, child_length_in_index,
                level - 1);
        uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
        double density =
            ((double)parent_byte_count) / (length_in_index_local * sizeof(T));
        // printf("level = %lu, start = %lu, length = %lu, density = %f, "
        //        "density_limit = %f, N() = %lu\n",
        //        level, parent_range_start, length_in_index_local, density,
        //        upper_density_bound(H() - level), N() / sizeof(T));
        if (length_in_index_local >= N() / sizeof(T) ||
            !bounds_check(level_for_density, density)) {
          if (length_in_index_local == N() / sizeof(T) &&
              bounds_check(0, density)) {
            full_opt = parent_byte_count;
          }

          ranges_to_redistribute_2[parent_range_start] =
              length_in_index_local * sizeof(T);

        } else {
          ranges_check[level][parent_range_start] = parent_byte_count;
        }
      }
    }

    if (full_opt != std::numeric_limits<uint64_t>::max()) {
      return {{}, full_opt};
    }

    std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;
    for (auto const &[key, value] : ranges_to_redistribute_2) {
      if (ranges_to_redistribute_3.empty()) {
        ranges_to_redistribute_3.emplace_back(key, value);
      } else {
        const auto &last =
            ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
        auto end_of_last_range =
            std::get<0>(last) + (std::get<1>(last) / sizeof(T));
        if (key >= end_of_last_range) {
          ranges_to_redistribute_3.emplace_back(key, value);
        }
      }
    }

    return {ranges_to_redistribute_3, {}};

  } else {
    // ranges to redistribute
    // start -> (length to redistribute)
    // needs to be sorted for deduplication
    ParallelTools::concurrent_hash_map<uint64_t, uint64_t>
        ranges_to_redistribute_2;
    std::atomic<uint64_t> full_opt = std::numeric_limits<uint64_t>::max();
    // timer timer_1("timer_1");
    // timer timer_2("timer_2");
    // timer timer_3("timer_3");
    // timer timer_4("timer_4");
    // timer timer_5("timer_5");
    // timer timer_6("timer_6");
    // timer_1.start();
    // (height) -> (start) -> (bytes_used)
    // timer_2.start();
    std::vector<ParallelTools::concurrent_hash_map<uint64_t, uint64_t>>
        ranges_check(2);

    // timer_2.stop();
    // timer_3.start();
    uint64_t length_in_index = 2 * (elts_per_leaf());
    {
      uint64_t level = 1;
      uint64_t child_length_in_index = length_in_index / 2;

      ParallelTools::parallel_for(0, leaves_to_check.size(), [&](uint64_t i) {
        for (const auto &[child_range_start, child_byte_count] :
             leaves_to_check[i]) {
          uint64_t parent_range_start =
              find_node(child_range_start, length_in_index);
          uint64_t length_in_index_local = length_in_index;
          if (parent_range_start + length_in_index > N() / sizeof(T)) {
            length_in_index_local = (N() / sizeof(T)) - parent_range_start;
          }
          bool left_child = parent_range_start == child_range_start;

          // get sibling byte count
          uint64_t sibling_range_start =
              (left_child) ? parent_range_start + child_length_in_index
                           : parent_range_start;
          uint64_t sibling_byte_count =
              get_ranges_to_redistibute_lookup_sibling_count(
                  ranges_check, sibling_range_start, child_length_in_index,
                  level - 1);
          uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
          double density =
              ((double)parent_byte_count) / (length_in_index_local * sizeof(T));
          // printf("level = %lu, start = %lu, length = %lu, density = %f, "
          //        "density_limit = %f, N() = %lu\n",
          //        level, parent_range_start, length_in_index_local, density,
          //        upper_density_bound(H() - level), N() / sizeof(T));
          if (length_in_index_local >= N() / sizeof(T) ||
              !bounds_check(H() - level, density)) {
            if (length_in_index_local == N() / sizeof(T) &&
                bounds_check(0, density)) {
              full_opt.store(parent_byte_count);
            }

            ranges_to_redistribute_2.insert_or_assign(
                parent_range_start, length_in_index_local * sizeof(T));
            // not theoretically true, but probably means something is wrong
            assert(length_in_index_local * sizeof(T) < (1UL << 60U));

          } else {
            ranges_check[level].insert_or_assign(parent_range_start,
                                                 parent_byte_count);
          }
        }
      });
    }
    // timer_3.stop();

    // timer_4.start();
    for (uint64_t level = 2; level <= get_depth(logN()) + 1; level++) {
      // auto seq = ranges_check[level - 1].entries();
      if (ranges_check[level - 1].unlocked_empty()) {
        break;
      }
      length_in_index *= 2;
      uint64_t child_length_in_index = length_in_index / 2;
      ranges_check.emplace_back();

      uint64_t level_for_density = H() - level;
      if (level > H()) {
        level_for_density = 0;
      }
      // double density_bound_for_level =
      // upper_density_bound(level_for_density);

      // timer_5.start();

      ranges_check[level - 1].for_each([&](uint64_t child_range_start,
                                           uint64_t child_byte_count) {
        uint64_t parent_range_start =
            find_node(child_range_start, length_in_index);
        uint64_t length_in_index_local = length_in_index;
        if (parent_range_start + length_in_index > N() / sizeof(T)) {
          length_in_index_local = (N() / sizeof(T)) - parent_range_start;
        }
        bool left_child = parent_range_start == child_range_start;

        // get sibling byte count
        uint64_t sibling_range_start =
            (left_child) ? parent_range_start + child_length_in_index
                         : parent_range_start;
        uint64_t sibling_byte_count =
            get_ranges_to_redistibute_lookup_sibling_count(
                ranges_check, sibling_range_start, child_length_in_index,
                level - 1);
        uint64_t parent_byte_count = child_byte_count + sibling_byte_count;
        double density =
            ((double)parent_byte_count) / (length_in_index_local * sizeof(T));
        // printf("level = %lu, start = %lu, length = %lu, density = %f, "
        //        "density_limit = %f, N() = %lu\n",
        //        level, parent_range_start, length_in_index_local, density,
        //        upper_density_bound(H() - level), N() / sizeof(T));
        if (length_in_index_local >= N() / sizeof(T) ||
            !bounds_check(level_for_density, density)) {
          if (length_in_index_local == N() / sizeof(T) &&
              bounds_check(0, density)) {
            full_opt.store(parent_byte_count);
          }

          ranges_to_redistribute_2.insert_or_assign(
              parent_range_start, length_in_index_local * sizeof(T));
          // not theoretically true, but probably means something is wrong
          assert(length_in_index_local * sizeof(T) < (1UL << 60U));

        } else {
          ranges_check[level].insert_or_assign(parent_range_start,
                                               parent_byte_count);
        }
      });
      // timer_5.stop();
    }

    // timer_4.stop();

    // timer_1.stop();
    if (full_opt.load() != std::numeric_limits<uint64_t>::max()) {
      return {{}, full_opt.load()};
    }

    std::vector<std::tuple<uint64_t, uint64_t>> ranges_to_redistribute_3;
    // timer_6.start();
    auto seq = ranges_to_redistribute_2.unlocked_entries();
    ParallelTools::sort(seq.begin(), seq.end());

    for (auto const &[key, value] : seq) {
      // not theoretically true, but probably means something is wrong
      assert(value < (1UL << 60U));
      if (ranges_to_redistribute_3.empty()) {
        ranges_to_redistribute_3.emplace_back(key, value);
      } else {
        const auto &last =
            ranges_to_redistribute_3[ranges_to_redistribute_3.size() - 1];
        auto end_of_last_range =
            std::get<0>(last) + (std::get<1>(last) / sizeof(T));
        if (key >= end_of_last_range) {
          ranges_to_redistribute_3.emplace_back(key, value);
        }
      }
    }
    // timer_6.stop();
    return {ranges_to_redistribute_3, {}};
  }
}

// input: batch, number of elts in a batch
// return true if the element was inserted, false if it was already there
// return number of things inserted (not already there)
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::insert_batch(
    T *e, uint64_t batch_size) {
  timer total_timer("insert_batch");
  total_timer.start();
  if (batch_size <= 100) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < batch_size; i++) {
      count += insert(e[i]);
    }
    return count;
  }

  // TODO(AUTHOR) make it work for the first batch
  if (get_element_count() == 0) {
    uint64_t count = 0;
    uint64_t end = std::min(batch_size, 1000UL);
    for (uint64_t i = 0; i < end; i++) {
      count += insert(e[i]);
    }
    if (batch_size == end) {
      total_timer.stop();
      return count;
    } else {
      e += end;
      batch_size -= end;
    }
  }

  assert(check_leaf_heads());

  timer sort_timer("sort");

  sort_timer.start();
#if CILK == 0
#if VQSORT == 0
  std::sort(e, e + batch_size);
#else
  if (batch_size * sizeof(T) < 8UL * 1024) {
    std::sort(e, e + batch_size);
  } else {
    sorter(e, batch_size, hwy::SortAscending());
  }
#endif
#else
  if (batch_size > 10000) {
    // TODO find out why this doesn't work
    if constexpr (false && sizeof(T) == 4) {
      ParallelTools::integerSort(e, batch_size);
    } else {
      std::vector<T> data_vector;
      wrapArrayInVector(e, batch_size, data_vector);
      parlay::integer_sort_inplace(data_vector);
      releaseVectorWrapper(data_vector);
    }
  } else {
    ParallelTools::sort(e, e + batch_size);
  }

#endif
  sort_timer.stop();

  // TODO(AUTHOR) currently only works for unsigned types
  while (*e == 0) {
    has_0 = true;
    e += 1;
    batch_size -= 1;
  }

  // total number of leaves
  uint64_t num_leaves = total_leaves();

  // leaves per partition
  uint64_t split_points =
      std::min({(uint64_t)num_leaves / 10, (uint64_t)batch_size / 100,
                (uint64_t)ParallelTools::getWorkers() * 10});
  split_points = std::max(split_points, 1UL);

  // which leaves were touched during the merge
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> leaves_to_check(
      split_points);

  uint64_t leaf_stride = num_leaves / split_points;

  uint64_t num_elts_merged = 0;

  std::vector<std::tuple<T, T, uint64_t, uint64_t>> leaf_bounds(split_points);

  // calculate split points before doing any batch insertions
  // TODO(AUTHOR) see why this can't be made parallel
  // right now it slows it down a ton
  ParallelTools::serial_for(0, split_points - 1, [&](uint64_t i) {
    uint64_t start_leaf_idx = i * leaf_stride * (elts_per_leaf());
    T start_elt = index_to_head(start_leaf_idx / elts_per_leaf());
    // printf("i %u: start elt %u, start leaf idx %u\n", i, start_elt,
    // start_leaf_idx);

    uint64_t end_leaf_idx = (i + 1) * leaf_stride * (elts_per_leaf());
    // head of leaf at start_leaf_idx
    T end_elt = index_to_head(end_leaf_idx / elts_per_leaf());

    assert(start_leaf_idx <= end_leaf_idx);
    leaf_bounds[i] = {start_elt, end_elt, start_leaf_idx, end_leaf_idx};
  });
  // last loop not in parallel due to weird compiler behavior
  {
    uint64_t i = split_points - 1;
    uint64_t start_leaf_idx = i * leaf_stride * (elts_per_leaf());
    T start_elt = index_to_head(start_leaf_idx / elts_per_leaf());
    // printf("i %u: start elt %u, start leaf idx %u\n", i, start_elt,
    // start_leaf_idx);
    T end_elt = std::numeric_limits<T>::max();

    uint64_t end_leaf_idx = N() / sizeof(T);
    assert(start_leaf_idx <= end_leaf_idx);
    leaf_bounds[i] = {start_elt, end_elt, start_leaf_idx, end_leaf_idx};
  }
  ParallelTools::Reducer_sum<uint64_t> num_elts_merged_reduce;

  timer merge_timer("merge_timer");
  merge_timer.start();
  ParallelTools::parallel_for(0, split_points, [&](uint64_t i) {
    auto bounds = leaf_bounds[i];
    T start_elt = std::get<0>(bounds);
    T end_elt = std::get<1>(bounds);
    uint64_t start_leaf_idx = std::get<2>(bounds);
    uint64_t end_leaf_idx = std::get<3>(bounds);

    // search for boundaries in batch
    T *batch_start = std::lower_bound(e, e + batch_size, start_elt);
    // if we are the first batch start at the begining
    if (i == 0) {
      batch_start = e;
    }
    T *batch_end = std::lower_bound(e, e + batch_size, end_elt);
    if (batch_start == batch_end || batch_start == e + batch_size) {
      return;
    }
    // number of elts we are merging
    uint64_t range_size = uint64_t(batch_end - batch_start);
    // do the merge
    num_elts_merged_reduce.add(
        insert_batch_internal(batch_start, range_size, leaves_to_check[i],
                              start_leaf_idx, end_leaf_idx));
  });
  num_elts_merged = num_elts_merged_reduce.get();
  merge_timer.stop();

  // std::cout << "###after merges, before redistibutes\n###";
  // print_pma();
  // std::cout << "######\n";

  // if most leaves need to be redistributed, or many elements were added,
  // just check the root first to hopefully save walking up the tree
  timer range_finder_timer("range_finder_timer");
  range_finder_timer.start();
  auto ranges_pair = get_ranges_to_redistibute(
      leaves_to_check, num_elts_merged, [&](uint64_t level, double density) {
        return density > upper_density_bound(level);
      });
  auto ranges_to_redistribute_3 = ranges_pair.first;
  auto full_opt = ranges_pair.second;

  range_finder_timer.stop();
#if DEBUG == 1
  auto [ranges_debug, full_opt_debug] =
      get_ranges_to_redistibute_debug(leaves_to_check, num_elts_merged);
  if (ranges_to_redistribute_3.size() != ranges_debug.size()) {
    printf("sizes don't match, got %lu, expected %lu\n",
           ranges_to_redistribute_3.size(), ranges_debug.size());
    printf("got:\n");
    for (const auto &element : ranges_to_redistribute_3) {
      std::cout << "( " << std::get<0>(element) << ", "
                << std::get<1>(element) / sizeof(T) << ") ";
    }
    std::cout << std::endl;
    printf("correct:\n");
    for (const auto &element : ranges_debug) {
      std::cout << "( " << std::get<0>(element) << ", "
                << std::get<1>(element) / sizeof(T) << ") ";
    }
    std::cout << std::endl;
  } else {
    for (size_t i = 0; i < ranges_to_redistribute_3.size(); i++) {
      if (ranges_to_redistribute_3[i] != ranges_debug[i]) {
        printf("element %lu doesn't match, got (%lu, %lu), expected (%lu,"
               "%lu)\n",
               i, std::get<0>(ranges_to_redistribute_3[i]),
               std::get<1>(ranges_to_redistribute_3[i]),
               std::get<0>(ranges_debug[i]), std::get<1>(ranges_debug[i]));
      }
    }
  }

  assert(ranges_to_redistribute_3 == ranges_debug);
  assert(full_opt == full_opt_debug);
#endif

  // doubling everything
  if (full_opt.has_value()) {
    timer double_timer("doubling");
    double_timer.start();

    uint64_t target_size = N();
    double growth = 1;
    auto bytes_occupied = full_opt.value();

    // min bytes necessary to meet the density bound
    uint64_t bytes_required = bytes_occupied / upper_density_bound(0);

    while (target_size <= bytes_required) {
      target_size *= growing_factor;
      growth *= growing_factor;
    }

    // printf("GROWING LIST by factor of %f, start size %lu, target size %u,
    // "
    //        "bytes_occupied %u, bytes required %u\n",
    //        growth, N(), target_size, bytes_occupied, bytes_required);
    // print_pma();
    grow_list(growth);
    double_timer.stop();
  } else { // not doubling
    // in parallel, redistribute ranges

    timer redistribute_timer("redistributing");
    redistribute_timer.start();
    ParallelTools::parallel_for(
        0, ranges_to_redistribute_3.size(), [&](uint64_t i) {
          auto start = std::get<0>(ranges_to_redistribute_3[i]);
          auto len = std::get<1>(ranges_to_redistribute_3[i]);
          // printf("REDISTRIBUTING RANGE: start %lu, len %lu, bytes %u\n",
          // key,
          //        value.first, value.second);
          // if constexpr (std::is_same_v<leaf, delta_compressed_leaf<
          //                                        typename
          //                                        leaf::value_type>>)
          //                                        {
          //   print_pma();
          // }

          std::pair<leaf, uint64_t> merged_data =
              leaf::template merge<head_form == InPlace, store_density>(
                  &(data_array[start]), len / logN(), logN(),
                  start / elts_per_leaf(),
                  [this](uint64_t index) -> T & {
                    return index_to_head(index);
                  },
                  density_array);

          if constexpr (std::is_same_v<leaf, delta_compressed_leaf<T>>) {
            // std::cout << std::get<2>(item) << std::endl;
            assert(((double)merged_data.second) / (len / logN()) <
                   (density_limit() * len));
          }

          // merged_data.first.print();
          // number of leaves, num elements in input leaf, num elements in
          // output leaf, dest region
          merged_data.first.template split<head_form == InPlace, store_density>(
              len / logN(), merged_data.second, logN(), &(data_array[start]),
              start / elts_per_leaf(),
              [this](uint64_t index) -> T & { return index_to_head(index); },
              density_array);
          // print_array_region(start / elts_per_leaf(),
          //                    start / elts_per_leaf() + (len / logN()));
          // printf("merged leaf\n");
          // merged_data.first.print();
          // printf("merged leaf done\n");
          free(reinterpret_cast<uint8_t *>(merged_data.first.array) -
               sizeof(T));
        });
    redistribute_timer.stop();
  }
  assert(check_nothing_full());
  // print_pma();

  assert(check_leaf_heads());
  count_elements += num_elts_merged;
  total_timer.stop();
  return num_elts_merged;
}

// input: batch, number of elts in a batch
// return number of things removed
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::remove_batch(
    T *e, uint64_t batch_size) {
  assert(check_leaf_heads());
  static_timer total_timer("remove_batch");
  total_timer.start();

  if (get_element_count() == 0 || batch_size == 0) {
    return 0;
  }
  if (batch_size <= 100) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < batch_size; i++) {
      count += remove(e[i]);
    }
    return count;
  }

  static_timer sort_timer("sort_remove_batch");

  sort_timer.start();
#if CILK == 0
#if VQSORT == 0
  std::sort(e, e + batch_size);
#else
  if (batch_size * sizeof(T) < 8UL * 1024) {
    std::sort(e, e + batch_size);
  } else {
    sorter(e, batch_size, hwy::SortAscending());
  }
#endif
#else
  if (batch_size > 10000) {
    // integerSort(e, batch_size);
    std::vector<T> data_vector;
    wrapArrayInVector(e, batch_size, data_vector);
    parlay::integer_sort_inplace(data_vector);
    releaseVectorWrapper(data_vector);
  } else {
    ParallelTools::sort(e, e + batch_size);
  }
#endif
  sort_timer.stop();

  // TODO(AUTHOR) currently only works for unsigned types
  while (*e == 0) {
    has_0 = false;
    e += 1;
    batch_size -= 1;
  }

  // total number of leaves
  uint64_t num_leaves = total_leaves();

  // leaves per partition
  uint64_t split_points =
      std::min({(uint64_t)num_leaves / 10, (uint64_t)batch_size / 100,
                (uint64_t)ParallelTools::getWorkers() * 10});
  split_points = std::max(split_points, 1UL);

  // which leaves were touched during the merge
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> leaves_to_check(
      split_points);
  int leaf_stride = num_leaves / split_points;

  uint64_t num_elts_removed = 0;

  std::vector<std::tuple<T, T, uint64_t, uint64_t>> leaf_bounds(split_points);

  // calculate split points before doing any batch insertions
  // TODO(AUTHOR) see why this can't be made parallel
  // right now it slows it down a ton
  ParallelTools::serial_for(0, split_points - 1, [&](uint64_t i) {
    uint64_t start_leaf_idx = i * leaf_stride * (elts_per_leaf());
    T start_elt = index_to_head(start_leaf_idx / elts_per_leaf());
    // printf("i %u: start elt %u, start leaf idx %u\n", i, start_elt,
    // start_leaf_idx);
    T end_elt = std::numeric_limits<T>::max();

    uint64_t end_leaf_idx = (i + 1) * leaf_stride * (elts_per_leaf());

    // head of leaf at start_leaf_idx
    end_elt = index_to_head(end_leaf_idx / elts_per_leaf());

    assert(start_leaf_idx <= end_leaf_idx);
    leaf_bounds[i] = {start_elt, end_elt, start_leaf_idx, end_leaf_idx};
  });
  {
    uint64_t i = split_points - 1;
    uint64_t start_leaf_idx = i * leaf_stride * (elts_per_leaf());
    T start_elt = index_to_head(start_leaf_idx / elts_per_leaf());
    // printf("i %u: start elt %u, start leaf idx %u\n", i, start_elt,
    // start_leaf_idx);
    T end_elt = std::numeric_limits<T>::max();

    uint64_t end_leaf_idx = N() / sizeof(T);
    assert(start_leaf_idx <= end_leaf_idx);
    leaf_bounds[i] = {start_elt, end_elt, start_leaf_idx, end_leaf_idx};
  }
  std::vector<uint64_t> num_elts_removed_vector(split_points);

  static_timer merge_timer("merge_timer_remove_batch");
  merge_timer.start();
  ParallelTools::parallel_for(0, split_points, [&](uint64_t i) {
    auto bounds = leaf_bounds[i];
    T start_elt = std::get<0>(bounds);
    T end_elt = std::get<1>(bounds);
    uint64_t start_leaf_idx = std::get<2>(bounds);
    uint64_t end_leaf_idx = std::get<3>(bounds);
    if (start_leaf_idx == end_leaf_idx) {
      return;
    }

    // search for boundaries in batch
    T *batch_start = std::lower_bound(e, e + batch_size, start_elt);
    // if we are the first batch start at the begining
    if (i == 0) {
      batch_start = e;
    }
    T *batch_end = std::lower_bound(e, e + batch_size, end_elt);

    if (batch_start == batch_end || batch_start == e + batch_size) {
      return;
    }

    // number of elts we are merging
    uint64_t range_size = uint64_t(batch_end - batch_start);

    // do the merge
    num_elts_removed_vector[i] +=
        remove_batch_internal(batch_start, range_size, leaves_to_check[i],
                              start_leaf_idx, end_leaf_idx);
  });
  for (auto el : num_elts_removed_vector) {
    num_elts_removed += el;
  }
  merge_timer.stop();

  auto ranges_pair = get_ranges_to_redistibute(
      leaves_to_check, num_elts_removed, [&](uint64_t level, double density) {
        return (density < lower_density_bound(level)) || (density == 0);
      });
  auto ranges_to_redistribute_3 = ranges_pair.first;
  assert(ranges_to_redistribute_3 ==
         get_ranges_to_redistibute_serial(
             leaves_to_check, num_elts_removed,
             [&](uint64_t level, double density) {
               return (density < lower_density_bound(level)) || (density == 0);
             })
             .first);
  auto full_opt = ranges_pair.second;

  // shrinking everything
  if (full_opt.has_value()) {
    static_timer shrinking_timer("shrinking");
    shrinking_timer.start();
    uint64_t target_size = N();
    double shrink = 1;
    auto bytes_occupied = full_opt.value();

    // min bytes necessary to meet the density bound
    uint64_t bytes_required = bytes_occupied / lower_density_bound(0);
    if (bytes_required == 0) {
      bytes_required = 1;
    }

    while (target_size >= bytes_required) {
      target_size /= growing_factor;
      shrink *= growing_factor;
    }

    // printf("SHRINKING LIST by factor of %f, start size %lu, target size
    // %u,
    // "
    //        "bytes_occupied %u, bytes required %u\n",
    //        shrink, N(), target_size, bytes_occupied, bytes_required);
    // print_pma();
    shrink_list(shrink);
    shrinking_timer.stop();
  } else { // not doubling
    // in parallel, redistribute ranges

    static_timer redistribute_timer("redistributing_remove_batch");
    redistribute_timer.start();
    ParallelTools::parallel_for(
        0, ranges_to_redistribute_3.size(), [&](uint64_t i) {
          const auto &item = ranges_to_redistribute_3[i];
          auto start = std::get<0>(item);
          auto len = std::get<1>(item);
          // printf("REDISTRIBUTING RANGE: start %lu, len %lu, bytes %u\n",
          // key,
          //        value.first, value.second);
          // if constexpr (std::is_same_v<leaf, delta_compressed_leaf<
          //                                        typename
          //                                        leaf::value_type>>)
          //                                        {
          //   print_pma();
          // }

          std::pair<leaf, uint64_t> merged_data =
              leaf::template merge<head_form == InPlace, store_density>(
                  &(data_array[start]), len / logN(), logN(),
                  start / elts_per_leaf(),
                  [this](uint64_t index) -> T & {
                    return index_to_head(index);
                  },
                  density_array);

          if constexpr (std::is_same_v<leaf, delta_compressed_leaf<T>>) {
            // std::cout << std::get<2>(item) << std::endl;
            assert(((double)merged_data.second) / (len / logN()) <
                   (density_limit() * len));
          }

          // merged_data.first.print();
          // number of leaves, num elemtns in input leaf, num elements in
          // output leaf, dest region
          merged_data.first.template split<head_form == InPlace, store_density>(
              len / logN(), merged_data.second, logN(), &(data_array[start]),
              start / elts_per_leaf(),
              [this](uint64_t index) -> T & { return index_to_head(index); },
              density_array);
          // printf("merged leaf\n");
          // merged_data.first.print();
          // printf("merged leaf done\n");
          free(reinterpret_cast<uint8_t *>(merged_data.first.array) -
               sizeof(T));
        });
    redistribute_timer.stop();
  }
  count_elements -= num_elts_removed;
  assert(check_nothing_full());
  assert(check_leaf_heads());
  total_timer.stop();
  return num_elts_removed;
}

// return true if the element was inserted, false if it was already there
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
bool CPMA<leaf, head_form, B_size, store_density>::insert(T e) {
  static_timer total_timer("total_insert_timer");
  static_timer find_timer("find_insert_timer");
  static_timer modify_timer("modify_insert_timer");
  static_timer rebalence_timer("rebalence_insert_timer");

  // std::cout << "before inserting " << e << std::endl;
  // print_pma();
  if (e == 0) {
    bool had_before = has_0;
    has_0 = true;
    return !had_before;
  }
  total_timer.start();
  find_timer.start();
  uint64_t leaf_start = find_containing_leaf_index(e);
  find_timer.stop();
  modify_timer.start();
  leaf l(index_to_head(leaf_start / elts_per_leaf()),
         index_to_data(leaf_start / elts_per_leaf()), leaf_size_in_bytes());
  // printf("leaf_start = %lu\n", leaf_start);
  // print_pma();
  auto [inserted, byte_count] = l.template insert<head_form == InPlace>(e);
  modify_timer.stop();
  count_elements += static_cast<uint64_t>(inserted);
  if (!inserted) {
    total_timer.stop();
    return false;
  }
  if constexpr (store_density) {
    density_array[leaf_start / elts_per_leaf()] = byte_count;
  }
  rebalence_timer.start();
  uint64_t byte_index = leaf_start * sizeof(T);

  uint64_t level = H();
  uint64_t len_bytes = logN();

  uint64_t node_byte_index = find_leaf(byte_index);

  // if we are not a power of 2, we don't want to go off the end
  // TODO(AUTHOR) don't just keep counting the same range until we get to
  // the top node also it means to make it grow just insert into the smaller
  // side still keeps the theoretical behavior since it still grows
  // geometrically
  uint64_t local_len_bytes = std::min(len_bytes, N() - node_byte_index);
  // std::cout << byte_count << ", " << upper_density_bound(level) << ", "
  //           << local_len_bytes << std::endl;

  while (byte_count >= upper_density_bound(level) * local_len_bytes) {
    len_bytes *= 2;

    if (len_bytes <= N()) {
      if (level > 0) {
        level--;
      }
      uint64_t new_byte_node_index = find_node(node_byte_index, len_bytes);
      local_len_bytes = std::min(len_bytes, N() - new_byte_node_index);

      if (local_len_bytes == len_bytes) {
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          byte_count += get_density_count_no_overflow(
              new_byte_node_index + len_bytes / 2, len_bytes / 2);
        }
      } else {
        // since its to the left it can never leave the range
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          uint64_t length = len_bytes / 2;
          if (new_byte_node_index + len_bytes > N()) {
            length = N() - (new_byte_node_index + (len_bytes / 2));
          }
          // only count if there were new real elements
          if (N() > new_byte_node_index + (len_bytes / 2)) {
            byte_count += get_density_count_no_overflow(
                new_byte_node_index + len_bytes / 2, length);
          }
        }
      }

      node_byte_index = new_byte_node_index;
    } else {
      grow_list(growing_factor);
      rebalence_timer.stop();
      total_timer.stop();
      return true;
    }
  }
  if (len_bytes > logN()) {
    // std::cout << "redistributing " << local_len_bytes / logN()
    //           << " leaves starting at byte index " << node_byte_index
    //           << std::endl;
    // print_pma();
    std::pair<leaf, uint64_t> merged_data =
        leaf::template merge<head_form == InPlace, store_density>(
            (T *)(byte_array() + node_byte_index), local_len_bytes / logN(),
            logN(), node_byte_index / logN(),
            [this](uint64_t index) -> T & { return index_to_head(index); },
            density_array);

    merged_data.first.template split<head_form == InPlace, store_density>(
        local_len_bytes / logN(), merged_data.second, logN(),
        (T *)(byte_array() + node_byte_index), node_byte_index / logN(),
        [this](uint64_t index) -> T & { return index_to_head(index); },
        density_array);
#if DEBUG == 1
    for (uint64_t i = node_byte_index; i < node_byte_index + local_len_bytes;
         i += logN()) {
      if constexpr (compressed) {
        if (get_density_count(i, logN()) >= logN() - leaf::max_element_size) {
          merged_data.first.print();
          print_array_region(node_byte_index / logN(),
                             (node_byte_index + local_len_bytes) / logN());
        }
        ASSERT(get_density_count(i, logN()) < logN() - leaf::max_element_size,
               "%lu >= %lu\n tried to split %lu bytes into %lu leaves\n i = "
               "%lu\n",
               get_density_count(i, logN()), logN() - leaf::max_element_size,
               merged_data.second, local_len_bytes / logN(),
               (i - node_byte_index) / logN());
      } else {
        ASSERT(get_density_count(i, logN()) <= logN() - leaf::max_element_size,
               "%lu > %lu\n tried to split %lu bytes into %lu leaves\n i = "
               "%lu\n",
               get_density_count(i, logN()), logN() - leaf::max_element_size,
               merged_data.second, local_len_bytes / logN(),
               (i - node_byte_index) / logN());
      }
    }
#endif
    free(reinterpret_cast<uint8_t *>(merged_data.first.array) - sizeof(T));
    // print_pma();
  }
  rebalence_timer.stop();
  total_timer.stop();
  return true;
}

// return true if the element was removed, false if it wasn't already there
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
bool CPMA<leaf, head_form, B_size, store_density>::remove(T e) {
  static_timer total_timer("total_remove_timer");
  static_timer find_timer("find_remove_timer");
  static_timer modify_timer("modify_remove_timer");
  static_timer rebalence_timer("rebalence_remove_timer");
  if (count_elements == 0) {
    return false;
  }
  if (e == 0) {
    bool had_before = has_0;
    has_0 = false;
    return had_before;
  }
  total_timer.start();
  find_timer.start();
  uint64_t leaf_start = find_containing_leaf_index(e);
  find_timer.stop();
  modify_timer.start();
  leaf l(index_to_head(leaf_start / elts_per_leaf()),
         index_to_data(leaf_start / elts_per_leaf()), leaf_size_in_bytes());
  auto [removed, byte_count] = l.template remove<head_form == InPlace>(e);
  modify_timer.stop();
  count_elements -= static_cast<uint64_t>(removed);
  if (!removed) {
    total_timer.stop();
    return false;
  }
  if constexpr (store_density) {
    density_array[leaf_start / elts_per_leaf()] = byte_count;
  }
  rebalence_timer.start();

  uint64_t byte_index = leaf_start * sizeof(T);

  uint64_t level = H();
  uint64_t len_bytes = logN();

  uint64_t node_byte_index = find_leaf(byte_index);

  // if we are not a power of 2, we don't want to go off the end
  // TODO(AUTHOR) don't just keep counting the same range until we get to
  // the top node also it means to make it grow just insert into the smaller
  // side still keeps the theoretical behavior since it still grows
  // geometrically
  uint64_t local_len_bytes = std::min(len_bytes, N() - node_byte_index);

  while (byte_count <= lower_density_bound(level) * local_len_bytes) {

    len_bytes *= 2;

    if (len_bytes <= N()) {

      if (level > 0) {
        level--;
      }
      uint64_t new_byte_node_index = find_node(node_byte_index, len_bytes);
      local_len_bytes = std::min(len_bytes, N() - new_byte_node_index);
      if (local_len_bytes == len_bytes) {
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          byte_count += get_density_count_no_overflow(
              new_byte_node_index + len_bytes / 2, len_bytes / 2);
        }
      } else {
        // since its to the left it can never leave the range
        if (new_byte_node_index < node_byte_index) {
          byte_count +=
              get_density_count_no_overflow(new_byte_node_index, len_bytes / 2);
        } else {
          uint64_t length = len_bytes / 2;
          if (new_byte_node_index + len_bytes > N()) {
            length = N() - (new_byte_node_index + (len_bytes / 2));
          }
          // only count if there were new real elements
          if (N() > new_byte_node_index + (len_bytes / 2)) {
            byte_count += get_density_count_no_overflow(
                new_byte_node_index + len_bytes / 2, length);
          }
        }
      }

      node_byte_index = new_byte_node_index;
    } else {
      shrink_list(growing_factor);
      rebalence_timer.stop();
      total_timer.stop();
      return true;
    }
  }
  if (len_bytes > logN()) {
    std::pair<leaf, uint64_t> merged_data =
        leaf::template merge<head_form == InPlace, store_density>(
            (T *)(byte_array() + node_byte_index), local_len_bytes / logN(),
            logN(), node_byte_index / logN(),
            [this](uint64_t index) -> T & { return index_to_head(index); },
            density_array);

    merged_data.first.template split<head_form == InPlace, store_density>(
        local_len_bytes / logN(), merged_data.second, logN(),
        (T *)(byte_array() + node_byte_index), node_byte_index / logN(),
        [this](uint64_t index) -> T & { return index_to_head(index); },
        density_array);
    free(reinterpret_cast<uint8_t *>(merged_data.first.array) - sizeof(T));
  }
  rebalence_timer.stop();
  total_timer.stop();
  return true;
}

// return the amount of memory the structure uses
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::get_size() const {
  uint64_t size_from_heads = 0;
  if constexpr (head_form != InPlace) {
    size_from_heads = head_array_size();
  }
  if constexpr (store_density) {
    return sizeof(*this) + N() + size_from_heads +
           total_leaves() * sizeof(uint16_t);
  }
  return sizeof(*this) + N() + size_from_heads;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t
CPMA<leaf, head_form, B_size, store_density>::sum_serial(uint64_t start,
                                                         uint64_t end) const {
  uint64_t total = 0;
  for (uint64_t i = start; i < end; i++) {
    leaf l(index_to_head(i), index_to_data(i), leaf_size_in_bytes());
    total += l.template sum<head_form == InPlace>();
  }
  return total;
}
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::sum_serial() const {
  uint64_t num_leaves = total_leaves();
  return sum_serial(0, num_leaves);
}
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
inline uint64_t
CPMA<leaf, head_form, B_size, store_density>::sum_parallel(uint64_t start,
                                                           uint64_t end) const {
  ParallelTools::Reducer_sum<uint64_t> total_red;
  uint64_t chunk_size = 100;
  ParallelTools::parallel_for(start, end, chunk_size, [&](int64_t i) {
    size_t local_end = i + chunk_size;
    if (local_end > end) {
      local_end = end;
    }
    uint64_t local_sum = 0;
    for (size_t j = i; j < local_end; j++) {
      leaf l(index_to_head(j), index_to_data(j), leaf_size_in_bytes());
      local_sum += l.template sum<head_form == InPlace>();
    }
    total_red.add(local_sum);
  });
  return total_red.get();
}

// return the sum of all elements stored
// just used to see how fast it takes to iterate
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint64_t CPMA<leaf, head_form, B_size, store_density>::sum() const {
  uint64_t num_leaves = total_leaves();
#if CILK == 0
  return sum_serial(0, num_leaves);
#endif
  if (num_leaves < (uint64_t)ParallelTools::getWorkers()) {
    return sum_serial(0, num_leaves);
  }
  return sum_parallel(0, num_leaves);
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
template <bool no_early_exit, class F>
void CPMA<leaf, head_form, B_size, store_density>::serial_map(
    F f, T start, T end, uint64_t start_hint, uint64_t end_hint) const {
  // printf("map start\n");
  // skips the all zeros element, but fine for now since its a self loop and
  // we don't care about it
  uint64_t leaf_idx = find_containing_leaf_index(start, start_hint, end_hint);
  while (leaf_idx <= N() / sizeof(T) &&
         index_to_head(leaf_idx / elts_per_leaf()) < end) {
    // printf("leaf start\n");
    // printf("leaf map from index %lu\n", leaf_idx);
    leaf l(index_to_head(leaf_idx / elts_per_leaf()),
           index_to_data(leaf_idx / elts_per_leaf()), leaf_size_in_bytes());
    if (l.template map<no_early_exit>(f, start, end)) {
      if constexpr (!no_early_exit) {
        return;
      }
    }
    leaf_idx += elts_per_leaf();
  }
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
template <bool no_early_exit, class F>
bool CPMA<leaf, head_form, B_size, store_density>::map(F f) const {
  // printf("map start\n");
  // skips the all zeros element, but fine for now since its a self loop and
  // we don't care about it
  for (uint64_t i = 0; i < total_leaves(); i++) {
    // printf("leaf start\n");
    // printf("leaf map from index %lu\n", leaf_idx);
    leaf l(index_to_head(i), index_to_data(i), leaf_size_in_bytes());
    if (l.template map<no_early_exit>(f)) {
      if constexpr (!no_early_exit) {
        return true;
      }
    }
  }
  return false;
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
template <bool no_early_exit, class F>
void CPMA<leaf, head_form, B_size, store_density>::serial_map_with_hint(
    F f, T end, const std::pair<uint8_t *, T> &hint) const {
  // do the partial map if need be
  uintptr_t byte_start_index = hint.first - byte_array();
  // printf("byte_start_index = %lu\n", byte_start_index);
  if (hint.first != nullptr) {
    // printf("patial lead map\n");
    if (leaf::template partial_map<no_early_exit>(f, end, hint.first,
                                                  hint.second)) {
      if constexpr (!no_early_exit) {
        return;
      }
    }
    // find the start of the next leaf
    byte_start_index -= (byte_start_index % logN());
    byte_start_index += logN();
  }
  // printf("byte_start_index at next leaf = %lu\n", byte_start_index);
  uint64_t leaf_idx = byte_start_index / sizeof(T);
  if (hint.first == nullptr) {
    leaf_idx = hint.second * elts_per_leaf();
  }
  // printf("leaf_idx = %lu, N() = %lu, logN() = %u\n", leaf_idx,
  //        (N() / sizeof(T)), logN());
  while (leaf_idx < (N() / sizeof(T))) {
    if (index_to_head(leaf_idx / elts_per_leaf()) >= end) {
      // printf("found end, leaf_idx = %lu, array[leaf_idx] = %lu, end =
      // %lu\n",
      //        leaf_idx, array[leaf_idx], end);
      return;
    }
    // printf("leaf map from index %lu\n", leaf_idx);
    leaf l(index_to_head(leaf_idx / elts_per_leaf()),
           index_to_data(leaf_idx / elts_per_leaf()), leaf_size_in_bytes());
    if (l.template map_no_start<no_early_exit>(f, end)) {
      if constexpr (!no_early_exit) {
        return;
      }
    }
    leaf_idx += elts_per_leaf();
  }
}
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
template <bool no_early_exit, class F>
void CPMA<leaf, head_form, B_size, store_density>::serial_map_with_hint_par(
    F f, T end, const std::pair<uint8_t *, T> &hint,
    const std::pair<uint8_t *, T> &end_hint) const {
  // do the partial map if need be
  uintptr_t byte_start_index = hint.first - byte_array();
  // printf("byte_start_index = %lu\n", byte_start_index);
  if (hint.first != nullptr) {
    // printf("patial lead map\n");
    if (leaf::template partial_map<no_early_exit>(f, end, hint.first,
                                                  hint.second)) {
      if constexpr (!no_early_exit) {
        return;
      }
    }
    // find the start of the next leaf
    byte_start_index -= (byte_start_index % logN());
    byte_start_index += logN();
  }
  uint64_t leaf_idx = byte_start_index / sizeof(T);
  if (hint.first == nullptr) {
    leaf_idx = hint.second * elts_per_leaf();
  }

  uint64_t leaf_end_idx = (end_hint.first - byte_array()) / sizeof(T);
  if (end_hint.first == nullptr) {
    leaf_end_idx = end_hint.second * elts_per_leaf();
  }
  // TODO(AUTHOR) find out why we need to look one more
  if (leaf_end_idx <= N() / sizeof(T)) {
    leaf_end_idx += elts_per_leaf();
  }
  ParallelTools::parallel_for(
      leaf_idx, leaf_end_idx, elts_per_leaf(), [&](uint64_t idx) {
        if (index_to_head(idx / elts_per_leaf()) >= end) {
          return;
        }
        leaf l(index_to_head(idx / elts_per_leaf()),
               index_to_data(idx / elts_per_leaf()), leaf_size_in_bytes());
        l.template map_no_start<no_early_exit>(f, end);
      });
}

template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
typename CPMA<leaf, head_form, B_size, store_density>::T
CPMA<leaf, head_form, B_size, store_density>::max() const {
  leaf l(index_to_head(total_leaves() - 1), index_to_data(total_leaves() - 1),
         leaf_size_in_bytes());
  return l.last();
}
template <typename leaf, HeadForm head_form, uint64_t B_size,
          bool store_density>
uint32_t CPMA<leaf, head_form, B_size, store_density>::num_nodes() const {
  return (max() >> 32) + 1;
}

#endif
