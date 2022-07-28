#ifndef PMA_HPP
#define PMA_HPP
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

#include <iostream>
#include <vector>

#include "AlignedAllocator.hpp"
#include "helpers.h"

#define SMALL_META_DATA 0

#define MIN_LEAF_SIZE 4

#define GROWING_FACTOR 1.1

static_assert(MIN_LEAF_SIZE == 4 || MIN_LEAF_SIZE == 8,
              "MIN_LEAF_SIZE must be either 4 or 8");

template <class T, T null_value> class PMA {
  static_assert(std::is_trivial<T>::value, "T must be a trivial type");

private:
  T *array;
  uint64_t n;

  uint64_t count_elements = 0;

  static constexpr uint64_t linear_search_threshold = 1024 / sizeof(T);
  // TODO(AUTHOR) do something so that this could be used when we want small
  // meta data, it works now, but it is big
  AlignedAllocator<T, 64, 64> allocator;

  std::pair<float, float> density_bound(uint64_t depth) const;

#if SMALL_META_DATA == 1
#if MIN_LEAF_SIZE == 4
  uint8_t loglogN() const { return bsr_word(bsr_word(N())); }
#elif MIN_LEAF_SIZE == 8
  uint8_t loglogN() const { return bsr_word(bsr_word(N())) + 1; }
#endif
  uint32_t logN() const { return (1U << loglogN()); }
  uint32_t mask_for_leaf() const { return ~(logN() - 1U); }
  uint32_t H() const { return bsr_word(N() / logN()); }
  float upper_density_bound(uint64_t depth) const {
    return density_bound(depth).second;
  }
  float lower_density_bound(uint64_t depth) const {
    return density_bound(depth).first;
  }

#else
  uint8_t loglogN_;
  uint8_t loglogN() const { return loglogN_; }
  uint32_t logN_;
  uint32_t logN() const { return logN_; }
  uint32_t mask_for_leaf_;
  uint32_t mask_for_leaf() const { return mask_for_leaf_; }
  uint32_t H_;
  uint32_t H() const { return H_; }

  std::pair<float, float> density_bound_[sizeof(T) * 8];

  float lower_density_bound(uint64_t depth) const {
    return density_bound_[depth].first;
  }
  float upper_density_bound(uint64_t depth) const {
    return density_bound_[depth].second;
  }

#endif

  double density_limit() const {
    return static_cast<float>(logN() - 1U) / logN();
  }

  T *pack_elements(uint64_t index, uint64_t original_n, uint64_t new_n,
                   [[maybe_unused]] bool from_double = false);

  void grow_list(double factor = 2.0);
  void shrink_list(double factor = 2.0);

  void slide_right(uint64_t index);
  void slide_left(uint64_t index);
  void redistribute(uint64_t index, uint64_t len, uint64_t num_elements,
                    bool for_double = false);

  uint64_t get_density_count(uint64_t index, uint64_t len) const;

  uint64_t linear_search(T e, uint64_t start = 0,
                         uint64_t end = UINT64_MAX) const;

  uint64_t search_no_empty_leaves(T e) const;

  void place(uint64_t index, T e);
  void take(uint64_t index);

  uint64_t find_prev_valid(uint64_t start) const;

  bool check_no_full_leaves() const;
  bool check_no_empty_leaves() const;

  uint64_t get_depth(uint64_t len) const { return bsr_word(N() / len); }

  uint64_t find_leaf(uint64_t index) const { return index & mask_for_leaf(); }
  uint64_t find_node(uint64_t index, uint64_t len) const {
    return (index / len) * len;
  }
  uint64_t next_leaf(uint64_t index) const {
    return ((index >> loglogN()) + 1) << (loglogN());
  }
  uint64_t next_leaf(uint64_t index, uint64_t llN) const {
    return ((index >> llN) + 1) << (llN);
  }

  bool check_packed_left() const;
  uint64_t search(T e) const;

public:
  explicit PMA(uint64_t init_n = 16);
  PMA(const PMA &source);
  ~PMA() { allocator.free(array, N()); }
  void print_pma() const;
  void print_array() const;
  bool has(T e) const;
  bool insert(T e);
  bool remove(T e);
  uint64_t get_size() const;
  uint64_t get_size_no_allocator() const;

  uint64_t get_element_count() const { return count_elements; }

  uint64_t N() const { return n; }
  uint64_t sum() const;

  bool check_sorted() const;

  // TODO(AUTHOR) iterator should deal with locks, maybe
  class iterator {
  public:
    uint64_t length;
    T *array;
    // T null_value;
    uint64_t place;
    uint8_t loglogN;

    explicit iterator(uint64_t pl)
        : length(0), array(nullptr), place(pl), loglogN(0) {}

    explicit iterator(const PMA &pma)
        : length(pma.N()), array(pma.array), place(0), loglogN(pma.loglogN()) {}

    iterator(const PMA &pma, uint64_t pl)
        : length(pma.N()), array(pma.array), place(pl), loglogN(pma.loglogN()) {
    }

    bool operator==(const iterator &other) const {
      return (place == other.place);
    }
    bool operator!=(const iterator &other) const {
      return (place != other.place);
    }
    bool operator<(const iterator &other) const {
      return (place < other.place);
    }
    bool operator>=(const iterator &other) const {
      return (place >= other.place);
    }
    iterator &operator++() {
      place += 1;
      while ((place < length) && array[place] == null_value) {
        place = ((place >> loglogN) + 1) << (loglogN);
      }
      return *this;
    }
    T operator*() const { return array[place]; }
    ~iterator() = default;
  };

  iterator begin() const {
    return (count_elements > 0) ? iterator(*this) : iterator(N());
  }
  iterator end() const { return iterator(N()); }
};

// when adjusting the list size, make sure you're still in the
// density bound

template <class T, T null_value>
std::pair<float, float>
PMA<T, null_value>::density_bound(uint64_t depth) const {
  std::pair<double, double> pair;

  // between 1/4 and 1/2
  // pair.x = 1.0/2.0 - (( .25*depth)/list->H);
  // between 1/8 and 1/4
  pair.first = 1.0 / 4.0 - ((.125 * depth) / H());
  // TODO(AUTHOR) not sure why I need this
  if (H() < 12) {
    pair.second = 3.0 / 4.0 + (((1.0 / 4.0) * depth) / H());
  } else {
    pair.second = 15.0 / 16.0 + (((1.0 / 16.0) * depth) / H());
  }

  if (pair.second > density_limit()) {
    pair.second = density_limit() + .001;
  }
  return pair;
}

// grabs elements from a region and packs them tightly together into a different
// array
// assumes the region is already locked
template <class T, T null_value>
T *PMA<T, null_value>::pack_elements(uint64_t index, uint64_t original_n,
                                     uint64_t new_n,
                                     [[maybe_unused]] bool from_double) {
  auto output_array = allocator.alloc(new_n);
  uint64_t start = 0;
  for (uint64_t i = index; i < index + original_n; i += MIN_LEAF_SIZE) {

#pragma clang loop unroll(full)
    for (uint64_t j = 0; j < MIN_LEAF_SIZE; j++) {
      output_array[start] = array[j + i];
      start += (array[j + i] != null_value);
    }
  }
  return output_array;
}
template <>
uint64_t *PMA<uint64_t, 0>::pack_elements(uint64_t index, uint64_t original_n,
                                          uint64_t new_n,
                                          [[maybe_unused]] bool from_double) {
  auto output_array = allocator.alloc(new_n);

  uint64_t start = 0;
  if (logN() == 4 || (logN() == 8 && from_double)) {
#ifdef __AVX2__
    __m256i zero = _mm256_setzero_si256();
    uint64_t i = index;
    __m256i data = _mm256_load_si256((__m256i *)&array[i]);
    for (; i < index + original_n - 4; i += 4) {
      __m256i data_next = _mm256_load_si256((__m256i *)&array[i + 4]);
#ifdef __AVX512VL__
      __mmask8 mask = _mm256_cmpeq_epi64_mask(data, zero);
      uint32_t count = __builtin_popcount(mask);
#else
      __m256i compare = _mm256_cmpeq_epi64(data, zero);
      int mask = _mm256_movemask_epi8(compare);
      uint32_t count = __builtin_popcount(mask) / 8;
#endif
      _mm256_storeu_si256((__m256i *)&output_array[start], data);
      data = data_next;
      start += 4 - count;
    }
    _mm256_storeu_si256((__m256i *)&output_array[start], data);
#else
    for (uint64_t i = index; i < index + original_n; i += 4) {
#pragma clang loop unroll(full)
      for (uint64_t j = 0; j < 4; j++) {
        output_array[start] = array[j + i];
        start += (array[j + i] != 0);
      }
    }
#endif
  } else { // logN() > 4 and __AVX512VL__
#ifdef __AVX512VL__
    __m512i zero = _mm512_setzero_si512();
    uint64_t i = index;
    //__m512i data = _mm512_load_si512((__m512i *)&array[i]);
    for (; i < index + original_n; i += 8) {
      __m512i data = _mm512_load_si512((__m512i *)&array[i]);
      //__m512i data_next = _mm512_load_si512((__m512i *)&array[i + 8]);
      __mmask8 mask = _mm512_cmpeq_epi64_mask(data, zero);
      uint32_t count = __builtin_popcount(mask);
      _mm512_storeu_si512((__m512i *)&output_array[start], data);
      // data = data_next;
      start += 8 - count;
    }
    //_mm512_storeu_si512((__m512i *)&output_array[start], data);
#else
#ifdef __AVX2__
    __m256i zero = _mm256_setzero_si256();
    uint64_t i = index;
    for (; i < index + original_n; i += 8) {
      __m256i data1 = _mm256_load_si256((__m256i *)&array[i]);
      __m256i data2 = _mm256_load_si256((__m256i *)&array[i + 4]);
      __m256i compare1 = _mm256_cmpeq_epi64(data1, zero);
      __m256i compare2 = _mm256_cmpeq_epi64(data2, zero);
      int mask1 = _mm256_movemask_epi8(compare1);
      int mask2 = _mm256_movemask_epi8(compare2);
      uint32_t count1 = __builtin_popcount(mask1) / 8;
      uint32_t count2 = __builtin_popcount(mask2) / 8;
      _mm256_storeu_si256((__m256i *)&output_array[start], data1);
      start += 4 - count1;
      _mm256_storeu_si256((__m256i *)&output_array[start], data2);
      start += 4 - count2;
    }
#else
    for (uint64_t i = index; i < index + original_n; i += 8) {
#pragma clang loop unroll(full)
      for (uint64_t j = 0; j < 8; j++) {
        output_array[start] = array[j + i];
        start += (array[j + i] != 0);
      }
    }
#endif
#endif
  }

  return output_array;
}

template <>
uint32_t *PMA<uint32_t, 0>::pack_elements(uint64_t index, uint64_t original_n,
                                          uint64_t new_n, bool from_double) {
  auto output_array = allocator.alloc(new_n);

  uint64_t start = 0;
#if MIN_LEAF_SIZE == 4
  if (logN() == 4 || (logN() == 8 && from_double)) {
    for (uint64_t i = index; i < index + original_n; i += 4) {
#ifdef __AVX2__
      __m128i zero = _mm_setzero_si128();
      __m128i data = _mm_load_si128((__m128i *)&array[i]);
#ifdef __AVX512VL__
      __mmask8 mask = _mm_cmpeq_epi32_mask(data, zero);
      uint32_t count = __builtin_popcount(mask);
#else
      __m128i compare = _mm_cmpeq_epi32(data, zero);
      int mask = _mm_movemask_epi8(compare);
      uint32_t count = __builtin_popcount(mask) / 4;
#endif
      _mm_storeu_si128((__m128i *)&output_array[start], data);
      start += 4 - count;
#else
#pragma clang loop unroll(full)
      for (uint64_t j = 0; j < 4; j++) {
        output_array[start] = array[j + i];
        start += (array[j + i] != 0);
      }
#endif
    }
  } else {
#endif
    for (uint64_t i = index; i < index + original_n; i += 8) {
#ifdef __AVX2__
      __m256i zero = _mm256_setzero_si256();
      __m256i data = _mm256_load_si256((__m256i *)&array[i]);
#ifdef __AVX512VL__
      __mmask8 mask = _mm256_cmpeq_epi32_mask(data, zero);
      uint32_t count = __builtin_popcount(mask);
#else
      __m256i compare = _mm256_cmpeq_epi32(data, zero);
      int mask = _mm256_movemask_epi8(compare);
      uint32_t count = __builtin_popcount(mask) / 4;
#endif
      _mm256_storeu_si256((__m256i *)&output_array[start], data);
      start += 8 - count;
#else
#pragma clang loop unroll(full)
    for (uint64_t j = 0; j < 8; j++) {
      output_array[start] = array[j + i];
      start += (array[j + i] != 0);
    }
#endif
    }
#if MIN_LEAF_SIZE == 4
  }
#endif
  return output_array;
}

// doubles the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <class T, T null_value>
void PMA<T, null_value>::grow_list(double factor) {

  uint64_t old_n = n;
  // std::cout << "before: " << n << " / " << count_elements << " = "
  //           << n / count_elements << std::endl;
  n *= factor;
  // std::cout << "after: " << n << " / " << count_elements << " = "
  //           << n / count_elements << std::endl;
  if (n % 16 != 0) {
    n += 16 - (n % 16);
  }
#if SMALL_META_DATA == 0
#if MIN_LEAF_SIZE == 4
  loglogN_ = bsr_word(bsr_word(N()));
#elif MIN_LEAF_SIZE == 8
  loglogN_ = bsr_word(bsr_word(N())) + 1;
#endif
  logN_ = 1U << loglogN();
  mask_for_leaf_ = ~(logN() - 1U);
  H_ = bsr_word(N() / logN());
  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
    // std::cout << "density_bound: " << i << ", " << density_bound_[i].first
    //           << ", " << density_bound_[i].second << std::endl;
  }
#endif
  // printf("doubling list, n = %lu, logN = %u, loglogN = %u\n", N(), logN(),
  //        loglogN());

  auto new_array = allocator.alloc(N());

  std::copy(array, &array[old_n], new_array);
  std::fill(&new_array[old_n], &new_array[N()], null_value);
  allocator.free(array, old_n);
  array = new_array;
  // std::cout << "after copy, before redistribute, num_elements=  "
  //           << count_elements << std::endl;
  // print_pma();
  /* +1 for the element added for this insert */
  redistribute(0, N(), count_elements + 1, true);

  // release_all_locks(task_id, true, GENERAL);
}

// halves the size of the base array
// assumes we already have all the locks from the lock array and the big lock
template <class T, T null_value>
void PMA<T, null_value>::shrink_list(double factor) {
  // printf("doubling list by worker %lu\n", get_worker_num());
  // grab_all_locks(task_id, true, DOUBLE);
  if (N() <= 16) {
    // don't let it get too small
    return;
  }
  uint64_t old_n = n;
  n /= factor;
  if (n % 16 != 0) {
    n += 16 - (n % 16);
  }
  if (n == old_n) {
    n -= 16;
  }
#if SMALL_META_DATA == 0
#if MIN_LEAF_SIZE == 4
  loglogN_ = bsr_word(bsr_word(N()));
#elif MIN_LEAF_SIZE == 8
  loglogN_ = bsr_word(bsr_word(N())) + 1;
#endif
  logN_ = 1U << loglogN();
  mask_for_leaf_ = ~(logN() - 1U);
  H_ = bsr_word(N() / logN());
  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
  }
#endif
  T *new_array = pack_elements(0, old_n, N());

  allocator.free(array, old_n);
  array = new_array;
  // TODO(AUTHOR) figure out what is going on here
  get_density_count(0, N());
  // -1 for the element removed during this delete
  redistribute(0, N(), count_elements - 1);

  // release_all_locks(task_id, true, GENERAL);
}

// index is the beginning of the sequence that you want to slide right.
// assumes you have the lock on the leaf, and that you will never leave your
// leaf
template <class T, T null_value>
void PMA<T, null_value>::slide_right(uint64_t index) {
  T val = array[index];
  array[index] = null_value;
  index += 1;
  // TODO(AUTHOR) maybe do this with a copy, but then it has to always go to
  // the end of the leaf
  while (index < N() && (array[index] != null_value)) {
    T temp_val = array[index];
    array[index] = val;
    val = temp_val;
    index += 1;
  }
  array[index] = val;
}

// index is the beginning of the sequence that you want to slide left.
// the element we start at will be deleted
// assumes you have the lock on the leaf, and that you will never leave your
// leaf
template <class T, T null_value>
void PMA<T, null_value>::slide_left(uint64_t index) {
  // TODO(AUTHOR) maybe do this with a copy, but then it has to always go to
  // the end of the leaf
  while (index + 1 < N()) {
    T val = array[index + 1];
    array[index] = val;
    if (array[index] == null_value) {
      break;
    }
    index += 1;
  }
}

// Evenly redistribute elements in the ofm, given a range to look into
// index: starting position in ofm structure
// len: area to redistribute
// should already be locked
template <class T, T null_value>
void PMA<T, null_value>::redistribute(uint64_t index, uint64_t len,
                                      uint64_t num_elements, bool for_double) {
  T *space_array = pack_elements(index, len, len, for_double);
  // for (uint32_t i = 0; i < len; i++) {
  //   std::cout << space_array[i] << ",";
  // }
  // std::cout << std::endl;

  const uint32_t ln = logN();
  const uint8_t lln = loglogN();

  const uint64_t num_leaves = len >> lln;
  const uint64_t count_per_leaf = num_elements / num_leaves;
  const uint64_t extra = num_elements % num_leaves;
  // std::cout << index << ", " << ln << ", " << (int)lln << ", " << num_leaves
  //           << ", " << count_per_leaf << ", " << extra << ", " <<
  //           num_elements
  //           << std::endl;

  uint64_t j2 = 0;
  uint64_t in = index;
  for (uint64_t i = 0; i < num_leaves; i++) {
    uint64_t count_for_leaf = count_per_leaf + (i < extra);
    // std::cout << count_for_leaf << std::endl;
    uint64_t j3 = j2;
    j2 += count_for_leaf;
    for (uint64_t k = in; k < count_for_leaf + in; k += MIN_LEAF_SIZE) {
      for (uint64_t j = 0; j < MIN_LEAF_SIZE; j++) {
        array[k + j] = space_array[j3 + j];
      }
      j3 += MIN_LEAF_SIZE;
    }
    uint64_t aligned_start =
        in + count_for_leaf + MIN_LEAF_SIZE - (count_for_leaf % MIN_LEAF_SIZE);
    for (uint64_t k = in + count_for_leaf; k < aligned_start; k++) {
      array[k] = null_value;
    }
    for (uint64_t k = aligned_start; k < in + ln; k += MIN_LEAF_SIZE) {
      for (uint64_t j = 0; j < MIN_LEAF_SIZE; j++) {
        array[k + j] = null_value;
      }
    }
    in += ln;
  }

  allocator.free(space_array, len);
}

// asumes the region is already locked
template <class T, T null_value>
uint64_t PMA<T, null_value>::get_density_count(uint64_t index, uint64_t len

) const {

  uint64_t total = 0;
  switch (len) {
#if MIN_LEAF_SIZE == 4
  case 4:
#pragma clang loop unroll(full) vectorize(enable)
    for (uint64_t j = index; j < index + 4; j++) {
      total += (array[j] != null_value);
    }
    return total;
#endif
  case 8:
#pragma clang loop unroll(full) vectorize(enable)
    for (uint64_t j = index; j < index + 8; j++) {
      total += (array[j] != null_value);
    }
    return total;
  default:
    for (uint64_t i = index; i < index + len; i += 16) {
      uint64_t add = 0;
#pragma clang loop unroll(full) vectorize(enable)
      for (uint64_t j = 0; j < 16; j++) {
        add += (array[j + i] != null_value);
      }
      total += add;
    }
    return total;
  }
}
#ifdef __AVX2__
template <>
uint64_t PMA<uint32_t, 0>::get_density_count(uint64_t index, uint64_t len

) const {

  uint64_t total = 0;
  switch (len) {
#if MIN_LEAF_SIZE == 4
  case 4:
    return (array[index] != 0) + (array[index + 1] != 0) +
           (array[index + 2] != 0) + (array[index + 3] != 0);
#endif
  case 8:
    return (array[index] != 0) + (array[index + 1] != 0) +
           (array[index + 2] != 0) + (array[index + 3] != 0) +
           (array[index + 4] != 0) + (array[index + 5] != 0) +
           (array[index + 6] != 0) + (array[index + 7] != 0);
  default:
    __m256i total_vec = _mm256_setzero_si256();
    __m256i zero = _mm256_setzero_si256();
    uint64_t negative_num_zeros;
    for (uint64_t i = index; i < index + len; i += 16) {
      __m256i a = _mm256_load_si256((__m256i *)&array[i]);
      __m256i b = _mm256_load_si256((__m256i *)&array[i + 8]);
      __m256i cmp_a = _mm256_cmpeq_epi32(a, zero);
      __m256i cmp_b = _mm256_cmpeq_epi32(b, zero);
      total_vec = _mm256_add_epi32(total_vec, cmp_a);
      total_vec = _mm256_add_epi32(total_vec, cmp_b);
    }

    total_vec = _mm256_hadd_epi32(total_vec, total_vec);
    total_vec = _mm256_hadd_epi32(total_vec, total_vec);
    negative_num_zeros =
        _mm256_extract_epi32(total_vec, 0) + _mm256_extract_epi32(total_vec, 4);
    total = len + negative_num_zeros;

    return total;
  }
}

template <>
uint64_t PMA<uint64_t, 0>::get_density_count(uint64_t index, uint64_t len

) const {

  uint64_t total = 0;
  switch (len) {
#if MIN_LEAF_SIZE == 4
  case 4:
    return (array[index] != 0) + (array[index + 1] != 0) +
           (array[index + 2] != 0) + (array[index + 3] != 0);
#endif
  case 8:
    return (array[index] != 0) + (array[index + 1] != 0) +
           (array[index + 2] != 0) + (array[index + 3] != 0) +
           (array[index + 4] != 0) + (array[index + 5] != 0) +
           (array[index + 6] != 0) + (array[index + 7] != 0);
  default:
    __m256i total_vec = _mm256_setzero_si256();
    __m256i zero = _mm256_setzero_si256();
    for (uint64_t i = index; i < index + len; i += 16) {
      __m256i a = _mm256_load_si256((__m256i *)&array[i]);
      __m256i b = _mm256_load_si256((__m256i *)&array[i + 4]);
      __m256i c = _mm256_load_si256((__m256i *)&array[i + 8]);
      __m256i d = _mm256_load_si256((__m256i *)&array[i + 12]);
      __m256i cmp_a = _mm256_cmpeq_epi64(a, zero);
      __m256i cmp_b = _mm256_cmpeq_epi64(b, zero);
      __m256i cmp_c = _mm256_cmpeq_epi64(c, zero);
      __m256i cmp_d = _mm256_cmpeq_epi64(d, zero);
      total_vec = _mm256_add_epi64(total_vec, cmp_a);
      total_vec = _mm256_add_epi64(total_vec, cmp_b);
      total_vec = _mm256_add_epi64(total_vec, cmp_c);
      total_vec = _mm256_add_epi64(total_vec, cmp_d);
    }
    uint64_t negative_num_zeros = _mm256_extract_epi64(total_vec, 0) +
                                  _mm256_extract_epi64(total_vec, 1) +
                                  _mm256_extract_epi64(total_vec, 2) +
                                  _mm256_extract_epi64(total_vec, 3);
    total = len + negative_num_zeros;
    return total;
  }
}

#endif

// searches in the unlocked array
// ref counts are dealed with seperatly
// the answer is checked if things have moved
template <class T, T null_value>
uint64_t PMA<T, null_value>::linear_search(T e, uint64_t start,
                                           uint64_t end) const {
  if (end == UINT64_MAX) {
    end = N();
  }
  for (uint64_t i = start; i < end; i++) {
    if ((array[i] >= e) && (array[i] != null_value)) {
      return i;
    }
  }
  return end;
}

// searches in the unlocked array
// ref counts are dealed with seperatly
// the answer is checked if things have moved
template <class T, T null_value>
uint64_t PMA<T, null_value>::search_no_empty_leaves(T e) const {
  uint64_t index;
  if constexpr (false) {
    uint64_t s = 0;
    uint64_t t = N();
    uint64_t mid = (s + t) / 2;
    while (s + linear_search_threshold < t) {
      // std::cout << "s = " << s << " mid = " << mid << " t = " << t <<
      // std::endl;
      T item = array[mid];
      if (item == e) {
        index = mid;
        return index;
      } else if (e < item) {
        t = mid;
        mid = (s + t) / 2;
      } else {
        s = mid;
        mid = (s + t) / 2;
      }
    }
    index = linear_search(e, s, t);
  } else { // branchless binary search
    T *pos = array;
    uint64_t logstep = bsr_word(N());
    uint64_t first_step = N() - (1U << logstep);
    uint64_t step = 1U << logstep;
    pos = (pos[step] < e) ? pos + first_step : pos;
    step >>= 1U;
    while (step >= logN()) {
      pos = (pos[step] < e) ? pos + step : pos;
      step >>= 1;
    }
    // std::cout << e << " should be between " << pos << " and " << pos + step
    //           << std::endl;
    index = linear_search(e, pos - array, pos + 2 * step - array);
  }
  return index;
}

// returns iterator at the smallest element bigger than you in the range
// [start, end)
// searches in the unlocked array
// ref counts are dealed with seperatly
// the answer is checked if things have moved
template <class T, T null_value>
uint64_t PMA<T, null_value>::search(T e) const {
  if (N() <= linear_search_threshold) {
    return linear_search(e);
  }
  if (lower_density_bound(H()) * logN() > 1.0 && N() > 16) {
    return search_no_empty_leaves(e);
  }

  uint64_t index;
  uint64_t s = 0;
  uint64_t t = N() - 1;
  uint64_t mid = (s + t) / 2;
  uint64_t lln = loglogN();
  while (s + linear_search_threshold < t) {
    // std::cout << "s = " << s << " mid = " << mid << " t = " << t <<
    // std::endl;
    T item = array[mid];
    uint64_t check = next_leaf(mid, lln);
    if (item == null_value) {
      // we can't get down to the case where next leaf is outside of the range
      // becuase we would be doing a linear search at that point
      if constexpr (linear_search_threshold > 64) {
        if (check > t) {
          t = mid;
          mid = (s + t) / 2;
          continue;
        }
      }
      // not sure if this case is optimal
      if (array[check] == null_value) {
        uint64_t late_check = next_leaf(check, lln);
        if (late_check > t) {
          t = mid;
          mid = (s + t) / 2;
          continue;
        }
        check = late_check;
      }
      item = array[check];
      if (item == e) {
        index = check;
        return index;
      } else if (e < item) {
        // t = find_prev_valid(mid) + 1;
        t = check;
      } else {
        if (check == s) {
          s = check + 1;
        } else {
          s = check;
        }
      }
      mid = (s + t) / 2;
      continue;
    } // end null case
    if (item == e) {
      index = mid;
      return index;
    } else if (e < item) {
      t = mid;
      mid = (s + t) / 2;
    } else {
      s = mid;
      mid = (s + t) / 2;
    }
  }
  t = linear_search(e, s, t);
  return t;
}

// inserts element e at index, assumes that e is not already in the PMA
// assumes lock for the location of index is held
template <class T, T null_value>
void PMA<T, null_value>::place(uint64_t index, T e) {
  // std::cout << "placeing " << e << " at "
  //           << "index " << index << std::endl;
  if (e == null_value) {
    printf("you can't insert the null value\n");
  }

  assert(index < N());
  uint64_t level = H();
  uint64_t len = logN();

  if (array[index] != null_value) {
    slide_right(index);
  }
  array[index] = e;

  uint64_t node_index = find_leaf(index);

  uint64_t element_count = get_density_count(node_index, len);

  // if we are not a power of 2, we don't want to go off the end
  // TODO(AUTHOR) don't just keep counting the same range until we get to the
  // top node also it means to make it grow just insert into the smaller side
  // still keeps the theoretical behavior since it still grows geometrically
  uint64_t local_len = std::min(len, N() - node_index);

  while (element_count >= upper_density_bound(level) * local_len) {
    len *= 2;

    if (len < N() * 2) {
      if (level > 0) {
        level--;
      }
      uint64_t new_node_index = find_node(node_index, len);
      local_len = std::min(len, N() - new_node_index);

      if (local_len == len) {
        if (new_node_index < node_index) {
          element_count += get_density_count(new_node_index, len / 2);
        } else {
          element_count += get_density_count(new_node_index + len / 2, len / 2);
        }
      } else {
        // since its to the left it can never leave the range
        if (new_node_index < node_index) {
          element_count += get_density_count(new_node_index, len / 2);
        } else {
          uint64_t length = len / 2;
          if (new_node_index + len > N()) {
            length = N() - (new_node_index + (len / 2));
          }
          // only count if there were new real elements
          if (N() > new_node_index + (len / 2)) {
            element_count +=
                get_density_count(new_node_index + len / 2, length);
          }
        }
      }

      node_index = new_node_index;
    } else {
      // std::cout << "printing before double, level = " << level << std::endl;
      // std::cout << element_count << ", "
      //           << upper_density_bound(level) * local_len << ", "
      //           << upper_density_bound(level) << ", " << local_len << ", "
      //           << N() << std::endl;
      // print_pma();
      grow_list(GROWING_FACTOR);
      return;
    }
  }
  if (len > logN()) {
    redistribute(node_index, local_len, element_count);
  }
}

// removes whatever element is at index
// assumes lock for the location of index is held
template <class T, T null_value> void PMA<T, null_value>::take(uint64_t index) {
  uint64_t level = H();
  uint64_t len = logN();
  slide_left(index);

  uint64_t node_index = find_leaf(index);

  uint64_t element_count = get_density_count(node_index, len);

  // if we are not a power of 2, we don't want to go off the end
  // TODO(AUTHOR) don't just keep counting the same range until we get to the
  // top node also it means to make it grow just insert into the smaller side
  // still keeps the theoretical behavior since it still grows geometrically
  uint64_t local_len = std::min(len, N() - node_index);

  while (element_count <= lower_density_bound(level) * local_len) {

    len *= 2;

    if (len <= N()) {

      level--;
      uint64_t new_node_index = find_node(node_index, len);
      local_len = std::min(len, N() - new_node_index);
      if (local_len == len) {
        if (new_node_index < node_index) {
          element_count += get_density_count(new_node_index, len / 2);
        } else {
          element_count += get_density_count(new_node_index + len / 2, len / 2);
        }
      } else {
        if (local_len > len / 2) {
          element_count = get_density_count(new_node_index, local_len);
        }
        // else we already counted since we are the same region
      }

      node_index = new_node_index;
    } else {
      shrink_list(GROWING_FACTOR);
      return;
    }
  }
  if (len > logN()) {
    redistribute(node_index, local_len, element_count);
  }
}

// assumes lock for start is held and we won't leave the leaf
template <class T, T null_value>
uint64_t PMA<T, null_value>::find_prev_valid(uint64_t start) const {
  while (array[start] == null_value && start > 0) {
    start -= 1;
  }
  return start;
}

template <class T, T null_value>
bool PMA<T, null_value>::check_no_full_leaves() const {
  for (uint64_t i = 0; i < N(); i += logN()) {
    bool full = true;
    for (uint64_t j = i; j < i + logN(); j++) {
      if (array[j] == null_value) {
        full = false;
      }
    }
    if (full) {
      return false;
    }
  }
  return true;
}
template <class T, T null_value>
bool PMA<T, null_value>::check_no_empty_leaves() const {
  for (uint64_t i = 0; i < N(); i += logN()) {
    bool empty = true;
    for (uint64_t j = i; j < i + logN(); j++) {
      if (array[j] != null_value) {
        empty = false;
        break;
      }
    }
    if (empty) {
      return false;
    }
  }
  return true;
}

template <class T, T null_value>
bool PMA<T, null_value>::check_packed_left() const {
  for (uint64_t i = 0; i < N(); i += logN()) {
    bool zero = false;
    for (uint64_t j = i; j < i + logN(); j++) {
      if (array[j] == null_value) {
        zero = true;
      } else if (zero) {
        for (uint64_t k = i; k < i + logN(); k++) {
          printf("arr[%lu]=", k);
          std::cout << array[k] << ",";
        }
        printf("\n");
        return false;
      }
    }
  }
  return true;
}

template <class T, T null_value> bool PMA<T, null_value>::check_sorted() const {
  uint64_t end = N();
  uint64_t start = 0;
  T last = array[start];
  for (uint32_t i = start + 1; i < end; i++) {
    if (array[i] != null_value) {
      if (array[i] < last) {
        std::cout << "bad at " << i << ", " << array[i] << " is less than "
                  << last << std::endl;
        return false;
      }
      last = array[i];
    }
  }
  return true;
}

template <class T, T null_value> void PMA<T, null_value>::print_array() const {
  if (N() > 10000) {
    printf("too big to print\n");
    return;
  }
  for (uint32_t i = 0; i < N(); i++) {
    if (array[i] != null_value) {
      std::cout << array[i] << ", ";
    }
  }
  printf("\n");
}

template <class T, T null_value> void PMA<T, null_value>::print_pma() const {
  printf("N = %lu, logN = %u, loglogN = %u, H = %u\n", N(), logN(), loglogN(),
         H());
  printf("count_elements %lu\n", count_elements);
  if (N() > 10000) {
    printf("too big to print\n");
    return;
  }
  for (uint32_t i = 0; i < N(); i += logN()) {
    for (uint32_t j = i; j < i + logN(); j++) {
      if (array[j] != null_value) {
        std::cout << array[j] << ", ";
      } else {
        std::cout << "_" << j << "_,";
      }
    }
    printf("\n");
  }
  printf("\n");
}

template <class T, T null_value>
PMA<T, null_value>::PMA(uint64_t init_n) : n(init_n) {
  array = allocator.alloc(N());
  std::fill(array, &array[N()], null_value);
#if SMALL_META_DATA == 0
#if MIN_LEAF_SIZE == 4
  loglogN_ = bsr_word(bsr_word(N()));
#elif MIN_LEAF_SIZE == 8
  loglogN_ = bsr_word(bsr_word(N())) + 1;
#endif
  logN_ = 1U << loglogN();
  mask_for_leaf_ = ~(logN() - 1U);
  H_ = bsr_word(N() / logN());
  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
  }
#endif
}

template <class T, T null_value>
PMA<T, null_value>::PMA(const PMA<T, null_value> &source)
    : count_elements(source.count_elements), n(source.n) {
  array = allocator.alloc(N());

  std::copy(source.array, &source.array[N()], array);
#if SMALL_META_DATA == 0
  loglogN_ = source.loglogN_;
  logN_ = source.logN_;
  mask_for_leaf_ = source.mask_for_leaf_;
  H_ = source.H_;
  for (uint64_t i = 0; i < sizeof(T) * 8; i++) {
    density_bound_[i] = density_bound(i);
  }
#endif
}

template <class T, T null_value> bool PMA<T, null_value>::has(T e) const {
  uint64_t i = search(e);
  if (i == N()) {
    return false;
  }
  return array[i] == e;
}

// return true if the element was inserted, false if it was already there
template <class T, T null_value> bool PMA<T, null_value>::insert(T e) {
  // std::cout << "before inserting " << e << std::endl;
  // print_pma();
  if (e == null_value) {
    printf("can't insert the null value\n");
    return false;
  }

  uint64_t count_elements_local = count_elements;

  if (count_elements_local == 0) {
    place(0, e);
    count_elements++;
    return true;
  }
  uint64_t loc = search(e);

  if (array[loc] == e) {
    return false;
  }
  // std::cout << "loc is " << loc << std::endl;
  if (loc >= N() - 1) {
    loc = find_prev_valid(loc - 1) + 1;
  }

  // std::cout << "loc was changed to " << loc << std::endl;
  place(loc, e);
  count_elements++;
  // print_pma();
  return true;
}

// return true if the element was removed, false if it wasn't already there
template <class T, T null_value> bool PMA<T, null_value>::remove(T e) {
  uint64_t loc = search(e);
  if (loc == N()) {
    return false;
  }
  if (array[loc] != e) {
    return false;
  }

  take(loc);
  count_elements--;
  // std::cout << count_elements << "," << e << std::endl;
  return true;
}

// return the amount of memory the structure uses
template <class T, T null_value> uint64_t PMA<T, null_value>::get_size() const {
  return sizeof(*this) + (sizeof(T) * N());
}
template <class T, T null_value>
uint64_t PMA<T, null_value>::get_size_no_allocator() const {
  return sizeof(*this) - sizeof(allocator) + (sizeof(T) * N());
}

// return the sum of all elements stored
// just used to see how fast it takes to iterate
template <class T, T null_value> uint64_t PMA<T, null_value>::sum() const {
  uint64_t total = 0;
  for (uint64_t i = 0; i < N(); i++) {
    if (array[i] != null_value) {
      total += array[i];
    }
  }
  return total;
}

#endif
