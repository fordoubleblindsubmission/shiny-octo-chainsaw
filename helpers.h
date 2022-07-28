#ifndef HELPERS_H
#define HELPERS_H

#include "ParallelTools/parallel.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <sstream>
#include <sys/time.h>
#include <vector>
#include <x86intrin.h>
#ifndef NDEBUG
#define dprintf(fmt, args...) fprintf(stderr, fmt, ##args)
#define ASSERT(PREDICATE, ...)                                                 \
  do {                                                                         \
    if (!(PREDICATE)) {                                                        \
      fprintf(stderr,                                                          \
              "%s:%d (%s) Assertion " #PREDICATE " failed: ", __FILE__,        \
              __LINE__, __PRETTY_FUNCTION__);                                  \
      fprintf(stderr, __VA_ARGS__);                                            \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#else
#define dprintf(fmt, args...) /* Don't do anything in release builds */
#define ASSERT(...)           // Nothing.
#endif

template <typename T> T *newA(size_t n) { return (T *)malloc(n * sizeof(T)); }

#define watch(x) (#x) << "=" << (x) << std::endl;

#define REPORT2(a, b) watch(a) << ", " << watch(b) << std::endl;
#define REPORT3(a, b, c) REPORT2(a, b) << ", " << watch(c) << std::endl;
#define REPORT4(a, b, c, d) REPORT3(a, b, c) << ", " << watch(d) << std::endl;
#define REPORT5(a, b, c, d, e)                                                 \
  REPORT4(a, b, c, d) << ", " << watch(e) << std::endl;

#define intT int32_t
#define uintT uint32_t

// find index of first 1-bit (least significant bit)
static inline uint32_t bsf_word(uint32_t word) {
  uint32_t result;
  __asm__("bsf %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline long bsf_long(long word) {
  long result;
  __asm__("bsfq %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline int bsr_word(int word) {
  int result;
  __asm__("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline long bsr_long(long word) {
  long result;
  __asm__("bsrq %1, %0" : "=r"(result) : "r"(word));
  return result;
}

static inline bool power_of_2(int word) {
  return __builtin_popcount(word) == 1;
}
static inline uint64_t nextPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1UL;
  n |= n >> 2UL;
  n |= n >> 4UL;
  n |= n >> 8UL;
  n |= n >> 16UL;
  n |= n >> 32UL;
  n++;
  return n;
}

//#define ENABLE_TRACE_TIMER

static inline uint64_t get_usecs() {
  struct timeval st {};
  gettimeofday(&st, nullptr);
  return st.tv_sec * 1000000 + st.tv_usec;
}

template <typename T> static inline T unaligned_load(const void *loc) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Size of T must be either 2, 4, or 8");
  T data;
  std::memcpy(&data, loc, sizeof(T));
  return data;
}

template <typename T> static inline void unaligned_store(void *loc, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Size of T must be either 2, 4, or 8");
  std::memcpy(loc, &value, sizeof(T));
}

inline std::string Join(std::vector<std::string> const &elements,
                        const char delimiter) {
  std::string out;
  for (size_t i = 0; i < elements.size() - 1; i++) {
    out += elements[i] + delimiter;
  }
  out += elements[elements.size() - 1];
  return out;
}

template <class T>
void wrapArrayInVector(T *sourceArray, size_t arraySize,
                       std::vector<T, std::allocator<T>> &targetVector) {
  typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *vectorPtr =
      (typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *)((
          void *)&targetVector);
  vectorPtr->_M_start = sourceArray;
  vectorPtr->_M_finish = vectorPtr->_M_end_of_storage =
      vectorPtr->_M_start + arraySize;
}

template <class T>
void releaseVectorWrapper(std::vector<T, std::allocator<T>> &targetVector) {
  typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *vectorPtr =
      (typename std::_Vector_base<T, std::allocator<T>>::_Vector_impl *)((
          void *)&targetVector);
  vectorPtr->_M_start = vectorPtr->_M_finish = vectorPtr->_M_end_of_storage =
      NULL;
}

template <class T> T prefix_sum_inclusive(std::vector<T> &data) {
  for (size_t i = 1; i < data.size(); i++) {
    data[i] += data[i - 1];
  }
  return data[data.size() - 1];
}

#if __AVX2__
template <class T> inline void Log(const __m256i &value) {
  const size_t n = sizeof(__m256i) / sizeof(T);
  T buffer[n];
  _mm256_storeu_si256((__m256i *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << +buffer[i] << " ";
  }
  std::cout << std::endl;
}

template <class T> inline void Log(const __m128i &value) {
  const size_t n = sizeof(__m128i) / sizeof(T);
  T buffer[n];
  _mm_storeu_si128((__m128i *)buffer, value);
  for (size_t i = 0; i < n; i++) {
    std::cout << +buffer[i] << " ";
  }
  std::cout << std::endl;
}
#endif

uint64_t tzcnt(uint64_t num) {
#ifdef __BMI__
  return _tzcnt_u64(num);
#endif
  uint64_t count = 0;
  while ((num & 1) == 0) {
    count += 1;
    num >>= 1;
  }
  return count;
}

[[nodiscard]] inline uint64_t e_index(uint64_t index, uint64_t length) {
  uint64_t pos_0 = tzcnt(~index) + 1;
  uint64_t num_bits = bsr_long(length) + 1;
  return (1UL << (num_bits - pos_0)) + (index >> pos_0) - 1;
}

template <uint64_t B>
[[nodiscard]] inline uint64_t bnary_index(uint64_t index,
                                          uint64_t length_rounded) {
  static_assert(B != 0, "B can't be zero\n");
  uint64_t start = length_rounded / B;
  uint64_t tester = B;
  while ((index + 1) % tester == 0) {
    tester *= B;
    start /= B;
  }
  if (start == 0) {
    start = 1;
  }
#if DEBUG == 1
  {
    uint64_t start_test = 1;
    uint64_t test_size = length_rounded / B;
    while ((index + 1) % test_size != 0) {
      test_size /= B;
      start_test *= B;
    }
    ASSERT(
        start == start_test,
        "bad start, got %lu, expected %lu, index = %lu, length_rounded = %lu\n",
        start, start_test, index, length_rounded);
  }
#endif
  uint64_t size_to_consider = length_rounded / start;
  return start + (B - 1) * (index / (size_to_consider)) +
         (index % size_to_consider) / (size_to_consider / B) - 1;
}

#endif
