#include "helpers.h"
#include <cstring>
#include <deque>

static inline uint64_t log(uint64_t word) {
  uint64_t result;
  __asm__ volatile("bsr %1, %0" : "=r"(result) : "r"(word));
  return result;
}

template <class T, int alignment, int log_max = 64> class AlignedAllocator {
  static_assert(alignment >= sizeof(T),
                "alignment must be bigger than or equal to sizeof(T)");

  std::deque<T *> free_array[log_max];

public:
  T *alloc(size_t size) {
    size_t log_size = log(size);
    if (power_of_2(size) && !free_array[log_size].empty()) {
      T *ret = free_array[log_size].front();
      free_array[log_size].pop_front();
      return ret;
    }
    T *ret = static_cast<T *>(aligned_alloc(alignment, size * sizeof(T)));
    if (ret == nullptr) {
      for (int i = 0; i < log_max; i++) {
        for (auto item : free_array[i]) {
          ::free(item);
        }
        free_array[i].clear();
      }
      ret = static_cast<T *>(aligned_alloc(alignment, size * sizeof(T)));
      if (ret == nullptr) {
        printf("bad alloc\n");
        exit(-1);
      }
    }
    return ret;
  }

  T *zalloc(size_t size) {
    T *ret = alloc(size);
    std::memset(ret, 0, size * sizeof(T));
    return ret;
  }

  void free(T *pointer, size_t size) {
    if (!power_of_2(size)) {
      ::free(pointer);
      return;
    }
    size_t log_size = log(size);
    free_array[log_size].push_front(pointer);
  }
  ~AlignedAllocator() {
    for (int i = 0; i < log_max; i++) {
      for (auto item : free_array[i]) {
        ::free(item);
      }
    }
  }
};