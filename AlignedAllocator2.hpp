#include "helpers.h"
#include <cstring>
#include <deque>

template <int alignment> class AlignedAllocatorMap {

  std::map<size_t, std::deque<void *>> free_arrays;

public:
  void *alloc(size_t size) {
    if (!free_array[size].empty()) {
      void *ret = free_array[size].front();
      free_array[size].pop_front();
      return ret;
    }
    void *ret = aligned_alloc(alignment, size);
    if (ret == nullptr) {
      for (auto &queue : free_arrays) {
        for (auto item : queue) {
          ::free(item);
        }
      }
      free_arrays.clear();
      ret = aligned_alloc(alignment, size);
      if (ret == nullptr) {
        printf("bad alloc\n");
        exit(-1);
      }
    }
    return ret;
  }

  void *zalloc(size_t size) {
    void *ret = alloc(size);
    std::memset(ret, 0, size);
    return ret;
  }

  void free(T *pointer, size_t size) { free_array[size].push_front(pointer); }
  ~AlignedAllocatorMap() {
    for (auto &queue : free_arrays) {
      for (auto item : queue) {
        ::free(item);
      }
    }
    free_arrays.clear();
  }
};