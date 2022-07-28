#include "CPMA.hpp"
#include "btree.h"
#include "helpers.h"
#include "leaf.hpp"

#include <cstdint>
#include <limits>

#include <set>
#include <sys/time.h>

#include <algorithm>  // generate
#include <functional> // bind
#include <iostream>   // cout
#include <iterator>   // begin, end, and ostream_iterator
#include <random>     // mt19937 and uniform_int_distribution
#include <vector>     // vector

template <typename T>
double test_vector_templated(uint64_t elements_per, uint64_t N) {
  if (elements_per * N > 100000000UL) {
    return -1;
  }
  std::vector<T> data(elements_per * N);
  for (uint64_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);

  std::vector<std::vector<T>> array_vec(N);
  uint64_t start = get_usecs();

  for (size_t j = 0; j < elements_per; j++) {
    for (size_t i = 0; i < N; i++) {
      T element = data[i * elements_per + j];
      auto pos =
          std::lower_bound(array_vec[i].begin(), array_vec[i].end(), element);
      if (pos == array_vec[i].end() || *pos != element) {
        array_vec[i].insert(pos, element);
      }
    }
  }
  uint64_t end = get_usecs();
  double average_time = (double)(end - start) / (N * elements_per);
  T sum = 0;
  for (size_t i = 0; i < N; i++) {
    for (auto e : array_vec[i]) {
      sum += e;
    }
  }
  std::cerr << sum << "\n";

  return average_time;
}

template <typename T>
double test_pma_templated(uint64_t elements_per, uint64_t N) {
  if (elements_per * N > 100000000UL) {
    return -1;
  }
  std::vector<T> data(elements_per * N);
  for (uint64_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);

  std::vector<CPMA<uncompressed_leaf<T>>> array_vec(N);
  uint64_t start = get_usecs();

  for (size_t j = 0; j < elements_per; j++) {
    for (size_t i = 0; i < N; i++) {
      T element = data[i * elements_per + j];
      array_vec[i].insert(element);
    }
  }
  uint64_t end = get_usecs();
  double average_time = (double)(end - start) / (N * elements_per);
  T sum = 0;
  for (size_t i = 0; i < N; i++) {
    sum += array_vec[i].sum();
  }
  std::cerr << sum << "\n";

  return average_time;
}

template <typename T>
double test_b_tree_templated(uint64_t elements_per, uint64_t N) {
  if (elements_per * N > 100000000UL) {
    return -1;
  }
  std::vector<T> data(elements_per * N);
  for (uint64_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);

  std::vector<BTree<T, T>> array_vec(N);
  uint64_t start = get_usecs();

  for (size_t j = 0; j < elements_per; j++) {
    for (size_t i = 0; i < N; i++) {
      T element = data[i * elements_per + j];
      array_vec[i].insert(element);
    }
  }
  uint64_t end = get_usecs();
  double average_time = (double)(end - start) / (N * elements_per);
  T sum = 0;
  for (size_t i = 0; i < N; i++) {
    sum += array_vec[i].sum();
  }
  std::cerr << sum << "\n";

  return average_time;
}

template <typename T> void test_vector(uint64_t max_size = 1000000) {
  std::seed_seq seed;

  std::cout << "elements_per";
  for (uint64_t i = 1; i < max_size; i *= 10) {
    std::cout << ", " << i;
  }
  std::cout << "\n";
  for (uint64_t elements_per = 4096; elements_per < 1UL << 20;
       elements_per *= 2) {
    std::cout << elements_per;
    for (uint64_t i = 1; i < max_size; i *= 10) {
      std::cout << ", " << test_vector_templated<T>(elements_per, i);
    }
    std::cout << std::endl;
  }
}

template <typename T> void test_pma(uint64_t max_size = 1000000) {
  std::seed_seq seed;

  std::cout << "elements_per";
  for (uint64_t i = 1; i < max_size; i *= 10) {
    std::cout << ", " << i;
  }
  std::cout << "\n";
  for (uint64_t elements_per = 8; elements_per < 1UL << 20; elements_per *= 2) {
    std::cout << elements_per;
    for (uint64_t i = 1; i < max_size; i *= 10) {
      std::cout << ", " << test_pma_templated<T>(elements_per, i);
    }
    std::cout << std::endl;
  }
}

template <typename T> void test_b_tree(uint64_t max_size = 1000000) {
  std::seed_seq seed;

  std::cout << "elements_per";
  for (uint64_t i = 1; i < max_size; i *= 10) {
    std::cout << ", " << i;
  }
  std::cout << "\n";
  for (uint64_t elements_per = 8; elements_per < 1UL << 20; elements_per *= 2) {
    std::cout << elements_per;
    for (uint64_t i = 1; i < max_size; i *= 10) {
      std::cout << ", " << test_b_tree_templated<T>(elements_per, i);
    }
    std::cout << std::endl;
  }
}

int main() {
  //   std::cout << "pma\n";
  //   test_pma<uint64_t>(1000000);
  std::cout << "vector\n";
  test_vector<uint64_t>(1000000);
  //   std::cout << "b-tree\n";
  //   test_b_tree<uint64_t>(1000000);
}