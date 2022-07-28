
#include "CPMA.hpp"
#include "PMA.hpp"
#include "ParallelTools/ParallelTools/parallel.h"
#include "ParallelTools/ParallelTools/reducer.h"
#include "ParallelTools/parallel.h"
#include "ParallelTools/reducer.h"
#include "tlx/tlx/container/btree_set.hpp"
#include <asm-generic/errno.h>
#include <cstdint>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wformat"
#pragma clang diagnostic ignored "-Wsign-compare"
#include "btree.h"
#pragma clang diagnostic pop
#include "helpers.h"
#include "io_util.h"
#include "leaf.hpp"
#include "rmat_util.h"
#include "zipf.hpp"

#include "EdgeMapVertexMap/algorithms/BC.h"
#include "EdgeMapVertexMap/algorithms/BFS.h"
#include "EdgeMapVertexMap/algorithms/Components.h"
#include "EdgeMapVertexMap/algorithms/PageRank.h"

#include <iomanip>
#include <limits>
#include <set>
#include <sys/time.h>
#include <unordered_map>
#include <unordered_set>

#include <algorithm> // generate
#include <iostream>  // cout
#include <iterator>  // begin, end, and ostream_iterator
#include <random>    // mt19937 and uniform_int_distribution
#include <vector>    // vector

#include "tlx/container/btree_set.hpp"

template <class T> std::vector<T> create_random_data(size_t n, size_t max_val) {

  std::random_device rd;
  auto seed = rd();
  // std::cout << seed << "\n";
  std::mt19937_64 eng(seed); // a source of random data

  std::uniform_int_distribution<T> dist(0, max_val);
  std::vector<T> v(n);
  for (auto &el : v) {
    el = dist(eng);
  }
  return v;
}
template <class T>
std::vector<T> create_random_data_with_seed(size_t n, size_t max_val,
                                            std::seed_seq &seed) {

  std::mt19937_64 eng(seed); // a source of random data

  std::uniform_int_distribution<T> dist(0, max_val);
  std::vector<T> v(n);
  for (auto &el : v) {
    el = dist(eng);
  }
  return v;
}
template <class T>
std::vector<T> create_random_data_in_parallel(size_t n, size_t max_val) {

  std::vector<T> v(n);
  uint64_t per_worker = n / ParallelTools::getWorkers();
  ParallelTools::parallel_for(0, ParallelTools::getWorkers(), [&](uint64_t i) {
    uint64_t start = i * per_worker;
    uint64_t end = (i + 1) * per_worker;
    if (end > n) {
      end = n;
    }
    std::random_device rd;
    std::mt19937_64 eng(rd()); // a source of random data

    std::uniform_int_distribution<T> dist(0, max_val);
    for (size_t i = start; i < end; i++) {
      v[i] = dist(eng);
    }
  });
  return v;
}

template <class T> void test_set_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::set<T> s;

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}
template <class T>
void test_set_unordered_insert(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::set<T> s;
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}
template <class T> void test_unordered_set_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::unordered_set<T> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("\tsum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}
template <class T>
void test_unordered_set_unordered_insert(uint64_t max_size,
                                         std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  std::unordered_set<T> s;
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);

  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();
  printf("insertion,\t %lu,\t", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time, \t%lu, \tsum_total, \t%lu\n", end - start, sum);
}

template <class T> void test_btree_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  BTree<T, T> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion,\t %lu,", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  auto it = s.begin();
  while (!it.done()) {
    T el = *it;
    sum += el;
    ++it;
  }
  end = get_usecs();
  printf("\tsum_time with iterator, \t%lu, \tsum_total, \t%lu, \t", end - start,
         sum);
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}
template <class T>
void test_btree_unordered_insert(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);
  uint64_t start = 0;
  uint64_t end = 0;
  BTree<T, T> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();

  printf("insertion,\t %lu,", end - start);
  for (uint32_t i = 1; i < max_size; i++) {
    auto node = s.find(data[i]);
    if (node == nullptr) {
      printf("couldn't find data in btree\n");
      exit(0);
    }
  }
  start = get_usecs();
  uint64_t sum = 0;
  auto it = s.begin();
  while (!it.done()) {
    T el = *it;
    sum += el;
    ++it;
  }
  end = get_usecs();
  printf("\tsum_time with iterator, \t%lu, \tsum_total, \t%lu, \t", end - start,
         sum);
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <class T> void test_pma_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  PMA<T, 0> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
    // s.print_pma();
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<leaf, head_form, B_size> s;
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

void test_tlx_btree_ordered_insert(uint64_t max_size) {
  if (max_size > std::numeric_limits<uint64_t>::max()) {
    max_size = std::numeric_limits<uint64_t>::max();
  }
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<uint64_t> s;
  start = get_usecs();
  for (uint32_t i = 0; i < max_size; i++) {
    s.insert(i);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <class T> void test_pma_size(uint64_t max_size) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  std::vector<double> sizes;
  PMA<T, 0> s;
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(i);
    if (i > 1000) {
      sizes.push_back(static_cast<double>(s.get_size_no_allocator()) / i);
    }
    if (i % (max_size / 8) == 0) {
      std::cout << "after " << i << " elements, size is "
                << s.get_size_no_allocator() << ", "
                << s.get_size_no_allocator() / i << " per element, no allocator"
                << std::endl;
    }
  }
  std::cout << "after " << max_size << " elements, size is "
            << s.get_size_no_allocator() << ", "
            << s.get_size_no_allocator() / max_size
            << " per element, no allocator" << std::endl;
  std::cout << std::endl;
  if (!sizes.empty()) {
    std::cout << "statisics ignoring the first 1000" << std::endl;
    double min_num = std::numeric_limits<double>::max();
    double max_num = 0;
    double sum = 0;
    for (auto x : sizes) {
      min_num = std::min(x, min_num);
      max_num = std::max(x, max_num);
      sum += x;
    }
    std::cout << "min = " << min_num << ", max = " << max_num
              << ", average = " << sum / (max_size - 1000) << std::endl;
  }
  std::cout << std::endl;
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_size(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  {
    std::cout << "numbers 1 to " << max_size << std::endl;
    std::vector<double> sizes;
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(i);
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) / i);
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " elements, size is " << s.get_size()
                  << ", " << s.get_size() / i << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " elements, size is " << s.get_size()
              << ", " << s.get_size() / max_size << " per element" << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }
  {
    std::cout << max_size << " uniform random 32 bit numbers" << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::uniform_int_distribution<typename leaf::value_type> distrib(
        1, std::numeric_limits<typename leaf::value_type>::max());
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }

  // zipf distributions
  for (uint32_t max = 1U << 22U; max < 1U << 27U; max *= 8) {
    for (double alpha = 1; alpha < 2.1; alpha += 1) {
      std::cout << "ZipF: max = " << max << " alpha = " << alpha << std::endl;
      zipf zip(max, alpha, seed);

      std::vector<double> sizes;
      CPMA<leaf, head_form, B_size> s;
      for (uint32_t i = 1; i < max_size; i++) {
        s.insert(zip.gen());
        if (i > 1000) {
          sizes.push_back(static_cast<double>(s.get_size()) /
                          s.get_element_count());
        }
        if (i % (max_size / 8) == 0) {
          std::cout << "after " << i << " inserts: number of elements is "
                    << s.get_element_count() << ", size is " << s.get_size()
                    << ", " << s.get_size() / s.get_element_count()
                    << " per element" << std::endl;
        }
      }
      std::cout << "after " << max_size << " inserts: number of elements is "
                << s.get_element_count() << ", size is " << s.get_size() << ", "
                << s.get_size() / s.get_element_count() << " per element"
                << std::endl;
      std::cout << std::endl;
      if (!sizes.empty()) {
        std::cout << "statisics ignoring the first 1000" << std::endl;
        double min_num = std::numeric_limits<double>::max();
        double max_num = 0;
        double sum = 0;
        for (auto x : sizes) {
          min_num = std::min(x, min_num);
          max_num = std::max(x, max_num);
          sum += x;
        }
        std::cout << "min = " << min_num << ", max = " << max_num
                  << ", average = " << sum / (max_size - 1000) << std::endl;
      }
      std::cout << std::endl;
    }
  }
  {
    std::cout << "binomial t=MAX_INT p = .5" << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::binomial_distribution<typename leaf::value_type> distrib(max_size, .5);
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }
  {
    std::cout << "geometrc p = 1/" << max_size << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::geometric_distribution<typename leaf::value_type> distrib(1.0 /
                                                                   max_size);
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }

  {
    std::cout << "poisson_distribution mean = " << max_size << std::endl;
    std::vector<double> sizes;
    std::mt19937 eng(seed);
    std::poisson_distribution<typename leaf::value_type> distrib(max_size);
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i < max_size; i++) {
      s.insert(distrib(eng));
      if (i > 1000) {
        sizes.push_back(static_cast<double>(s.get_size()) /
                        s.get_element_count());
      }
      if (i % (max_size / 8) == 0) {
        std::cout << "after " << i << " inserts: number of elements is "
                  << s.get_element_count() << ", size is " << s.get_size()
                  << ", " << s.get_size() / s.get_element_count()
                  << " per element" << std::endl;
      }
    }
    std::cout << "after " << max_size << " inserts: number of elements is "
              << s.get_element_count() << ", size is " << s.get_size() << ", "
              << s.get_size() / s.get_element_count() << " per element"
              << std::endl;
    std::cout << std::endl;
    if (!sizes.empty()) {
      std::cout << "statisics ignoring the first 1000" << std::endl;
      double min_num = std::numeric_limits<double>::max();
      double max_num = 0;
      double sum = 0;
      for (auto x : sizes) {
        min_num = std::min(x, min_num);
        max_num = std::max(x, max_num);
        sum += x;
      }
      std::cout << "min = " << min_num << ", max = " << max_num
                << ", average = " << sum / (max_size - 1000) << std::endl;
    }
    std::cout << std::endl;
  }
}

// template <typename leaf>
void test_cpma_size_file_out_from_data( // std::vector<uint32_t> data,
    const std::vector<std::string> &filenames, char *outfilename) {
  char pma_outfilename[80];
  strcpy(pma_outfilename, "pma_");
  strcat(pma_outfilename, outfilename);

  char cpma_outfilename[80];
  strcpy(cpma_outfilename, "cpma_");
  strcat(cpma_outfilename, outfilename);

  /*
  size_t max_size = data.size();
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  */

  std::vector<std::string> header;
  std::vector<std::vector<double>> uncompressed_sizes(filenames.size());
  std::vector<std::vector<double>> compressed_sizes(filenames.size());

  int idx = 0;
  for (const auto &filename : filenames) {
    auto data = get_data_from_file<uint32_t>(filename);
    header.push_back(filename);

    cout << filename << endl;

    uncompressed_sizes[idx].push_back(0);
    compressed_sizes[idx].push_back(0);

    CPMA<uncompressed_leaf<uint32_t>, Linear> uncompressed;
    CPMA<delta_compressed_leaf<uint32_t>, Linear> compressed;

    // CPMA<leaf> s;
    for (uint32_t i = 0; i < data.size(); i++) {
      uncompressed.insert(data[i]);
      compressed.insert(data[i]);

      double uncompressed_ratio =
          (static_cast<double>(uncompressed.get_size()) / (i + 1));
      double compressed_ratio =
          (static_cast<double>(compressed.get_size()) / (i + 1));
      printf(
          "i %u, uncompressed size %zu, ratio %f, compressed size %zu, ratio "
          "%f\n",
          i, uncompressed.get_size(), uncompressed_ratio, compressed.get_size(),
          compressed_ratio);

      uncompressed_sizes[idx].push_back(
          static_cast<double>(uncompressed.get_size()) / (i + 1));
      compressed_sizes[idx].push_back(
          static_cast<double>(compressed.get_size()) / (i + 1));
    }
    idx++;
  }

  ofstream myfile;
  myfile.open(pma_outfilename, std::ios_base::app);

  ofstream c_file;
  c_file.open(cpma_outfilename, std::ios_base::app);

  cout << "pma out " << pma_outfilename << ", cpma out " << cpma_outfilename
       << endl;
  const char delim = ',';
  myfile << Join(header, delim) << std::endl;
  c_file << Join(header, delim) << std::endl;

  for (size_t i = 0; i < uncompressed_sizes[0].size(); i++) {
    std::vector<std::string> stringVec;
    std::vector<std::string> cstringVec;

    for (size_t j = 0; j < filenames.size(); j++) {
      if (i < uncompressed_sizes[j].size()) {
        stringVec.push_back(std::to_string(uncompressed_sizes[j][i]));
        cstringVec.push_back(std::to_string(compressed_sizes[j][i]));
      } else {
        stringVec.push_back(std::to_string(0));
        cstringVec.push_back(std::to_string(0));
      }
    }
    myfile << Join(stringVec, delim) << std::endl;
    c_file << Join(cstringVec, delim) << std::endl;
  }
  myfile.close();
  c_file.close();
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_size_file_out(uint64_t max_size, std::seed_seq &seed,
                             const std::string &filename) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  std::vector<std::string> header;
  std::vector<std::vector<double>> sizes(max_size + 1);
  {
    header.emplace_back("sequence");
    sizes[0].push_back(0);
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(i);
      sizes[i].push_back(static_cast<double>(s.get_size()) / i);
    }
    std::cout << "the sum of the numbers from 1 to " << max_size << " is "
              << s.sum() << std::endl;
  }
  {
    header.emplace_back("uniform_random");
    std::vector<typename leaf::value_type> elements =
        create_random_data<typename leaf::value_type>(
            max_size, std::numeric_limits<typename leaf::value_type>::max(),
            seed);
    sizes[0].push_back(0);
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(elements[i - 1]);
      sizes[i].push_back(static_cast<double>(s.get_size()) /
                         s.get_element_count());
    }
  }

  // zipf distributions
  // for (uint32_t max = 1 << 25U; max < 1 << 26; max *= 32) {
  uint32_t max = 1UL << 25U;
  for (double alpha = 1; alpha < 2.1; alpha += 1) {
    zipf zip(max, alpha, seed);
    header.push_back("zipf_" + std::to_string(max) + "_" +
                     std::to_string(alpha));
    std::vector<uint64_t> elements = zip.gen_vector(max_size);
    sizes[0].push_back(0);
    CPMA<leaf, head_form, B_size> s;
    for (uint32_t i = 1; i <= max_size; i++) {
      s.insert(elements[i - 1]);
      sizes[i].push_back(static_cast<double>(s.get_size()) /
                         s.get_element_count());
    }
  }
  // }
  ofstream myfile;
  myfile.open(filename);
  const char delim = ',';
  myfile << Join(header, delim) << std::endl;
  for (const auto &row : sizes) {
    std::vector<std::string> stringVec;
    stringVec.reserve(row.size());
    for (const auto &e : row) {
      stringVec.push_back(std::to_string(e));
    }
    myfile << Join(stringVec, delim) << std::endl;
  }
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_size_file_out_simple(uint64_t max_size, std::seed_seq &seed,
                                    const std::string &filename) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  ofstream myfile;
  myfile.open(filename);

  std::vector<typename leaf::value_type> elements =
      create_random_data_with_seed<typename leaf::value_type>(max_size,
                                                              1UL << 40U, seed);
  CPMA<leaf, head_form, B_size> s;
  uint64_t start = get_usecs();
  std::vector<double> sizes(max_size, 0);
  for (uint32_t i = 1; i <= max_size; i++) {
    s.insert(elements[i - 1]);
    sizes[i] = static_cast<double>(s.get_size()) / s.get_element_count();
  }
  uint64_t end = get_usecs();
  std::cout << "took " << end - start << " micros" << std::endl;
  for (const auto s : sizes) {
    myfile << s << std::endl;
  }
  myfile.close();
}

template <class T>
void test_pma_unordered_insert(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  std::vector<T> data =
      create_random_data<T>(max_size, std::numeric_limits<T>::max(), seed);
  uint64_t start = 0;
  uint64_t end = 0;
  PMA<T, 0> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
    // s.print_pma();
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time with iterator, %lu, sum_total, %lu, ", end - start, sum);
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_unordered_insert(uint64_t max_size, std::seed_seq &seed) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  std::vector<typename leaf::value_type> data =
      create_random_data_with_seed<typename leaf::value_type>(max_size,
                                                              1UL << 40, seed);
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<leaf, head_form, B_size> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
    // std::cout << "inserting " << data[i] << std::endl;
    // s.print_pma();
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu, size per element was %f\n",
         end - start, sum, ((double)s.get_size()) / s.get_element_count());
}

void test_tlx_btree_unordered_insert(uint64_t max_size, std::seed_seq &seed) {
  std::vector<uint64_t> data =
      create_random_data_with_seed<uint64_t>(max_size, 1UL << 40, seed);
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<uint64_t> s;
  start = get_usecs();
  for (uint32_t i = 1; i < max_size; i++) {
    s.insert(data[i]);
    // std::cout << "inserting " << data[i] << std::endl;
    // s.print_pma();
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

void test_btree_ordered_and_unordered_insert(uint64_t max_size,
                                             uint64_t num_ordered,
                                             uint64_t num_random,
                                             std::seed_seq &seed) {
  if (max_size > std::numeric_limits<uint64_t>::max()) {
    max_size = std::numeric_limits<uint64_t>::max();
  }
  std::vector<uint64_t> data =
      create_random_data_with_seed<uint64_t>(max_size, 1UL << 40, seed);
  uint64_t cycle_length = num_random + num_ordered;
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<uint64_t> s;
  start = get_usecs();
  for (int64_t i = max_size - 1; i >= 0; i--) {
    if (i % cycle_length < num_ordered) {
      s.insert(i);
    } else {
      s.insert(data[i] + max_size);
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_ordered_and_unordered_insert(uint64_t max_size,
                                            uint64_t num_ordered,
                                            uint64_t num_random,
                                            std::seed_seq &seed) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  std::vector<typename leaf::value_type> data =
      create_random_data_with_seed<typename leaf::value_type>(max_size,
                                                              1UL << 40, seed);
  uint64_t cycle_length = num_random + num_ordered;
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<leaf, head_form, B_size> s;
  start = get_usecs();
  for (int64_t i = max_size - 1; i >= 0; i--) {
    if (i % cycle_length < num_ordered) {
      s.insert(i);
    } else {
      s.insert(data[i] + max_size);
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

void test_btree_multi_seq_insert(uint64_t max_size, uint64_t groups) {
  if (max_size > std::numeric_limits<uint64_t>::max()) {
    max_size = std::numeric_limits<uint64_t>::max();
  }
  uint64_t elements_per_group = max_size / groups;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<uint64_t> s;
  start = get_usecs();
  for (uint32_t i = 0; i < elements_per_group; i++) {
    for (uint64_t j = 0; j < groups; j++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_multi_seq_insert(uint64_t max_size, uint64_t groups) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  uint64_t elements_per_group = max_size / groups;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<leaf, head_form, B_size> s;
  start = get_usecs();
  for (uint32_t i = 0; i < elements_per_group; i++) {
    for (uint64_t j = 0; j < groups; j++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

void test_btree_bulk_insert(uint64_t max_size, uint64_t num_per) {
  if (max_size > std::numeric_limits<uint64_t>::max()) {
    max_size = std::numeric_limits<uint64_t>::max();
  }
  uint64_t groups = max_size / num_per;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  tlx::btree_set<uint64_t> s;
  start = get_usecs();
  for (uint32_t j = 0; j < groups; j++) {
    for (uint64_t i = 0; i < num_per; i++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  for (const auto e : s) {
    sum += e;
  }
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_bulk_insert(uint64_t max_size, uint64_t num_per) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  uint64_t groups = max_size / num_per;
  std::vector<uint64_t> group_position(groups);
  for (uint64_t i = 0; i < groups; i++) {
    group_position[i] = i;
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::shuffle(group_position.begin(), group_position.end(), eng);
  uint64_t start = 0;
  uint64_t end = 0;
  CPMA<leaf, head_form, B_size> s;
  start = get_usecs();
  for (uint32_t j = 0; j < groups; j++) {
    for (uint64_t i = 0; i < num_per; i++) {
      s.insert(i + (group_position[j] << 30U));
    }
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_unordered_insert_batches(uint64_t max_size, uint64_t batch_size,
                                        std::seed_seq &seed) {
  if (max_size > std::numeric_limits<typename leaf::value_type>::max()) {
    max_size = std::numeric_limits<typename leaf::value_type>::max();
  }
  if (batch_size > max_size) {
    batch_size = max_size;
  }
  std::vector<typename leaf::value_type> data =
      create_random_data<typename leaf::value_type>(
          max_size, std::numeric_limits<typename leaf::value_type>::max(),
          seed);
  CPMA<leaf, head_form, B_size> s;
  uint64_t start = get_usecs();
  for (uint32_t i = 1; i < max_size; i += batch_size) {
    if (i + batch_size > max_size) {
      batch_size = max_size - i;
    }
    s.insert_batch(data.data() + i, batch_size);
    // std::cout << "inserting " << data[i] << std::endl;
    // s.print_pma();
  }
  uint64_t end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  uint64_t sum = 0;
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu, size per element was %f\n",
         end - start, sum, ((double)s.get_size()) / s.get_element_count());
}

// take data as a vector
template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_unordered_insert_batches_from_data(
    uint64_t batch_size, std::vector<typename leaf::value_type> data,
    int num_trials, char *filename, char *outfilename) {
  uint64_t max_size = data.size();

  if (batch_size > max_size) {
    batch_size = max_size;
  }

#ifdef VERIFY
  std::unordered_set<typename leaf::value_type> correct;
  for (auto i : data) {
    correct.insert(i);
  }
  uint64_t correct_sum = 0;
  for (auto e : correct) {
    correct_sum += e;
  }
#endif

  uint64_t temp_batch_size = 0;
  uint64_t total_insert_time = 0;
  uint64_t total_delete_time = 0;
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t batch_time = 0;
  for (int i = 0; i < num_trials; i++) {
    CPMA<leaf, head_form, B_size> s;
    start = get_usecs();
    if (batch_size > 1) {
      temp_batch_size = batch_size;
      // do batch inserts
      for (uint64_t i = 0; i < max_size; i += batch_size) {
        if (i + batch_size > max_size) {
          temp_batch_size = max_size % batch_size;
        }
        s.insert_batch(data.data() + i, temp_batch_size);
        // std::cout << "the pma has " << s.get_element_count()
        //           << " unique elements so far" << endl;
        // std::cout << "inserting " << data[i] << std::endl;
        // s.print_pma();
      }
    } else {
      for (uint64_t i = 0; i < max_size; i++) {
        s.insert(data[i]);
      }
    }
    end = get_usecs();
    total_insert_time += end - start;

    start = get_usecs();
    if (batch_size > 1) {
      // do batch deletes
      temp_batch_size = batch_size;
      for (uint64_t i = 0; i < max_size; i += batch_size) {
        if (i + batch_size > max_size) {
          temp_batch_size = max_size % batch_size;
        }
        s.remove_batch(data.data() + i, temp_batch_size);
        // std::cout << "inserting " << data[i] << std::endl;
        // s.print_pma();
      }
    } else {
      for (uint64_t i = 0; i < max_size; i++) {
        s.remove(data[i]);
      }
    }
    end = get_usecs();
    batch_time = end - start;
    // cout << "delete batch time: " << batch_time << endl;
    total_delete_time += batch_time;
  }
  double avg_insert = ((double)total_insert_time / 1000000) / num_trials;
  double avg_delete = ((double)total_delete_time / 1000000) / num_trials;
  // cout << "avg insert: " << avg_insert << endl;
  //  cout << "avg delete: " << avg_delete << endl;

  // do sum
  CPMA<leaf, head_form, B_size> s;
  s.insert_batch(data.data(), data.size());

  num_trials *= 5; // do more trials for sum
  uint64_t total_time = 0;

  for (int i = 0; i < num_trials; i++) {
    uint64_t sum = 0;
    start = get_usecs();
    sum = s.sum();
    end = get_usecs();
    total_time += end - start;
#ifdef VERIFY
    assert(correct_sum == sum);
    if (i == 0) {
      cout << "got sum: " << sum << " expected sum: " << correct_sum << endl;
    }
#endif
  }
  double avg_sum = ((double)total_time / 1000000) / num_trials;

  auto size = s.get_size();

  // write out to file
  if (filename != nullptr) {
    std::ofstream outfile;
    outfile.open(outfilename, std::ios_base::app);
    outfile << filename << "," << batch_size << "," << avg_insert << ","
            << avg_delete << "," << avg_sum << "," << size << ","
            << ParallelTools::getWorkers() << endl;
    outfile.close();
  } else {
    std::cout << std::setprecision(2) << std::setw(12) << std::setfill(' ')
              << max_size << ", " << std::setw(12) << std::setfill(' ')
              << batch_size << ", " << std::setw(12) << std::setfill(' ')
              << avg_insert << ", " << std::setw(12) << std::setfill(' ')
              << avg_delete << ", " << std::setw(12) << std::setfill(' ')
              << avg_sum << ", " << size << endl;
  }
  // printf("sum_time, %lu, correct sum %lu, sum_total, %lu\n", end - start,
  // correct_sum, sum);
}

template <class T> void pma_micro_benchmark(uint64_t n, std::seed_seq &seed) {
  printf("pma:   n, %lu,\t", n);
  std::vector<T> data =
      create_random_data<T>(n, std::numeric_limits<T>::max(), seed);
  uint64_t start = 0;
  uint64_t end = 0;
  PMA<T, 0> s;
  start = get_usecs();
  for (uint32_t i = 1; i < n; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  std::mt19937 eng(seed); // a source of random data
  std::shuffle(data.begin(), data.end(), eng);
  bool has_all = true;
  start = get_usecs();
  for (uint32_t i = 1; i < n / 10; i++) {
    has_all &= s.has(data[i]);
  }
  end = get_usecs();
  if (!has_all) {
    std::cerr << "lost something";
  }
  printf("get, \t %lu,", end - start);

  start = get_usecs();
  uint64_t sum = 0;
  for (auto el : s) {
    sum += el;
  }
  end = get_usecs();
  printf("sum_time with iterator, %lu, sum_total, %lu, ", end - start, sum);
  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <class T> void btree_micro_benchmark(uint64_t n, std::seed_seq &seed) {
  printf("btree: n,  %lu,\t", n);
  std::vector<T> data =
      create_random_data_with_seed<T>(n, std::numeric_limits<T>::max(), seed);
  uint64_t start = 0;
  uint64_t end = 0;
  BTree<T, T> s;
  start = get_usecs();
  for (uint32_t i = 1; i < n; i++) {
    s.insert(data[i]);
  }
  end = get_usecs();
  printf("insertion, \t %lu,", end - start);
  std::mt19937 eng(seed); // a source of random data
  std::shuffle(data.begin(), data.end(), eng);
  bool has_all = true;
  start = get_usecs();
  for (uint32_t i = 1; i < n / 10; i++) {
    has_all &= (s.find(data[i]) != nullptr);
  }
  end = get_usecs();
  if (!has_all) {
    std::cerr << "lost something";
  }
  printf("get, \t %lu,", end - start);

  start = get_usecs();
  uint64_t sum = 0;
  auto it = s.begin();
  while (!it.done()) {
    T el = *it;
    sum += el;
    ++it;
  }
  end = get_usecs();
  printf("\tsum_time with iterator, \t%lu, \tsum_total, \t%lu, \t", end - start,
         sum);

  start = get_usecs();
  sum = s.sum();
  end = get_usecs();
  printf("sum_time, %lu, sum_total, %lu\n", end - start, sum);
}

template <class T> bool verify_pma(uint64_t number_trials, bool fast = false) {
  uint64_t max_num = 100000;
  if (max_num > std::numeric_limits<T>::max()) {
    max_num = std::numeric_limits<T>::max() - 1;
  }
  std::set<T> correct;
  PMA<T, 0> test;
  std::seed_seq seed{0};
  auto random_numbers = create_random_data<T>(number_trials, max_num, seed);
  for (uint64_t i = 0; i < number_trials; i++) {
    T x = random_numbers[i] + 1;
    // std::cout << "inserting: " << x << std::endl;
    // test.print_pma();
    correct.insert(x);
    test.insert(x);
    if (!fast) {
      if (!test.check_sorted()) {
        test.print_array();
        std::cout << "not sorted, just inserted " << x << std::endl;
        test.print_array();
        test.print_pma();
        return true;
      }
      if (!test.has(x)) {
        std::cout << "not there after insert, missing " << x << std::endl;
        test.print_array();
        test.print_pma();
        return true;
      }
      for (auto e : correct) {
        if (!test.has(e)) {
          std::cout << "lost something from earlier, missing " << e
                    << std::endl;
          test.print_array();
          test.print_pma();
          return true;
        }
      }
    }
  }
  // std::cout << "checking after inserts" << std::endl;
  if (!test.check_sorted()) {
    test.print_array();
    std::cout << "not sorted after inserts" << std::endl;
    test.print_array();
    test.print_pma();
    return true;
  }
  for (auto e : correct) {
    if (!test.has(e)) {
      std::cout << "after inserts missing " << e << std::endl;
      test.print_array();
      test.print_pma();
      return true;
    }
  }
  for (auto e : test) {
    if (!correct.count(e)) {
      std::cout << "after inserts has " << e << std::endl;
      test.print_array();
      test.print_pma();
      return true;
    }
  }
  random_numbers = create_random_data<T>(number_trials, max_num, seed);
  for (uint64_t i = 0; i < number_trials; i++) {
    T x = random_numbers[i] + 1;
    // std::cout << "removing: " << x << std::endl;
    correct.erase(x);
    test.remove(x);
    if (!fast) {
      if (!test.check_sorted()) {
        test.print_array();
        std::cout << "not sorted, just inserted " << x << std::endl;
        test.print_array();
        test.print_pma();
        return true;
      }
      if (test.has(x)) {
        std::cout << "there after removing, has " << x << std::endl;
        test.print_array();
        test.print_pma();
        return true;
      }
      for (auto e : correct) {
        if (!test.has(e)) {
          std::cout
              << "lost something thats in correct during deletes, missing " << e
              << std::endl;
          test.print_array();
          test.print_pma();
          return true;
        }
      }
    }
  }
  // std::cout << "checking after deletes" << std::endl;
  if (!test.check_sorted()) {
    test.print_array();
    std::cout << "not sorted after deletes" << std::endl;
    test.print_array();
    test.print_pma();
    return true;
  }
  for (auto e : correct) {
    if (!test.has(e)) {
      std::cout << "after deletes missing " << e << std::endl;
      test.print_array();
      test.print_pma();
      return true;
    }
  }
  for (auto e : test) {
    if (!correct.count(e)) {
      std::cout << "after deletes has " << e << std::endl;
      test.print_array();
      test.print_pma();
      return true;
    }
  }
  return false;
}

template <class pma_type, class set_type>
bool pma_different_from_set(const pma_type &pma, const set_type &set) {
  // check that the right data is in the set
  uint64_t correct_sum = 0;
  for (auto e : set) {
    correct_sum += e;
    if (!pma.has(e)) {
      printf("pma missing %lu\n", (uint64_t)e);
      return true;
    }
  }
  bool have_something_wrong = pma.map<false>([&set](uint64_t element) {
    if (set.find(element) == set.end()) {
      printf("have something (%lu) that the set doesn't have\n",
             (uint64_t)element);

      return true;
    }
    return false;
  });
  if (have_something_wrong) {
    printf("pma has something is shouldn't\n");
    return true;
  }
  if (correct_sum != pma.sum()) {
    printf("pma has bad sum\n");
    return true;
  }
  return false;
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
bool verify_cpma(uint64_t number_trials, bool fast = false) {
  using T = typename leaf::value_type;
  uint64_t max_num = std::numeric_limits<T>::max() - 1;
  if (max_num > std::numeric_limits<T>::max()) {
    max_num = std::numeric_limits<T>::max() - 1;
  }
  {
    CPMA<leaf, head_form, B_size> t;
    uint64_t sum = 0;
    for (uint64_t i = 1; i < number_trials; i += 1) {
      // t.print_pma();
      t.insert(i);
      sum += i;
      if (sum != t.sum()) {
        std::cout << "bad sum after inserting sequntial numbers" << std::endl;
        std::cout << "got " << t.sum() << " expected " << sum << std::endl;
        std::cout << "just inserted " << i << std::endl;
        t.print_pma();
        return true;
      }
    }
  }
  tlx::btree_set<T> correct;
  CPMA<leaf, head_form, B_size> test;
  // test.print_pma();
  auto random_numbers = create_random_data<T>(number_trials, max_num);
  for (uint64_t i = 0; i < number_trials; i++) {
    T x = random_numbers[i] + 1;
    // if (test.compressed) {
    //   test.print_pma();
    //   std::cout << "inserting: " << x << std::endl;
    // }
    correct.insert(x);
    test.insert(x);
    assert(test.check_nothing_full());
    if (!fast) {
      if (pma_different_from_set(test, correct)) {
        printf("issue during inserts\n");
        return true;
      }
    }
  }
  if (pma_different_from_set(test, correct)) {
    printf("issue after inserts\n");
    return true;
  }
  random_numbers = create_random_data<T>(number_trials, max_num);
  for (uint64_t i = 0; i < number_trials; i++) {
    T x = random_numbers[i] + 1;
    // test.print_pma();
    // std::cout << "removing: " << x << std::endl;
    correct.erase(x);
    test.remove(x);
    if (!fast) {
      if (pma_different_from_set(test, correct)) {
        printf("issue during deletes\n");
        return true;
      }
    }
  }

  if (pma_different_from_set(test, correct)) {
    printf("issue after deletes\n");
    return true;
  }
  // test.print_pma();
  uint64_t num_rounds = 10;
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<T> batch;
    auto random_numbers =
        create_random_data<T>(number_trials / num_rounds, max_num);
    for (uint64_t i = 0; i < number_trials / num_rounds; i++) {
      T x = random_numbers[i] + 1;
      batch.push_back(x);
      correct.insert(x);
    }

    // printf("before insert\n");
    // test.print_pma();
    // // test.print_array();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try inserting batch
    test.insert_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (auto e : batch) {
      if (!test.has(e)) {
        std::cout << "missing something in batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    // everything in correct has to be in test
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
      if (!test.has(e)) {
        std::cout << "missing something not in batch " << e << std::endl;
        // test.print_array();
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    bool have_something_wrong = test.map<false>([&correct](T element) {
      if (correct.find(element) == correct.end()) {
        printf("have something (%lu) that the set doesn't have\n",
               (uint64_t)element);

        return true;
      }
      return false;
    });
    if (have_something_wrong) {
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
    }

    // sum
    if (test.sum() != correct_sum) {
      std::cout << "sum got " << test.sum() << ", should be " << correct_sum
                << std::endl;
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
    }
  }

  // random batch
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<T> batch;
    auto random_numbers =
        create_random_data<T>(number_trials / num_rounds, max_num);
    for (uint64_t i = 0; i < number_trials / num_rounds; i++) {
      T x = random_numbers[i] + 1;
      batch.push_back(x);
      correct.erase(x);
    }

    // printf("before remove\n");
    // test.print_pma();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try removing batch
    test.remove_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (auto e : batch) {
      if (test.has(e)) {
        std::cout << "has something in random batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    // everything in correct has to be in test
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
      if (!test.has(e)) {
        std::cout << "missing something not in random batch after deletes " << e
                  << std::endl;
        // test.print_array();
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }
    bool have_something_wrong = test.map<false>([&correct](T element) {
      if (correct.find(element) == correct.end()) {
        printf("have something (%lu) that the set doesn't have after random "
               "batch deletes\n",
               (uint64_t)element);

        return true;
      }
      return false;
    });
    if (have_something_wrong) {
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
    }

    // sum
    if (test.sum() != correct_sum) {
      std::cout << "random batch: sum got " << test.sum() << ", should be "
                << correct_sum << std::endl;
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
    }
  }
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<T> dist(0, correct.size());
  // batch of elements
  for (uint64_t round = 0; round < num_rounds; round++) {
    // put stuff into the pma
    std::vector<T> batch;
    for (uint64_t i = 0; i < number_trials / num_rounds; i++) {
      if (correct.empty()) {
        break;
      }
      // this is quite slow, but will generate random elements from the set
      uint32_t steps = dist(eng);
      auto it = std::begin(correct);
      std::advance(it, steps);
      if (it == std::end(correct)) {
        continue;
      }
      batch.push_back(*it);
      correct.erase(*it);
    }

    // printf("before insert\n");
    // test.print_pma();
    // test.print_array();

    // std::sort(batch.begin(), batch.end());
    // printf("\n*** BATCH %lu ***\n", round);
    // for (auto elt : batch) {
    //   std::cout << elt << ", ";
    // }
    // std::cout << std::endl;

    // try inserting batch
    test.remove_batch(batch.data(), batch.size());

    // everything in batch has to be in test
    for (auto e : batch) {
      if (test.has(e)) {
        std::cout << "has something in batch " << e << std::endl;
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    // everything in correct has to be in test
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
      if (!test.has(e)) {
        std::cout << "batch deletes: missing something not in batch " << e
                  << std::endl;
        // test.print_array();
        test.print_pma();

        printf("\n*** BATCH ***\n");
        std::sort(batch.begin(), batch.end());
        for (auto elt : batch) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;

        printf("\n*** CORRECT ***\n");
        for (auto elt : correct) {
          std::cout << elt << ", ";
        }
        std::cout << std::endl;
        return true;
      }
    }

    bool have_something_wrong = test.map<false>([&correct](T element) {
      if (correct.find(element) == correct.end()) {
        printf("have something (%lu) that the set doesn't have after "
               "batch deletes\n",
               (uint64_t)element);

        return true;
      }
      return false;
    });
    if (have_something_wrong) {
      test.print_pma();
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
    }

    // sum
    if (test.sum() != correct_sum) {
      std::cout << "after deletes: sum got " << test.sum() << ", should be "
                << correct_sum << std::endl;
      test.print_pma();
      test.map<true>([](T element) { std::cout << element << ", "; });
      printf("\n*** CORRECT ***\n");
      for (auto elt : correct) {
        std::cout << elt << ", ";
      }
      std::cout << std::endl;
      return true;
    }
  }

  return false;
}

template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
bool verify_cpma_different_sizes(
    const std::vector<std::pair<uint64_t, bool>> &args) {
  for (auto arg : args) {
    printf("testing size %lu, fast = %d\n", std::get<0>(arg), std::get<1>(arg));
    if (verify_cpma<leaf, head_form, B_size>(std::get<0>(arg),
                                             std::get<1>(arg))) {
      return true;
    }
  }
  return false;
}

template <typename leaf_type>
bool verify_leaf(uint32_t size, uint32_t num_ops, uint32_t range_start,
                 uint32_t range_end, uint32_t print = 0,
                 uint32_t bit_mask = 0) {
  static constexpr bool head_in_place = false;
  assert(size % 32 == 0);
  std::cout << "testing leaf size=" << size << ", num_ops=" << num_ops
            << ", range_start=" << range_start << ", range_end=" << range_end
            << ", bit_mask = " << bit_mask << "\n";
  uint8_t *array = (uint8_t *)memalign(32, size);
  typename leaf_type::value_type head = 0;
  for (uint32_t i = 0; i < size; i++) {
    array[i] = 0;
  }
  leaf_type leaf(head, array, size);

  // uncompressed_leaf<uint32_t> leaf(array, size);
  std::set<typename leaf_type::value_type> correct;
  auto random_numbers = create_random_data<typename leaf_type::value_type>(
      num_ops, range_end - range_start);
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<uint32_t> dist(0, 1);
  for (uint32_t i = 0; i < num_ops; i++) {
    uint32_t random_number = random_numbers[i];
    if (bit_mask) {
      random_number &= ~bit_mask;
    };
    uint32_t num = range_start + random_number;
    if (dist(eng) == 0) {
      if (print) {
        std::cout << "inserting " << num << std::endl;
      }
      leaf.template insert<head_in_place>(num);
      correct.insert(num);
    } else {
      if (print) {
        std::cout << "removing " << num << std::endl;
      }
      leaf.template remove<head_in_place>(num);
      correct.erase(num);
    }
    if (print > 1) {
      leaf.print();
    }

    bool leaf_missing = false;
    for (auto e : correct) {
      if (leaf.contains(e) != true) {
        std::cout << "leaf is missing " << e << std::endl;
        leaf_missing = true;
      }
    }
    if (leaf_missing) {
      std::cout << "correct has: ";
      for (auto e : correct) {
        std::cout << e << ", ";
      }
      std::cout << std::endl;
      leaf.print();
      free(array);
      return true;
    }
    uint64_t correct_sum = 0;
    for (auto e : correct) {
      correct_sum += e;
    }
    uint64_t leaf_sum = leaf.sum();
    if (correct_sum != leaf_sum) {
      std::cout << "the sums do not match, got " << leaf_sum << ", expected "
                << correct_sum << std::endl;
      std::cout << "correct has: ";
      for (auto e : correct) {
        std::cout << e << ", ";
      }
      std::cout << std::endl;
      leaf.print();
      free(array);
      return true;
    }
  }

  free(array);
  int num_leaves = 16;
  array = (uint8_t *)memalign(32, size * num_leaves);
  for (uint32_t i = 0; i < size * num_leaves; i++) {
    array[i] = 0;
  }
  uint32_t elts_per_leaf = size / 10;
  if (elts_per_leaf > num_ops) {
    elts_per_leaf = num_ops;
  }

  std::set<typename leaf_type::value_type> inputs;
  std::uniform_int_distribution<typename leaf_type::value_type> dist_1000(0,
                                                                          1000);
  while (inputs.size() < elts_per_leaf) {
    inputs.insert(dist_1000(eng));
  }
  // std::sort(inputs.begin(), inputs.end());

  // printf("inputs = {");

  uint64_t sum = 0;
  std::vector<typename leaf_type::value_type> heads(num_leaves);
  for (int i = 0; i < num_leaves; i++) {
    leaf_type leaf(heads[i], array + (i * size), size);

    for (uint32_t elt : inputs) {
      sum += 10000 * i + elt;
      leaf.template insert<head_in_place>(10000 * i + elt);
      // printf("%u, ", 10000 * i + elt);
    }
    if (print > 1) {
      leaf.print();
    }
  }
  // printf("}\n");

  auto result = leaf_type::template merge<head_in_place, false>(
      (typename leaf_type::value_type *)array, num_leaves, size, 0,
      [&heads](size_t idx) ->
      typename leaf_type::value_type & { return heads[idx]; },
      nullptr);
  free(array);
  if (result.first.sum() != sum) {
    printf("MERGE got sum %lu, should be %lu\n", result.first.sum(), sum);
    result.first.print();
    free(reinterpret_cast<uint8_t *>(result.first.array) -
         sizeof(typename leaf_type::value_type));
    return true;
  }
  // free(result.first.array);

  // testing split

  uint8_t *dest_array = (uint8_t *)malloc(size * num_leaves);
  if (print > 1) {
    result.first.print();
  }
  result.first.template split<head_in_place, false>(
      num_leaves, result.second, size,
      (typename leaf_type::value_type *)dest_array, 0,
      [&heads](size_t idx) ->
      typename leaf_type::value_type & { return heads[idx]; },
      nullptr);
  uint64_t total_sum = 0;
  for (int i = 0; i < num_leaves; i++) {
    leaf_type leaf(heads[i], dest_array + (i * size), size);
    if (print > 1) {
      leaf.print();
    }
    total_sum += leaf.sum();
  }
  free(dest_array);
  free(reinterpret_cast<uint8_t *>(result.first.array) -
       sizeof(typename leaf_type::value_type));

  if (total_sum != sum) {
    printf("SPLIT got sum %lu, should be %lu\n", total_sum, sum);
    return true;
  }
  return false;
}

template <class leaf>
bool time_leaf(uint32_t num_added_per_leaf, uint32_t num_leafs,
               uint32_t range_start, uint32_t range_end) {
  num_leafs =
      (num_leafs / ParallelTools::getWorkers()) * ParallelTools::getWorkers();
  std::cout << "timing leaf num_added_per_leaf=" << num_added_per_leaf
            << ", num_leafs=" << num_leafs << ", range_start=" << range_start
            << ", range_end=" << range_end << std::endl;
  size_t size_per_leaf = 5 * num_added_per_leaf;
  size_t num_cells = size_per_leaf * num_leafs;

  uint8_t *array = (uint8_t *)malloc(num_cells);
  if (!array) {
    std::cerr << "bad alloc array" << std::endl;
    return true;
  }

  ParallelTools::parallel_for(0, num_cells, [&](uint32_t i) { array[i] = 0; });
  auto random_numbers = create_random_data<uint32_t>(
      num_added_per_leaf * num_leafs, range_end - range_start);
  for (uint32_t i = 0; i < num_added_per_leaf * num_leafs; i++) {
    random_numbers[i] += range_start;
  }
  {
    timer insert_timer("inserts");
    insert_timer.start();
    ParallelTools::parallel_for(0, ParallelTools::getWorkers(), [&](int j) {
      for (uint32_t i = 0; i < num_added_per_leaf; i++) {
        for (uint32_t k = 0; k < num_leafs / ParallelTools::getWorkers(); k++) {
          uint32_t which_leaf =
              j * (num_leafs / ParallelTools::getWorkers()) + k;
          leaf l(array + (which_leaf * size_per_leaf), size_per_leaf);
          l.insert(random_numbers[i * num_leafs + which_leaf]);
        }
      }
    });
    insert_timer.stop();
  }
  timer sum_timer("sum");
  sum_timer.start();

  ParallelTools::Reducer_sum<uint64_t> counts;
  ParallelTools::parallel_for(0, num_leafs, [&](uint32_t j) {
    leaf l(array + (j * size_per_leaf), size_per_leaf);
    counts.add(l.sum());
  });
  uint64_t total_sum = counts.get();
  sum_timer.stop();
  uint64_t total_size = 0;
  for (uint32_t j = 0; j < num_leafs; j++) {
    leaf l(array + (j * size_per_leaf), size_per_leaf);
    total_size += l.used_size();
  }

  std::cout << "the total leaf sum is " << total_sum << std::endl;
  std::cout << "the average leaf size is " << total_size / num_leafs
            << std::endl;
  free(array);
  return false;
}

// read in file and time the pma
void timing_pma_from_data(uint64_t batch_size, char *filename,
                          char *outfilename, int num_trials) {
  // read from file
  // return pair of array and len

  char pma_outfilename[80];
  strcpy(pma_outfilename, outfilename);
  strcat(pma_outfilename, "_pma");

  char cpma_outfilename[80];
  strcpy(cpma_outfilename, outfilename);
  strcat(cpma_outfilename, "_cpma");

  cout << "pma out: " << pma_outfilename << endl;
  cout << "cpma out: " << cpma_outfilename << endl;
  if (true || std::string(filename).ends_with("twitter_max")) {
    auto data = get_data_from_file<uint32_t>(filename);
    printf("inserting %lu elements in batch size %lu from file %s\n",
           data.size(), batch_size, filename);

    printf("unordered cpma<uncompressed_leaf<uint32_t>> batch_size %lu, num "
           "elts "
           "%lu\t\t",
           batch_size, data.size());
    test_cpma_unordered_insert_batches_from_data<uncompressed_leaf<uint32_t>,
                                                 Linear>(
        batch_size, data, num_trials, filename, pma_outfilename);
    std::cout << std::endl;

    printf(
        "unordered cpma<delta_compressed_leaf<uint32_t>> batch_size %lu, num "
        "elts %lu\t\t",
        batch_size, data.size());
    test_cpma_unordered_insert_batches_from_data<
        delta_compressed_leaf<uint32_t>, Linear>(batch_size, data, num_trials,
                                                 filename, cpma_outfilename);
    std::cout << std::endl;

  } else {
    auto data = get_data_from_file<uint64_t>(filename);
    printf("inserting %lu elements in batch size %lu from file %s\n",
           data.size(), batch_size, filename);

    printf("unordered cpma<uncompressed_leaf<uint64_t>> batch_size %lu, num "
           "elts "
           "%lu\t\t",
           batch_size, data.size());
    test_cpma_unordered_insert_batches_from_data<uncompressed_leaf<uint64_t>,
                                                 Linear>(
        batch_size, data, num_trials, filename, pma_outfilename);
    std::cout << std::endl;

    printf(
        "unordered cpma<delta_compressed_leaf<uint64_t>> batch_size %lu, num "
        "elts %lu\t\t",
        batch_size, data.size());
    test_cpma_unordered_insert_batches_from_data<
        delta_compressed_leaf<uint64_t>, Linear>(batch_size, data, num_trials,
                                                 filename, cpma_outfilename);
    std::cout << std::endl;
  }
}

// take data as a vector
template <typename leaf, HeadForm head_form, uint64_t B_size = 0>
void test_cpma_scalability_from_data(
    std::vector<typename leaf::value_type> data, int num_trials, char *filename,
    char *outfilename, int serial) {
  uint64_t total_insert_time = 0;
  uint64_t total_sum_time = 0;
  uint64_t start = 0;
  uint64_t end = 0;
  for (int i = 0; i < num_trials; i++) {
    CPMA<leaf, head_form, B_size> s;
    start = get_usecs();
    s.insert_batch(data.data(), data.size());
    end = get_usecs();
    total_insert_time += end - start;

    start = get_usecs();
    auto sum = s.sum();
    end = get_usecs();
    total_sum_time += end - start;
    cout << "got sum " << sum << endl;
  }
  double avg_insert = ((double)total_insert_time / 1000000) / num_trials;
  double avg_sum = ((double)total_sum_time / 1000000) / num_trials;

  // write out to file
  std::ofstream outfile;
  outfile.open(outfilename, std::ios_base::app);
  if (serial) {
    outfile << "," << avg_insert << "," << avg_sum << endl;
  } else {
    outfile << filename << "," << avg_insert << "," << avg_sum;
  }
  outfile.close();
  // printf("sum_time, %lu, correct sum %lu, sum_total, %lu\n", end - start,
  // correct_sum, sum);
}

// read in file and time the pma
void pma_scalability_from_data(char *filename, char *outfilename,
                               int num_trials, int serial) {
  // read from file
  // return pair of array and len
  auto data = get_data_from_file<uint32_t>(filename);

  char pma_outfilename[80];
  strcpy(pma_outfilename, "pma_");
  strcat(pma_outfilename, outfilename);

  char cpma_outfilename[80];
  strcpy(cpma_outfilename, "cpma_");
  strcat(cpma_outfilename, outfilename);

  test_cpma_scalability_from_data<uncompressed_leaf<uint32_t>, Linear>(
      data, num_trials, filename, pma_outfilename, serial);
  std::cout << std::endl;

  test_cpma_scalability_from_data<delta_compressed_leaf<uint32_t>, Linear>(
      data, num_trials, filename, cpma_outfilename, serial);
  std::cout << std::endl;
}

template <class T>
void timing_cpma_helper(uint64_t max_size, uint64_t start_batch_size,
                        uint64_t end_batch_size, uint64_t num_trials,
                        std::seed_seq &seed) {
  if (max_size > std::numeric_limits<T>::max()) {
    max_size = std::numeric_limits<T>::max();
  }
  std::vector<T> data = create_random_data_with_seed<T>(
      max_size, std::numeric_limits<T>::max(), seed);
  std::cout << "uncompressed:" << std::endl;
  std::cout << std::setw(12) << std::setfill(' ') << "max_size"
            << ", " << std::setw(12) << std::setfill(' ') << "batch_size"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_insert"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_delete"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_sum"
            << ", "
            << "size" << std::endl;
  for (uint64_t batch_size = start_batch_size; batch_size < end_batch_size;
       batch_size *= 10) {
    test_cpma_unordered_insert_batches_from_data<uncompressed_leaf<T>, Linear>(
        batch_size, data, num_trials, nullptr, nullptr);
  }

  std::cout << "compressed:" << std::endl;
  std::cout << std::setw(12) << std::setfill(' ') << "max_size"
            << ", " << std::setw(12) << std::setfill(' ') << "batch_size"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_insert"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_delete"
            << ", " << std::setw(12) << std::setfill(' ') << "avg_sum"
            << ", "
            << "size" << std::endl;
  for (uint64_t batch_size = start_batch_size; batch_size < end_batch_size;
       batch_size *= 10) {
    test_cpma_unordered_insert_batches_from_data<delta_compressed_leaf<T>,
                                                 Linear>(
        batch_size, data, num_trials, nullptr, nullptr);
  }
}

void timing_inserts(uint64_t max_size, bool pma, bool btree, bool sets,
                    bool cpma) {
  std::random_device r;
  std::seed_seq seed{0};
  if (sets) {

    printf("ordered std::set<64>,\t\t");
    test_set_ordered_insert<uint64_t>(max_size);
    printf("ordered std::set<32>,\t\t");
    test_set_ordered_insert<uint32_t>(max_size);
    // printf("ordered std::set, b, 16,");
    // test_set_ordered_insert<uint16_t>(max_size);
    // printf("ordered std::set, b, 8,");
    // test_set_ordered_insert<uint8_t>(max_size);
    std::cout << std::endl;

    printf("unordered std::set<64>,\t\t");
    test_set_unordered_insert<uint64_t>(max_size, seed);
    printf("unordered std::set<32>,\t\t");
    test_set_unordered_insert<uint32_t>(max_size, seed);
    // printf("unordered std::set, b, 16,");
    // test_set_unordered_insert<uint16_t>(max_size, seed);
    // printf("unordered std::set, b, 8,");
    // test_set_unordered_insert<uint8_t>(max_size, seed);
    std::cout << std::endl;

    printf("ordered std::unordered_set<64>,\t");
    test_unordered_set_ordered_insert<uint64_t>(max_size);
    printf("ordered std::unordered_set<32>,\t");
    test_unordered_set_ordered_insert<uint32_t>(max_size);
    // printf("ordered std::unordered_set, b, 16,");
    // test_unordered_set_ordered_insert<uint16_t>(max_size);
    // printf("ordered std::unordered_set, b, 8,");
    // test_unordered_set_ordered_insert<uint8_t>(max_size);
    std::cout << std::endl;

    printf("unordered unordered_set<64>,\t");
    test_unordered_set_unordered_insert<uint64_t>(max_size, seed);
    printf("unordered unordered_set<32>,\t");
    test_unordered_set_unordered_insert<uint32_t>(max_size, seed);
    // printf("unordered std::unordered_set, b, 16,");
    // test_unordered_set_unordered_insert<uint16_t>(max_size, seed);
    // printf("unordered std::unordered_set, b, 8,");
    // test_unordered_set_unordered_insert<uint8_t>(max_size, seed);
    std::cout << std::endl;
  }
  if (btree) {

    printf("ordered btree<64>,\t\t");
    test_btree_ordered_insert<uint64_t>(max_size);
    printf("ordered btree<32>,\t\t");
    test_btree_ordered_insert<uint32_t>(max_size);
    // printf("ordered btree<16>,");
    // test_btree_ordered_insert<uint16_t>(max_size);
    // printf("ordered btree<8>");
    // test_btree_ordered_insert<uint8_t>(max_size);
    std::cout << std::endl;

    printf("unordered btree<64>,\t\t");
    test_btree_unordered_insert<uint64_t>(max_size, seed);
    printf("unordered btree<32>,\t\t");
    test_btree_unordered_insert<uint32_t>(max_size, seed);
    // printf("unordered btree<16>,");
    // test_btree_unordered_insert<uint16_t>(max_size, seed);
    // printf("unordered btree<8>");
    // test_btree_unordered_insert<uint8_t>(max_size, seed);
    std::cout << std::endl;
  }

  if (pma) {
    /*
    printf("ordered pma<64>,\t\t");
    test_pma_ordered_insert<uint64_t>(max_size);
    printf("ordered pma<32>,\t\t");
    test_pma_ordered_insert<uint32_t>(max_size);
    printf("ordered pma<i32>, ");
    test_pma_ordered_insert<int32_t>(max_size);
    // printf("ordered pma, b, 16,");
    // test_pma_ordered_insert<uint16_t>(max_size);
    // printf("ordered pma, b, 8,");
    // test_pma_ordered_insert<uint8_t>(max_size);
    std::cout << std::endl;

    printf("unordered pma<64>,\t\t");
    test_pma_unordered_insert<uint64_t>(max_size, seed);
    printf("unordered pma<32>,\t\t");
    test_pma_unordered_insert<uint32_t>(max_size, seed);
    printf("unordered pma<i32>, ");
    test_pma_unordered_insert<int32_t>(max_size, seed);
    // printf("unordered pma, b, 16,");
    // test_pma_unordered_insert<uint16_t>(max_size, seed);
    // printf("unordered pma, b, 8,");
    // test_pma_unordered_insert<uint8_t>(max_size, seed);
    std::cout << std::endl;

    */
  }

  if (cpma) {
    uint64_t start_batch_size = 1;
#if CILK == 1
    start_batch_size = 1000;
#endif
    timing_cpma_helper<uint32_t>(max_size, start_batch_size, 1000000, 1, seed);
    timing_cpma_helper<uint64_t>(max_size, start_batch_size, 1000000, 1, seed);
  }
}

bool real_graph(const std::string &filename, int iters = 20,
                uint32_t start_node = 0, uint32_t max_batch_size = 100000) {
  uint32_t num_nodes = 0;
  uint64_t num_edges = 0;
  auto edges = get_edges_from_file_adj_sym(filename, &num_edges, &num_nodes);

  printf("done reading in the file, n = %u, m = %lu\n", num_nodes, num_edges);
  CPMA<delta_compressed_leaf<uint64_t>, InPlace> g;

  auto start = get_usecs();
  // for (auto edge : edges) {
  //   g.insert(edge);
  // }
  g.insert_batch(edges.data(), edges.size());
  auto end = get_usecs();
  printf("inserting the edges took %lums\n", (end - start) / 1000);
  num_nodes = g.num_nodes();
  int64_t size = g.get_size();
  printf("size = %lu bytes, num_edges = %lu, num_nodes = %u\n", size,
         g.get_element_count(), num_nodes);

  int32_t parallel_bfs_result2_ = 0;
  uint64_t parallel_bfs_time2 = 0;

  for (int i = 0; i < iters; i++) {
    start = get_usecs();
    int32_t *parallel_bfs_result = EdgeMapVertexMap::BFS(g, start_node);
    end = get_usecs();
    parallel_bfs_result2_ += parallel_bfs_result[0];
    if (i == 0 && parallel_bfs_result != nullptr) {
      uint64_t reached = 0;
      for (uint32_t j = 0; j < num_nodes; j++) {
        reached += parallel_bfs_result[j] != -1;
      }
      printf("the bfs from source %u, reached %lu vertices\n", start_node,
             reached);
      std::vector<uint32_t> depths(num_nodes, UINT32_MAX);
      ParallelTools::parallel_for(0, num_nodes, [&](uint32_t j) {
        uint32_t current_depth = 0;
        int32_t current_parent = j;
        if (parallel_bfs_result[j] < 0) {
          return;
        }
        while (current_parent != parallel_bfs_result[current_parent]) {
          current_depth += 1;
          current_parent = parallel_bfs_result[current_parent];
        }
        depths[j] = current_depth;
      });
      std::ofstream myfile;
      myfile.open("bfs.out");
      for (unsigned int i = 0; i < num_nodes; i++) {
        myfile << depths[i] << "\n";
      }
      myfile.close();
    }

    free(parallel_bfs_result);
    parallel_bfs_time2 += (end - start);
  }
  // printf("bfs took %lums, parent of 0 = %d\n", (bfs_time)/(1000*iters),
  // bfs_result_/iters);
  printf("parallel_bfs with edge_map took %lums, parent of 0 = %d\n",
         parallel_bfs_time2 / (1000 * iters), parallel_bfs_result2_ / iters);
  printf("F-Graph, %d, BFS, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)parallel_bfs_time2 / (iters * 1000000));
  double pagerank_value = 0;
  uint64_t pagerank_time = 0;
  double *values3 = nullptr;
  for (int i = 0; i < iters; i++) {
    if (values3 != nullptr) {
      free(values3);
    }
    start = get_usecs();
    values3 = EdgeMapVertexMap::PR_S<double>(g, 10);
    end = get_usecs();
    pagerank_value += values3[0];
    pagerank_time += end - start;
  }
  printf(
      "pagerank with MAPS took %f microsecond, value of 0 = %f, for %d iters\n",
      (double)pagerank_time / iters, values3[0], iters);
  printf("F-Graph, %d, PageRank, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)pagerank_time / (iters * 1000000));
  std::ofstream myfile;
  myfile.open("pr.out");
  for (unsigned int i = 0; i < num_nodes; i++) {
    myfile << values3[i] << "\n";
  }
  myfile.close();
  free(values3);

  double *values4 = nullptr;
  double dep_0 = 0;
  uint64_t bc_time = 0;
  for (int i = 0; i < iters; i++) {
    if (values4 != nullptr) {
      free(values4);
    }
    start = get_usecs();
    values4 = EdgeMapVertexMap::BC(g, start_node);
    end = get_usecs();
    bc_time += end - start;
    dep_0 += values4[0];
  }

  printf("BC took %lums, value of 0 = %f\n", bc_time / (1000 * iters),
         dep_0 / iters);

  printf("F-Graph, %d, BC, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)bc_time / (iters * 1000000));
  if (values4 != nullptr) {
    std::ofstream myfile;
    myfile.open("bc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values4[i] << "\n";
    }
    myfile.close();
    free(values4);
  }

  uint32_t *values5 = nullptr;
  uint32_t id_0 = 0;
  uint64_t cc_time = 0;
  for (int i = 0; i < iters; i++) {
    if (values5) {
      free(values5);
    }
    start = get_usecs();
    values5 = EdgeMapVertexMap::CC(g);
    end = get_usecs();
    cc_time += end - start;
    id_0 += values5[0];
  }

  printf("CC took %lums, value of 0 = %u\n", cc_time / (1000 * iters),
         id_0 / iters);
  printf("F-Graph, %d, Components, %u, %s, ##, %f\n", iters, start_node,
         filename.c_str(), (double)cc_time / (iters * 1000000));
  if (values5 != nullptr) {
    std::unordered_map<uint32_t, uint32_t> components;
    for (uint32_t i = 0; i < num_nodes; i++) {
      components[values5[i]] += 1;
    }
    printf("there are %zu components\n", components.size());
    uint32_t curent_max = 0;
    uint32_t curent_max_key = 0;
    for (auto p : components) {
      if (p.second > curent_max) {
        curent_max = p.second;
        curent_max_key = p.first;
      }
    }
    printf("the element with the biggest component is %u, it has %u members "
           "to its component\n",
           curent_max_key, curent_max);
    std::ofstream myfile;
    myfile.open("cc.out");
    for (uint32_t i = 0; i < num_nodes; i++) {
      myfile << values5[i] << "\n";
    }
    myfile.close();
  }

  free(values5);

  if (true) {
    for (uint32_t b_size = 10; b_size <= max_batch_size; b_size *= 10) {
      auto r = random_aspen(b_size);
      double batch_insert_time = 0;
      double batch_remove_time = 0;
      for (int it = 0; it < iters + 1; it++) {
        // uint64_t size = g.get_memory_size();
        // printf("size start = %lu\n", size);
        double a = 0.5;
        double b = 0.1;
        double c = 0.1;
        size_t nn = 1UL << (log2_up(num_nodes) - 1);
        auto rmat = rMat<uint32_t>(nn, r.ith_rand(it), a, b, c);
        std::vector<uint64_t> es(b_size);
        ParallelTools::parallel_for(0, b_size, [&](uint32_t i) {
          std::pair<uint32_t, uint32_t> edge = rmat(i);
          es[i] = (static_cast<uint64_t>(edge.first) << 32U) | edge.second;
        });

        start = get_usecs();
        g.insert_batch(es.data(), b_size);
        end = get_usecs();
        // printf("%lu\n", end - start);
        if (it > 0) {
          batch_insert_time += end - start;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(es.begin(), es.end(), gen);
        // size = g.get_memory_size();
        // printf("size end = %lu\n", size);
        start = get_usecs();
        g.remove_batch(es.data(), b_size);
        end = get_usecs();
        if (it > 0) {
          batch_remove_time += end - start;
        }
      }
      batch_insert_time /= (1000000 * iters);
      batch_remove_time /= (1000000 * iters);
      // printf("batch_size = %d, time to insert = %f seconds, throughput =
      // %4.2e "
      //        "updates/second\n",
      //        b_size, batch_insert_time, b_size / (batch_insert_time));
      // printf("batch_size = %d, time to remove = %f seconds, throughput =
      // %4.2e "
      //        "updates/second\n",
      //        b_size, batch_remove_time, b_size / (batch_remove_time));
      printf("%u, %f, %f\n", b_size, batch_insert_time, batch_remove_time);
    }
  }
  return true;
}

template <HeadForm head_form, uint64_t B_size = 0>
bool scan_bench(size_t num_elements, size_t num_bits, size_t iters = 5) {
  uint64_t uncompressed_time = 0;
  uint64_t compressed_time = 0;
  for (size_t i = 0; i < iters; i++) {
    std::seed_seq seed;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sum = 0;
    auto data =
        create_random_data_in_parallel<uint64_t>(num_elements, 1UL << num_bits);

    {
      CPMA<uncompressed_leaf<uint64_t>, head_form, B_size> uncompressed_pma;
      uncompressed_pma.insert_batch(data.data(), data.size());
      start = get_usecs();
      sum = uncompressed_pma.sum();
      end = get_usecs();
      printf("uncompressed sum took %lu microseconds was %lu: size per element "
             "was %f: num_unique = %lu\n",
             end - start, sum,
             uncompressed_pma.get_size() /
                 (double)uncompressed_pma.get_element_count(),
             uncompressed_pma.get_element_count());
      uncompressed_time += (end - start);
    }
    {

      CPMA<delta_compressed_leaf<uint64_t>, head_form, B_size> compressed_pma;
      compressed_pma.insert_batch(data.data(), data.size());
      start = get_usecs();
      sum = compressed_pma.sum();
      end = get_usecs();
      printf("compressed sum took %lu microseconds was %lu: size per element "
             "was %f\n",
             end - start, sum,
             compressed_pma.get_size() /
                 (double)compressed_pma.get_element_count());
      compressed_time += (end - start);
    }

    {
      CPMA<delta_compressed_leaf<uint64_t>, head_form, B_size> compressed_pma2;
      compressed_pma2.insert_batch(data.data(), data.size());
      start = get_usecs();
      sum = compressed_pma2.sum();
      end = get_usecs();
      printf("compressed sum took %lu microseconds was %lu: size per element "
             "was %f\n",
             end - start, sum,
             compressed_pma2.get_size() /
                 (double)compressed_pma2.get_element_count());
      compressed_time += (end - start);
    }
    {
      CPMA<uncompressed_leaf<uint64_t>, head_form, B_size> uncompressed_pma2;
      uncompressed_pma2.insert_batch(data.data(), data.size());
      start = get_usecs();
      sum = uncompressed_pma2.sum();
      end = get_usecs();
      printf("uncompressed sum took %lu microseconds was %lu: size per "
             "element was %f\n ",
             end - start, sum,
             uncompressed_pma2.get_size() /
                 (double)uncompressed_pma2.get_element_count());
      uncompressed_time += (end - start);
    }
  }
  printf("average after %zu iters was: uncompressed compressed \n %zu, %zu\n",
         iters * 2, uncompressed_time / (iters * 2),
         compressed_time / (iters * 2));

  return true;
}

bool scan_bench_btree(size_t num_elements, size_t num_bits, size_t iters = 10) {
  uint64_t btree_time = 0;
  for (size_t i = 0; i < iters; i++) {
    std::seed_seq seed;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t sum = 0;
    auto data =
        create_random_data_in_parallel<uint64_t>(num_elements, 1UL << num_bits);

    {

      tlx::btree_set<uint64_t> btree_set(data.begin(), data.end());

      uint64_t num_partitions = ParallelTools::getWorkers() * 10;
      uint64_t per_partition = num_elements / num_partitions;
      std::vector<tlx::btree_set<uint64_t>::const_iterator> iterators(
          num_partitions + 1);
      uint64_t position = 0;
      uint64_t partion_num = 0;
      for (auto it = btree_set.begin(); it != btree_set.end(); it++) {
        if (position % per_partition == 0) {
          iterators[partion_num] = it;
          partion_num += 1;
        }
        position += 1;
      }
      uint64_t correct_sum = 0;
      uint64_t serial_start = get_usecs();
      for (auto it = btree_set.begin(); it != btree_set.end(); it++) {
        correct_sum += *it;
      }
      uint64_t serial_end = get_usecs();
      iterators[num_partitions] = btree_set.end();
      ParallelTools::Reducer_sum<uint64_t> partial_sums;
      start = get_usecs();
      ParallelTools::parallel_for(0, num_partitions, [&](uint64_t i) {
        uint64_t local_sum = 0;
        auto start = iterators[i];
        auto end = iterators[i + 1];
        for (auto it = start; it != end; it++) {
          local_sum += *it;
        }
        partial_sums.add(local_sum);
      });
      end = get_usecs();
      sum = partial_sums.get();
      printf("parallel sum took %lu microseconds was %lu: serial sum took %lu, "
             "got "
             "%lu\n",
             end - start, sum, serial_end - serial_start, correct_sum);
      btree_time += std::min((end - start), serial_end - serial_start);
    }
  }

  printf("average after %zu iters was: \n %zu\n", iters, btree_time / (iters));

  return true;
}

template <HeadForm head_form, uint64_t B_size = 0>
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>
batch_test(size_t num_elements_start, size_t batch_size, size_t num_bits = 40,
           size_t iters = 5, bool verify = false) {
  uint64_t insert_total_uncompressed = 0;
  uint64_t delete_total_uncompressed = 0;
  uint64_t insert_total_compressed = 0;
  uint64_t delete_total_compressed = 0;
  for (size_t i = 0; i < iters + 1; i++) {

    std::seed_seq seed;
    uint64_t start = 0;
    uint64_t end = 0;
    auto data = create_random_data_with_seed<uint64_t>(num_elements_start * 2,
                                                       1UL << num_bits, seed);
    std::set<uint64_t> correct;
    if (verify) {
      correct.insert(data.begin(), data.end());
    }
    {
      CPMA<uncompressed_leaf<uint64_t>, head_form, B_size> pma;
      pma.insert_batch(data.data(), data.size() / 2);
      start = get_usecs();
      for (uint64_t *batch_start = data.data() + data.size() / 2;
           batch_start < data.data() + data.size(); batch_start += batch_size) {
        pma.insert_batch(batch_start, batch_size);
      }
      end = get_usecs();
      if (i > 0) {
        insert_total_uncompressed += end - start;
      }

      if (!verify) {
        printf("batch_size = %lu, total sum = %lu, time = %lu\n", batch_size,
               pma.sum(), end - start);
      }
      if (verify) {
        if (pma_different_from_set(pma, correct)) {
          printf("bad uncompressed pma\n");
          return {-1UL, -1UL, -1UL, -1UL};
        }
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(data.data() + data.size() / 2, data.data() + data.size(), g);

      start = get_usecs();
      for (uint64_t *batch_start = data.data() + data.size() / 2;
           batch_start < data.data() + data.size(); batch_start += batch_size) {
        pma.remove_batch(batch_start, batch_size);
      }
      end = get_usecs();
      if (i > 0) {
        delete_total_uncompressed += end - start;
      }
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.data() + data.size() / 2, data.data() + data.size(), g);
    {
      CPMA<delta_compressed_leaf<uint64_t>, head_form, B_size> pma;
      pma.insert_batch(data.data(), data.size() / 2);
      start = get_usecs();
      for (uint64_t *batch_start = data.data() + data.size() / 2;
           batch_start < data.data() + data.size(); batch_start += batch_size) {
        pma.insert_batch(batch_start, batch_size);
      }
      end = get_usecs();
      if (i > 0) {
        insert_total_compressed += end - start;
      }
      if (!verify) {
        printf("batch_size = %lu, total sum = %lu, time = %lu\n", batch_size,
               pma.sum(), end - start);
      }
      if (verify) {
        if (pma_different_from_set(pma, correct)) {
          printf("bad uncompressed pma\n");
          return {-1UL, -1UL, -1UL, -1UL};
        }
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(data.data() + data.size() / 2, data.data() + data.size(), g);
      start = get_usecs();
      for (uint64_t *batch_start = data.data() + data.size() / 2;
           batch_start < data.data() + data.size(); batch_start += batch_size) {
        pma.remove_batch(batch_start, batch_size);
      }
      end = get_usecs();
      if (i > 0) {
        delete_total_compressed += end - start;
      }
    }
  }
  return {insert_total_uncompressed / iters, delete_total_uncompressed / iters,
          insert_total_compressed / iters, delete_total_compressed / iters};
}

template <HeadForm head_form, uint64_t B_size = 0>
bool batch_bench(size_t num_elements_start, size_t num_bits = 40,
                 size_t iters = 5, bool verify = false) {
  std::map<size_t, uint64_t> insert_total_uncompressed;
  std::map<size_t, uint64_t> delete_total_uncompressed;
  std::map<size_t, uint64_t> insert_total_compressed;
  std::map<size_t, uint64_t> delete_total_compressed;

  for (size_t batch_size = 1; batch_size < num_elements_start;
       batch_size *= 10) {
    auto results = batch_test<head_form, B_size>(num_elements_start, batch_size,
                                                 num_bits, iters, verify);
    insert_total_uncompressed[batch_size] = std::get<0>(results);
    delete_total_uncompressed[batch_size] = std::get<1>(results);
    insert_total_compressed[batch_size] = std::get<2>(results);
    delete_total_compressed[batch_size] = std::get<3>(results);
  }

  for (size_t batch_size = 1; batch_size < num_elements_start;
       batch_size *= 10) {
    printf("%lu, %lu, %lu, %lu, %lu\n", batch_size,
           insert_total_uncompressed[batch_size] / iters,
           delete_total_uncompressed[batch_size] / iters,
           insert_total_compressed[batch_size] / iters,
           delete_total_compressed[batch_size] / iters);
  }
  return false;
}

template <HeadForm head_form, uint64_t B_size = 0>
bool find_bench(size_t num_elements_start, uint64_t num_searches,
                size_t num_bits = 40, size_t iters = 5, bool verify = false) {
  size_t uncompressed_find_time = 0;
  size_t compressed_find_time = 0;
  for (size_t i = 0; i < iters; i++) {

    uint64_t start;
    uint64_t end;
    std::random_device rd1;
    std::seed_seq seed1{rd1()};
    // auto data_to_insert = create_random_data<uint64_t>(num_elements_start,
    //                                                    1UL << num_bits,
    //                                                    seed1);
    auto data_to_insert = create_random_data_in_parallel<uint64_t>(
        num_elements_start, 1UL << num_bits);

    // std::random_device rd2;
    // std::seed_seq seed2{rd2()};
    // auto data_to_search =
    //     create_random_data<uint64_t>(num_searches, 1UL << num_bits, seed2);
    auto data_to_search =
        create_random_data_in_parallel<uint64_t>(num_searches, 1UL << num_bits);
    std::unordered_set<uint64_t> correct;
    uint64_t correct_num_contains = 0;
    if (verify) {
      correct.insert(data_to_insert.begin(), data_to_insert.end());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(correct.contains(data_to_search[i]));
      });
      correct_num_contains = number_contains.get();
    }
    {
      CPMA<uncompressed_leaf<uint64_t>, head_form, B_size> pma;
      pma.insert_batch(data_to_insert.data(), data_to_insert.size());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      start = get_usecs();
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(pma.has(data_to_search[i]));
      });

      end = get_usecs();
      uint64_t num_contains = number_contains.get();
      uncompressed_find_time += end - start;

      if (!verify) {
        printf("found %lu elements, pma had %lu elements, the heads took %lu "
               "bytes, total took %lu bytes\n",
               num_contains, pma.get_element_count(),
               pma.get_head_structure_size(), pma.get_size());
      }
      if (verify) {
        if (correct_num_contains != num_contains) {
          printf("something wrong with the finds, we found %lu, while the "
                 "correct found %lu\n",
                 num_contains, correct_num_contains);
        }
      }
    }
    {
      CPMA<delta_compressed_leaf<uint64_t>, head_form, B_size> pma;
      pma.insert_batch(data_to_insert.data(), data_to_insert.size());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      start = get_usecs();
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(pma.has(data_to_search[i]));
      });

      end = get_usecs();
      uint64_t num_contains = number_contains.get();
      compressed_find_time += end - start;

      if (!verify) {
        printf("found %lu elements, pma had %lu elements, the heads took %lu"
               "bytes\n",
               num_contains, pma.get_element_count(),
               pma.get_head_structure_size());
      }
      if (verify) {
        if (correct_num_contains != num_contains) {
          printf("something wrong with the finds, we found %lu, while the "
                 "correct found %lu\n",
                 num_contains, correct_num_contains);
        }
      }
    }
  }
  printf("uncompressed_find_time = %lu, compressed_find_time = %lu\n",
         uncompressed_find_time / iters, compressed_find_time / iters);
  return false;
}

bool find_bench_tlx_btree(size_t num_elements_start, uint64_t num_searches,
                          size_t num_bits = 40, size_t iters = 5,
                          bool verify = false) {
  size_t uncompressed_find_time = 0;
  size_t compressed_find_time = 0;
  for (size_t i = 0; i < iters; i++) {

    uint64_t start;
    uint64_t end;
    std::random_device rd1;
    std::seed_seq seed1{rd1()};
    // auto data_to_insert = create_random_data<uint64_t>(num_elements_start,
    //                                                    1UL << num_bits,
    //                                                    seed1);
    auto data_to_insert = create_random_data_in_parallel<uint64_t>(
        num_elements_start, 1UL << num_bits);

    // std::random_device rd2;
    // std::seed_seq seed2{rd2()};
    // auto data_to_search =
    //     create_random_data<uint64_t>(num_searches, 1UL << num_bits, seed2);
    auto data_to_search =
        create_random_data_in_parallel<uint64_t>(num_searches, 1UL << num_bits);
    std::unordered_set<uint64_t> correct;
    uint64_t correct_num_contains = 0;
    if (verify) {
      correct.insert(data_to_insert.begin(), data_to_insert.end());
      ParallelTools::Reducer_sum<uint64_t> number_contains;
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(correct.contains(data_to_search[i]));
      });
      correct_num_contains = number_contains.get();
    }
    {
      std::sort(data_to_insert.begin(), data_to_insert.end());

      tlx::btree_set<uint64_t> btree_set;

      btree_set.bulk_load(data_to_insert.begin(), data_to_insert.end());

      ParallelTools::Reducer_sum<uint64_t> number_contains;
      start = get_usecs();
      ParallelTools::parallel_for(0, data_to_search.size(), [&](uint64_t i) {
        number_contains.add(btree_set.exists(data_to_search[i]));
      });

      end = get_usecs();
      uint64_t num_contains = number_contains.get();
      uncompressed_find_time += end - start;

      if (!verify) {
        printf("found %lu elements, memory size was %lu\n", num_contains,
               btree_set.get_stats().memory_size());
      }
      if (verify) {
        if (correct_num_contains != num_contains) {
          printf("something wrong with the finds, we found %lu, while the "
                 "correct found %lu\n",
                 num_contains, correct_num_contains);
        }
      }
    }
  }
  printf("uncompressed_find_time = %lu, compressed_find_time = %lu\n",
         uncompressed_find_time / iters, compressed_find_time / iters);
  return false;
}

int main(int32_t argc, char *argv[]) {
  if (false) {
    uint64_t number_of_elements = std::strtol(argv[1], nullptr, 10);
    auto data = create_random_data<uint32_t>(
        number_of_elements, std::numeric_limits<uint32_t>::max());
    ParallelTools::Reducer_sum<uint64_t> sums_red;
    uint64_t start = get_usecs();
    ParallelTools::parallel_for(0, data.size(),
                                [&](size_t i) { sums_red.add(data[i]); });
    uint64_t sum = sums_red.get();
    uint64_t end = get_usecs();
    printf("the sum was %lu, took %lu usecs to get it\n", sum, end - start);
    return 0;
  }
  if (false) {
    std::ofstream outfile;
    std::seed_seq seed{1};
    uint64_t number_of_elements = std::strtol(argv[1], nullptr, 10);
    uint64_t max_element = std::strtol(argv[2], nullptr, 10);
    std::cout << "max_element = " << max_element
              << " number_of_elements = " << number_of_elements << std::endl;

    double alpha = std::strtod(argv[3], nullptr);
    zipf zip(max_element, alpha, seed);
    std::cout << "done making the generator" << endl;

    outfile.open("zipf_" + std::to_string(max_element) + "_" +
                 std::to_string(number_of_elements) + "_" +
                 std::to_string(alpha));
    std::vector<uint64_t> data = zip.gen_vector(number_of_elements);
    std::cout << "done making the data" << endl;
    for (const auto &d : data) {
      outfile << d << endl;
    }
    outfile.close();
    CPMA<uncompressed_leaf<uint64_t>, Linear> s;
    s.insert_batch(data.data(), data.size());
    std::cout << "unique elements = " << s.get_element_count() << endl;
    return 0;
  }
  if (false) {
    std::ofstream outfile;
    std::seed_seq seed{1};
    uint64_t number_of_elements = std::strtol(argv[1], nullptr, 10);
    uint64_t max_element = std::strtol(argv[2], nullptr, 10);
    auto data = create_random_data<uint64_t>(number_of_elements, max_element);
    std::cout << "max_element = " << max_element
              << " number_of_elements = " << number_of_elements << std::endl;
    auto data2 = data;

    std::cout << "done making the data" << endl;

    outfile.open(argv[3]);
    for (const auto &d : data) {
      outfile << d << endl;
    }
    outfile.close();
    CPMA<uncompressed_leaf<uint64_t>, Linear> s;
    s.insert_batch(data.data(), data.size());
    std::cout << "unique elements = " << s.get_element_count() << endl;
    std::cout << "PMA took " << s.get_size() << " bytes" << std::endl;
    CPMA<delta_compressed_leaf<uint64_t>, Linear> c;
    c.insert_batch(data2.data(), data2.size());
    std::cout << "CPMA took " << c.get_size() << " bytes" << std::endl;
    return 0;
  }

  if (argc == 1) {
    std::cout << "usage options" << std::endl
              << argv[0] << " v" << std::endl
              << argv[0] << " p size # for pgo" << std::endl
              << argv[0] << " m for microbenchmark of pma and btree"
              << std::endl
              << argv[0] << " t {size list} # for perf testing" << std::endl;
    return 1;
  }
  if (std::string("cpma_helper") == argv[1]) {
    std::seed_seq seed{0};
    uint64_t max_size = std::strtol(argv[2], nullptr, 10);
    uint64_t start_batch_size = std::strtol(argv[3], nullptr, 10);
    uint64_t end_batch_size = std::strtol(argv[4], nullptr, 10);
    uint64_t trials = std::strtol(argv[5], nullptr, 10);
    uint64_t size_to_run = std::strtol(argv[6], nullptr, 10);
    if (size_to_run == 32) {
      timing_cpma_helper<uint32_t>(max_size, start_batch_size, end_batch_size,
                                   trials, seed);
    } else if (size_to_run == 64) {
      timing_cpma_helper<uint64_t>(max_size, start_batch_size, end_batch_size,
                                   trials, seed);
    }
  }
  if (std::string("m") == argv[1]) {
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    for (int i = 10; i <= 10000000; i *= 10) {
      // pma_micro_benchmark<uint32_t>(i, seed);
    }
    for (int i = 10; i <= 10000000; i *= 10) {
      btree_micro_benchmark<uint32_t>(i, seed);
    }
  }
  if (std::string("v_leaf") == argv[1]) {
    // bool verify_leaf(uint32_t size, uint32_t num_ops, uint32_t range_start,
    // uint32_t range_end)
    if (verify_leaf<uncompressed_leaf<uint64_t>>(4 * 32, 10, 1, 10)) {
      verify_leaf<uncompressed_leaf<uint64_t>>(4 * 32, 10, 1, 10, 2);
      return 1;
    }
    if (verify_leaf<uncompressed_leaf<uint64_t>>(64 * 32, 200, 1, 100)) {
      return 1;
    }
    if (verify_leaf<delta_compressed_leaf<uint32_t>>(4 * 32, 10, 1, 10)) {
      std::cout << "test failed, running again with more details printed\n";
      verify_leaf<delta_compressed_leaf<uint32_t>>(4 * 32, 10, 1, 10, 2);
      return 1;
    }
    if (verify_leaf<delta_compressed_leaf<uint32_t>>(64 * 32, 10, 1, 10)) {
      return 1;
    }
    if (verify_leaf<delta_compressed_leaf<uint32_t>>(64 * 32, 100, 1, 100)) {
      return 1;
    }
    if (verify_leaf<uncompressed_leaf<uint32_t>>(64 * 32, 10, 1, 10)) {
      return 1;
    }
    if (verify_leaf<uncompressed_leaf<uint32_t>>(64 * 32, 200, 1, 100)) {
      return 1;
    }
    if (verify_leaf<delta_compressed_leaf<uint64_t>>(64 * 32, 10, 1, 10)) {
      return 1;
    }
    if (verify_leaf<delta_compressed_leaf<uint64_t>>(64 * 32, 100, 1, 100)) {
      return 1;
    }
    if (verify_leaf<uncompressed_leaf<uint64_t>>(64 * 32, 10, 1, 10)) {
      return 1;
    }
    if (verify_leaf<uncompressed_leaf<uint64_t>>(64 * 32, 200, 1, 100)) {
      return 1;
    }
  }
  if (std::string("v_uncompressed") == argv[1]) {

    std::cout << "testing cpma<uint32_t> uncompressed_leaf InPlace\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint32_t>, InPlace>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }

    std::cout << "testing cpma<uint32_t> uncompressed_leaf Linear\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint32_t>, Linear>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint32_t> uncompressed_leaf Eytzinger\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint32_t>, Eytzinger>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint32_t> uncompressed_leaf BNary 5\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint32_t>, BNary, 5>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint64_t> uncompressed_leaf InPlace\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint64_t>, InPlace>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint64_t> uncompressed_leaf Linear\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint64_t>, Linear>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint64_t> uncompressed_leaf Eytzinger\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint64_t>, Eytzinger>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }

    std::cout << "testing cpma<uint64_t> uncompressed_leaf BNary 5\n";
    if (verify_cpma_different_sizes<uncompressed_leaf<uint64_t>, BNary, 5>(
            {{100, false}, {1000, false}, {10000, false}, {20000, true}})) {
      return 1;
    }
  }
  if (std::string("v_compressed") == argv[1]) {

    std::cout << "testing cpma<uint32_t> delta_compressed_leaf InPlace\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint32_t>, InPlace>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }

    std::cout << "testing cpma<uint32_t> delta_compressed_leaf Linear\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint32_t>, Linear>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint32_t> delta_compressed_leaf Eytzinger\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint32_t>, Eytzinger>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint32_t> delta_compressed_leaf BNary 5\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint32_t>, BNary, 5>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }

    std::cout << "testing cpma<uint64_t> uncompressed_leaf InPlace\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint64_t>, InPlace>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }

    std::cout << "testing cpma<uint64_t> uncompressed_leaf Linear\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint64_t>, Linear>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }
    std::cout << "testing cpma<uint64_t> delta_compressed_leaf Eytzinger\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint64_t>, Eytzinger>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }

    std::cout << "testing cpma<uint64_t> delta_compressed_leaf BNary 5\n";
    if (verify_cpma_different_sizes<delta_compressed_leaf<uint64_t>, BNary, 5>(
            {{100, false}, {1000, false}, {10000, false}, {100000, true}})) {
      return 1;
    }
  }
  if (std::string("v_batch") == argv[1]) {

    std::cout << "batch bench test\n";
    if (batch_bench<InPlace>(10000, 40, 5, true)) {
      printf("testing: batch bench failed InPlace\n");
      return 1;
    }
    if (batch_bench<Linear>(10000, 40, 5, true)) {
      printf("testing: batch bench failed Linear\n");
      return 1;
    }
    if (batch_bench<Eytzinger>(10000, 40, 5, true)) {
      printf("testing: batch bench failed Eytzinger\n");
      return 1;
    }
    if (batch_bench<BNary, 5>(10000, 40, 5, true)) {
      printf("testing: batch bench failed BNary 5\n");
      return 1;
    }
    if (batch_bench<BNary, 9>(10000, 40, 5, true)) {
      printf("testing: batch bench failed BNary 9\n");
      return 1;
    }
  }
  if (std::string("p") == argv[1]) {
    if (argc != 3) {
      std::cout << "specify 2 arguments for the profile" << std::endl;
    }
    // seed_rand(0);
    // time_leaf<uncompressed_leaf<uint32_t>>(500, 100000, 1, 50000000);
    // seed_rand(0);
    // time_leaf<delta_compressed_leaf<uint32_t>>(500, 100000, 1, 50000000);
    // timing_inserts(std::strtol(argv[2], nullptr, 10), true, true, true,
    // true);

    std::seed_seq s;
    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]), s);

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]));
    test_tlx_btree_ordered_insert(atoll(argv[2]));
  }
  if (std::string("t") == argv[1]) {
    if (argc < 3) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
    }
    for (int i = 2; i < argc; i++) {
      std::cout << "running with size " << argv[i] << std::endl;
      timing_inserts(std::strtol(argv[i], nullptr, 10), true, true, true, true);
    }
  }
  if (std::string("cpma") == argv[1]) {
    if (argc < 3) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
    }
    for (int i = 2; i < argc; i++) {
      std::cout << "running with size " << argv[i] << std::endl;
      timing_inserts(std::strtol(argv[i], nullptr, 10), false, false, false,
                     true);
    }
  }
  if (std::string("s") == argv[1]) {
    if (argc < 3) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
    }
    std::cout << "sizeof(PMA) == " << sizeof(PMA<uint32_t, 0>) << std::endl;
    std::cout << "sizeof(CPMA<uncompressed_leaf<uint32_t>>) == "
              << sizeof(CPMA<uncompressed_leaf<uint32_t>, Linear>) << std::endl;
    std::cout << "sizeof(CPMA<delta_compressed_leaf<uint32_t>>) == "
              << sizeof(CPMA<delta_compressed_leaf<uint32_t>, Linear>)
              << std::endl;
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    std::cout << "############ pma sizes ############" << std::endl;
    test_pma_size<uint32_t>(std::strtol(argv[2], nullptr, 10));
    std::cout << "############ cpma<uncompressed> sizes ############"
              << std::endl;
    test_cpma_size<uncompressed_leaf<uint32_t>, Linear>(
        std::strtol(argv[2], nullptr, 10), seed);
    std::cout << "############ cpma<delta_compressed> sizes ############"
              << std::endl;
    test_cpma_size<delta_compressed_leaf<uint32_t>, Linear>(
        std::strtol(argv[2], nullptr, 10), seed);
  }

  if (std::string("sizes_file") == argv[1]) {
    if (argc < 3) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
    }
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

    test_cpma_size_file_out_simple<uncompressed_leaf<uint64_t>, Linear>(
        std::strtol(argv[2], nullptr, 10), seed, "uncompressed_sizes_1.1.csv");

    test_cpma_size_file_out_simple<delta_compressed_leaf<uint64_t>, Linear>(
        std::strtol(argv[2], nullptr, 10), seed, "compressed_sizes_1.1.csv");

    /*
    test_cpma_size_file_out<uncompressed_leaf<uint32_t>>(
        std::strtol(argv[2], nullptr, 10), seed, "uncompressed_sizes.csv");
    test_cpma_size_file_out<delta_compressed_leaf<uint32_t>>(
        std::strtol(argv[2], nullptr, 10), seed, "compressed_sizes.csv");
        */
    // test_cpma_size_file_out<uncompressed_leaf<uint32_t>>(
    //     std::strtol(argv[2], nullptr, 10), seed, "uncompressed_sizes.csv");
    // test_cpma_size_file_out<delta_compressed_leaf<uint32_t>>(
    //     std::strtol(argv[2], nullptr, 10), seed,
    //     "compressed_sizes_uint32_t.csv");
    // test_cpma_size_file_out<delta_compressed_leaf<uint64_t>>(
    //     std::strtol(argv[2], nullptr, 10), seed,
    //     "compressed_sizes_uint64_t.csv");
  }
  if (std::string("b") == argv[1]) {
    if (argc < 3) {
      std::cout << "specify at least 2 arguments for the test";
    }
    for (int i = 2; i < argc; i++) {
      std::cout << "running with size " << argv[i] << std::endl;
      timing_inserts(std::strtol(argv[i], nullptr, 10), false, true, false,
                     false);
    }
  }
  if (std::string("d") == argv[1]) {
    if (argc < 4) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
      return 1;
    }
    uint64_t batch_size = atoi(argv[2]);
    char *filename = argv[3];
    char *outfilename = argv[4];
    int num_trials = atoi(argv[5]);
    printf("start timing pma from data with filename %s\n", filename);

    timing_pma_from_data(batch_size, filename, outfilename, num_trials);
  }
  if (std::string("d_scalability") == argv[1]) {
    if (argc < 4) {
      std::cout << "specify at least 2 arguments for the test" << std::endl;
      return 1;
    }
    char *filename = argv[2];
    char *outfilename = argv[3];
    int num_trials = atoi(argv[4]);
    int serial = atoi(argv[5]); // 0 for parallel, 1 for serial
    printf("start timing pma from data with filename %s\n", filename);

    pma_scalability_from_data(filename, outfilename, num_trials, serial);
  }
  if (std::string("d_sizes_file") == argv[1]) {
    std::vector<std::string> filenames;

    char *outfilename = argv[2]; // output file suffix
    // list input files as the remaining args
    for (int i = 3; i < argc; i++) {
      filenames.emplace_back(argv[i]);
    }

    // will output as csv in
    // pma_outfilename, cpma_outfilename
    cout << "output " << outfilename << endl;
    test_cpma_size_file_out_from_data(filenames, outfilename);
  }

  if (std::string("g") == argv[1]) {
    real_graph(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
  }
  if (std::string("scan_inplace") == argv[1]) {
    scan_bench<InPlace>(atoll(argv[2]), atoll(argv[3]));
  }
  if (std::string("scan_lin") == argv[1]) {
    scan_bench<Linear>(atoll(argv[2]), atoll(argv[3]));
  }
  if (std::string("scan_eyt") == argv[1]) {
    scan_bench<Eytzinger>(atoll(argv[2]), atoll(argv[3]));
  }
  if (std::string("scan_B5") == argv[1]) {
    scan_bench<BNary, 5>(atoll(argv[2]), atoll(argv[3]));
  }
  if (std::string("scan_B9") == argv[1]) {
    scan_bench<BNary, 9>(atoll(argv[2]), atoll(argv[3]));
  }
  if (std::string("scan_B17") == argv[1]) {
    scan_bench<BNary, 17>(atoll(argv[2]), atoll(argv[3]));
  }
  if (std::string("scan_btree") == argv[1]) {
    scan_bench_btree(atoll(argv[2]), atoll(argv[3]));
  }

  if (std::string("batch_bench") == argv[1]) {
    batch_bench<InPlace>(atoll(argv[2]), 40, 10);
    batch_bench<Linear>(atoll(argv[2]), 40, 10);
    batch_bench<Eytzinger>(atoll(argv[2]), 40, 10);
    batch_bench<BNary, 5>(atoll(argv[2]), 40, 10);
    batch_bench<BNary, 9>(atoll(argv[2]), 40, 10);
  }
  if (std::string("batch") == argv[1]) {
    {
      auto results =
          batch_test<InPlace>(atoll(argv[2]), atoll(argv[3]), 40, 10);
      std::cout << std::get<0>(results) << ", " << std::get<1>(results) << ","
                << std::get<2>(results) << ", " << std::get<3>(results) << "\n";
    }
#if 0
    {
      auto results = batch_test<Linear>(atoll(argv[2]), atoll(argv[3]));
      std::cout << std::get<0>(results) << ", " << std::get<1>(results) << ","
                << std::get<2>(results) << ", " << std::get<3>(results) << "\n";
    }
    {
      auto results = batch_test<Eytzinger>(atoll(argv[2]), atoll(argv[3]));
      std::cout << std::get<0>(results) << ", " << std::get<1>(results) << ", "
                << std::get<2>(results) << ", " << std::get<3>(results) << "\n";
    }
    {
      auto results = batch_test<BNary, 5>(atoll(argv[2]), atoll(argv[3]));
      std::cout << std::get<0>(results) << ", " << std::get<1>(results) << ", "
                << std::get<2>(results) << ", " << std::get<3>(results) << "\n";
    }
    {
      auto results = batch_test<BNary, 9>(atoll(argv[2]), atoll(argv[3]));
      std::cout << std::get<0>(results) << ", " << std::get<1>(results) << ", "
                << std::get<2>(results) << ", " << std::get<3>(results) << "\n";
    }
#endif
  }
  if (std::string("single") == argv[1]) {
    std::seed_seq s;

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, InPlace>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, InPlace>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]), s);

    test_cpma_unordered_insert<uncompressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]), s);
    test_cpma_unordered_insert<delta_compressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]), s);
  }
  if (std::string("single_btree") == argv[1]) {
    std::seed_seq s;
    test_tlx_btree_unordered_insert(atoll(argv[2]), s);
  }

  if (std::string("single_seq") == argv[1]) {
    std::seed_seq s;
    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, InPlace>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, InPlace>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, Linear>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, Eytzinger>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, BNary, 9>(
        atoll(argv[2]));

    test_cpma_ordered_insert<uncompressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]));
    test_cpma_ordered_insert<delta_compressed_leaf<uint64_t>, BNary, 17>(
        atoll(argv[2]));
    test_tlx_btree_ordered_insert(atoll(argv[2]));
  }
  if (std::string("single_alt") == argv[1]) {
    std::seed_seq s;
    uint64_t num_items = atoll(argv[2]);
    uint64_t proportion_ordered = atoll(argv[3]);
    uint64_t proportion_random = atoll(argv[4]);
    // test_cpma_ordered_and_unordered_insert<uncompressed_leaf<uint64_t>,
    // Linear>(
    //     num_items, proportion_ordered, proportion_random, s);

    test_cpma_ordered_and_unordered_insert<uncompressed_leaf<uint64_t>,
                                           Eytzinger>(
        num_items, proportion_ordered, proportion_random, s);

    // test_cpma_ordered_and_unordered_insert<uncompressed_leaf<uint64_t>,
    // BNary,
    //                                        9>(num_items, proportion_ordered,
    //                                           proportion_random, s);

    // test_cpma_ordered_and_unordered_insert<uncompressed_leaf<uint64_t>,
    // BNary,
    //                                        17>(num_items, proportion_ordered,
    //                                            proportion_random, s);

    test_btree_ordered_and_unordered_insert(num_items, proportion_ordered,
                                            proportion_random, s);
  }

  if (std::string("multi_seq") == argv[1]) {
    uint64_t num_items = atoll(argv[2]);
    uint64_t groups = atoll(argv[3]);

    test_cpma_multi_seq_insert<uncompressed_leaf<uint64_t>, Eytzinger>(
        num_items, groups);

    test_btree_multi_seq_insert(num_items, groups);
  }

  if (std::string("bulk") == argv[1]) {
    uint64_t num_items = atoll(argv[2]);
    uint64_t num_per = atoll(argv[3]);

    test_cpma_bulk_insert<uncompressed_leaf<uint64_t>, Eytzinger>(num_items,
                                                                  num_per);

    test_btree_bulk_insert(num_items, num_per);
  }
  if (std::string("find") == argv[1]) {
    find_bench<InPlace>(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
    find_bench<Linear>(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
    find_bench<Eytzinger>(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
    find_bench<BNary, 5>(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
    find_bench<BNary, 9>(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
    find_bench<BNary, 17>(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
    find_bench_tlx_btree(atoll(argv[2]), atoll(argv[3]), 40, 1, false);
  }
  return 0;
}
