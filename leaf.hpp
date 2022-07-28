#pragma once
#include "ParallelTools/parallel.h"
#include "helpers.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <malloc.h>

#include "parlaylib/include/parlay/sequence.h"

/*
  Only works to store unique elements
  top bit is 1 for continue, or 0 for the end of an element
*/

template <class T> class delta_compressed_leaf {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "T can only be 4 or 8 bytes");

public:
  static constexpr size_t max_element_size = sizeof(T) + (sizeof(T) / 4);
  static constexpr bool compressed = true;
  using value_type = T;
  T &head;
  uint8_t *array;
  const int64_t length_in_bytes;

  T *T_array() const { return (T *)array; }

private:
  class FindResult {
  public:
    T difference;
    int64_t loc;
    int64_t size;
    void print() {
      std::cout << "FindResult { difference=" << difference << ", loc=" << loc
                << ", size=" << size << " }" << std::endl;
    }
  };
  class EncodeResult {
  public:
    static constexpr int storage_size = std::max(max_element_size + 1, 8UL);
    static constexpr std::array<uint64_t, 5> dep_masks = {
        0, 0x0000000000000080UL, 0x0000000000008080UL, 0x0000000000808080UL,
        0x0000000080808080UL};
    uint8_t data[storage_size] = {0};
    int64_t size;
    void print() {
      std::cout << "EncodeResult { data={";
      for (int64_t i = 0; i < size; i++) {
        std::cout << static_cast<uint32_t>(data[i]) << ", ";
      }
      std::cout << "} , size=" << size << " }" << std::endl;
    }
    static int64_t write_encoded(T difference, uint8_t *loc) {
      // if constexpr (sizeof(T) == 4) {
      //   uint64_t out = _pdep_u64(difference, 0x7F7F7F7F7FUL);
      //   int32_t size = bsr_long(out);
      //   size /= 8;
      //   out |= dep_masks[size];
      //   memcpy(loc, &out, 8);
      //   return size + 1;
      // }
      loc[0] = difference & 0x7FU;
      int64_t num_bytes = difference > 0;
      difference >>= 7;
      while (difference) {
        loc[num_bytes - 1] |= 0x80U;
        loc[num_bytes] = difference & 0x7FU;
        num_bytes += 1;
        difference >>= 7;
      }
      return num_bytes;
    }
    EncodeResult(T difference) {
      assert(difference != 0);
      // not always true, but a sanity check to help me debug
      // assert(difference < 1000000000);

      size = write_encoded(difference, data);
      assert((size_t)size <= max_element_size);
    }
  };
  class DecodeResult {
  public:
    T difference = 0;
    int64_t old_size = 0;
    void print() {
      std::cout << "DecodeResult { difference=" << difference
                << ", old_size=" << old_size << " }" << std::endl;
    }
    static constexpr std::array<uint64_t, 8> extract_masks = {
        0x000000000000007FUL, 0x0000000000007F7FUL, 0x00000000007F7F7FUL,
        0x000000007F7F7F7FUL, 0x0000007F7F7F7F7FUL, 0x00007F7F7F7F7F7FUL,
        0x007F7F7F7F7F7F7FUL, 0x7F7F7F7F7F7F7F7FUL};

    static constexpr std::array<uint64_t, 16> masks_for_4 = {
        0x7FUL, 0x7F7FUL,     0x7FUL, 0x7F7F7FUL,    0x7FUL, 0x7F7FUL,
        0x7FUL, 0x7F7F7F7FUL, 0x7FUL, 0x7F7FUL,      0x7FUL, 0x7F7F7FUL,
        0x7FUL, 0x7F7FUL,     0x7FUL, 0x7F7F7F7F7FUL};
    static constexpr std::array<int8_t, 16> index_for_4 = {
        1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5};

    DecodeResult() = default;

    DecodeResult(T d, int64_t s) : difference(d), old_size(s) {}

    DecodeResult(const uint8_t *loc) {

      if (*loc == 0) {
        difference = 0;
        old_size = 0;
        return;
      }
#if __BMI2__ == 1
      uint64_t chunks = unaligned_load<uint64_t>(loc);

      // static_assert(sizeof(T) == 4, "fix this to work for 8 byte items");

      // these come in and out depending on the relative cost of
      // parallel extract and count_trailing_zeroes (tzcount)
      // and branches on your specific hardware
      if ((chunks & 0x80UL) == 0) {
        difference = *loc;
        old_size = 1;
        return;
      }
      // if ((chunks & 0x8000UL) == 0) {
      //   // T difference_guess = _pext_u64(chunks, 0x0000000000007F7FUL);
      //   difference = (*loc & 0x7F) | (*(loc + 1) << 7);
      //   // difference = difference_guess;
      //   old_size = 2;
      //   return;
      // }
      // if ((chunks & 0x800000UL) == 0) {
      //   // T difference_guess = _pext_u64(chunks, 0x00000000007F7F7FUL);
      //   difference =
      //       (*loc & 0x7F) | ((*(loc + 1) & 0x7F) << 7) | (*(loc + 2) << 14);

      //   // difference = difference_guess;
      //   old_size = 3;
      //   return;
      // }
      // if ((chunks & 0x80000000UL) == 0) {
      //   T difference_guess = _pext_u64(chunks, 0x000000007F7F7F7FUL);
      //   difference = difference_guess;
      //   old_size = 4;
      //   return;
      // }
      // T difference_guess = _pext_u64(chunks, 0x0000007F7F7F7F7FUL);
      // difference = difference_guess;
      // old_size = 5;
      // return;
      uint64_t mask = _pext_u64(chunks, 0x8080808080808080UL);
      if (sizeof(T) == 4 ||
          (chunks & 0x8080808080808080UL) != 0x8080808080808080UL) {
        int32_t index = _mm_tzcnt_64(~mask);
        difference = _pext_u64(chunks, extract_masks[index]);
        old_size = index + 1;
        // printf("chunks = %lx, mask = %lx, index = %u, difference =%lu\n",
        //        chunks, mask, old_size, difference);
        return;
      }

#endif
      difference = *loc & 0x7FU;
      old_size = 1;
      uint32_t shift_amount = 7;
      if (*loc & 0x80U) {
        do {
          ASSERT(shift_amount < 8 * sizeof(T), "shift_amount = %u\n",
                 shift_amount);
          loc += 1;
          difference = difference | ((*loc & 0x7FUL) << shift_amount);
          old_size += 1;
          shift_amount += 7;
        } while (*loc & 0x80U);
      }
    }
    DecodeResult(const uint8_t *loc, int64_t max_size) {
      if (*loc == 0) {
        difference = 0;
        old_size = 0;
        return;
      }
      difference = *loc & 0x7FU;
      old_size = 1;
      uint32_t shift_amount = 7;
      if (*loc & 0x80U) {
        do {
          if (old_size >= max_size) {
            break;
          }
          ASSERT(shift_amount < 8 * sizeof(T), "shift_amount = %u\n",
                 shift_amount);
          loc += 1;
          difference = difference | ((*loc & 0x7FUL) << shift_amount);
          old_size += 1;
          shift_amount += 7;
        } while (*loc & 0x80U);
      }
    }
  };

  // returns the starting byte location, the length of the specified element,
  // and the difference from the previous element length is 0 if the element
  // is not found starting byte location of 0 means this is the head
  // if we have just changed the head we pass in the old head val so we can
  // correctly interpret the bytes after that
  FindResult find(T x) const {
    T curr_elem = head;
    T prev_elem = 0;
    // heads are dealt with seprately
    assert(x > head);
    int64_t curr_loc = 0;
    DecodeResult dr;
    while (curr_elem < x) {
      dr = DecodeResult(array + curr_loc);
      prev_elem = curr_elem;
      if (dr.old_size == 0) {
        break;
      }
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
    }
    assert(curr_loc < length_in_bytes);
    if (x == curr_elem) {
      return {0, curr_loc - dr.old_size, dr.old_size};
    }
    // std::cout << "x = " << x << ", curr_elem = " << curr_elem << std::endl;
    return {x - prev_elem, curr_loc - dr.old_size, 0};
  }
  static T get_out_of_place_used_bytes(T *data) { return data[0]; }
  static void set_out_of_place_used_bytes(T *data, T bytes) { data[0] = bytes; }

  static T *get_out_of_place_pointer(T *data) {
    T *pointer = nullptr;
    memcpy(&pointer, data + 1, sizeof(T *));
    return pointer;
  }
  static void set_out_of_place_pointer(T *data, T *pointer) {
    memcpy(data + 1, &pointer, sizeof(T *));
  }
  static T get_out_of_place_last_written(T *data) { return data[3]; }
  static void set_out_of_place_last_written(T *data, T last) { data[3] = last; }
  static T get_out_of_place_temp_size(T *data) { return data[4]; }
  static void set_out_of_place_temp_size(T *data, T size) { data[4] = size; }

  T get_out_of_place_used_bytes() const {
    return get_out_of_place_used_bytes(T_array());
  }
  void set_out_of_place_used_bytes(T bytes) const {
    set_out_of_place_used_bytes(T_array(), bytes);
  }

  T *get_out_of_place_pointer() const {
    return get_out_of_place_pointer(T_array());
  }
  void set_out_of_place_pointer(T *pointer) const {
    set_out_of_place_pointer(T_array(), pointer);
  }
  T get_out_of_place_last_written() const {
    return get_out_of_place_last_written(T_array());
  }
  void set_out_of_place_last_written(T last) const {
    set_out_of_place_last_written(T_array(), last);
  }
  T get_out_of_place_temp_size() const {
    return get_out_of_place_temp_size(T_array());
  }
  void set_out_of_place_temp_size(T size) const {
    set_out_of_place_temp_size(T_array(), size);
  }

  void slide_right(const uint8_t *loc, uint64_t amount) {
    assert(amount <= max_element_size);
    std::memmove((void *)(loc + amount), (void *)loc,
                 length_in_bytes - (loc + amount - array));
    // std::memset((void *)loc, 0, amount);
  }

  void slide_left(const uint8_t *loc, uint64_t amount) {
    assert(amount <= max_element_size);
    std::memmove((void *)(loc - amount), (void *)loc,
                 length_in_bytes - (loc + amount - array));
    std::memset((void *)(array + (length_in_bytes - amount)), 0, amount);
    // TODO(AUTHOR) get early exit working
    // uint16_t *long_data_pointer = (uint16_t *)loc;
    // uint16_t *long_data_pointer_offset = (uint16_t *)(loc - amount);
    // while (*long_data_pointer) {
    //   *long_data_pointer_offset = *long_data_pointer;
    //   long_data_pointer += 1;
    //   long_data_pointer_offset += 1;
    // }
    // *long_data_pointer_offset = 0;
    // assert(((uint8_t *)long_data_pointer) - array < length_in_bytes);
  }

  static void small_memcpy(uint8_t *dest, uint8_t *source, size_t n) {
    assert(n < 16);
    if (n >= 8) {
      unaligned_store(dest, unaligned_load<uint64_t>(source));
      source += 8;
      dest += 8;
      n -= 8;
    }
    assert(n < 8);
    if (n >= 4) {
      unaligned_store(dest, unaligned_load<uint32_t>(source));
      source += 4;
      dest += 4;
      n -= 4;
    }
    assert(n < 4);
    if (n >= 2) {
      unaligned_store(dest, unaligned_load<uint16_t>(source));
      source += 2;
      dest += 2;
      n -= 2;
    }
    assert(n < 2);
    if (n >= 1) {
      *dest = *source;
    }
  }

public:
  delta_compressed_leaf(T &head, void *data_, int64_t length)
      : head(head), array(static_cast<uint8_t *>(data_)),
        length_in_bytes(length) {}

  // Input: pointer to the start of this merge in the batch, end of batch,
  // value in the PMA at the next head (so we know when to stop merging)
  // Output: returns a tuple (ptr to where this merge stopped in the batch,
  // number of distinct elements merged in, and number of bytes used in this
  // leaf)
  template <bool head_in_place>
  std::tuple<T *, uint64_t, uint64_t> merge_into_leaf(T *batch_start,
                                                      T *batch_end, T end_val) {
    // std::cout << "end_val = " << end_val << std::endl;
    // std::cout << "batch to add\n";
    // for (T *e = batch_start; e < batch_end && *e < end_val; e++) {
    //   std::cout << *e << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "starting leaf: \n";
    // print();

    // case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      // printf("case 1, batch start %p, end %p, end val %u\n", batch_start,
      //        batch_end, end_val);
      // if (batch_start + 1 < batch_end) {
      //   printf("next val %u\n", batch_start[1]);
      // }
      auto [inserted, byte_count] = insert<head_in_place>(*batch_start);
      return {batch_start + 1, inserted, byte_count};
    }

    // case 2: more than 1 elt from batch
    // two-finger merge into extra space, might overflow leaf
    // printf("case 2\n");
    uint64_t temp_size = length_in_bytes / sizeof(T);

    // get enough size for the batch
    while (batch_start + temp_size < batch_end &&
           batch_start[temp_size] < end_val) {
      temp_size *= 2;
    }
    // get enough size for the leaf
    temp_size = (temp_size * sizeof(T)) + length_in_bytes;

    // add sizeof(T) to store the head since it is out of place in the pma, but
    // with the data in the temp storage
    temp_size += sizeof(T);
    if (temp_size % 32 != 0) {
      temp_size = (temp_size / 32) * 32 + 32;
    }
    uint8_t *temp_arr = (uint8_t *)aligned_alloc(32, temp_size);

#ifndef NDEBUG
    std::memset(temp_arr, 0, temp_size);
#endif

    T *batch_ptr = batch_start;
    uint8_t *leaf_ptr = array;
    uint8_t *temp_ptr = temp_arr;

    uint64_t distinct_batch_elts = 0;
    const uint8_t *leaf_end = array + length_in_bytes;
    T last_written = 0;
    // deal with the head
    const T old_head = head;
    T last_in_leaf = old_head;
    // new head from batch
    if (batch_ptr[0] < old_head) {
      *((T *)temp_ptr) = *batch_ptr;
      last_written = *batch_ptr;
      distinct_batch_elts++;
      temp_ptr += sizeof(T);
      batch_ptr++;
    } else { // copy over the old head
      *((T *)temp_ptr) = old_head;
      last_written = old_head;
      temp_ptr += sizeof(T);
    }
    // anything that needs go from the batch before the old head
    while (batch_ptr < batch_end && batch_ptr[0] < old_head) {
      T new_difference = *batch_ptr - last_written;
      if (new_difference > 0) {
        int64_t er_size = EncodeResult::write_encoded(new_difference, temp_ptr);
        last_written = *batch_ptr;
        distinct_batch_elts++;
        temp_ptr += er_size;
      }
      batch_ptr++;
    }
    // if we still need to copy the old leaf head
    if (leaf_ptr == array) {
      T new_difference = old_head - last_written;
      if (new_difference > 0) {
        int64_t er_size = EncodeResult::write_encoded(new_difference, temp_ptr);
        last_written = old_head;
        temp_ptr += er_size;
      }
    }

    // deal with the rest of the elements
    while (batch_ptr < batch_end && batch_ptr[0] < end_val &&
           leaf_ptr < leaf_end && *leaf_ptr != 0) {
      DecodeResult dr(leaf_ptr);
      // if duplicates in batch, skip
      if (*batch_ptr == last_written) {
        batch_ptr++;
        // std::cout << "skipping duplicate\n";
        continue;
      }
      T leaf_element = last_in_leaf + dr.difference;
      // std::cout << last_written << ", " << leaf_element << ", " << *batch_ptr
      //           << ", " << temp_ptr - temp_arr << std::endl;
      assert(leaf_element > last_written);
      // otherwise, do a step of the merge
      if (leaf_element == *batch_ptr) {
        // std::cout << leaf_element << " from both\n";
        int64_t er_size =
            EncodeResult::write_encoded(*batch_ptr - last_written, temp_ptr);
        leaf_ptr += dr.old_size;
        temp_ptr += er_size;
        last_written = *batch_ptr;
        assert(last_written < end_val);
        last_in_leaf = *batch_ptr;
        batch_ptr++;
      } else if (leaf_element > *batch_ptr) {
        // std::cout << *batch_ptr << " from batch\n";
        int64_t er_size =
            EncodeResult::write_encoded(*batch_ptr - last_written, temp_ptr);
        last_written = *batch_ptr;
        assert(last_written < end_val);
        batch_ptr++;
        distinct_batch_elts++;
        temp_ptr += er_size;
      } else {
        // std::cout << leaf_element << " from leaf\n";
        int64_t er_size =
            EncodeResult::write_encoded(leaf_element - last_written, temp_ptr);
        last_written = leaf_element;
        assert(last_written < end_val);
        last_in_leaf = leaf_element;
        temp_ptr += er_size;
        leaf_ptr += dr.old_size;
        assert(temp_ptr < temp_arr + temp_size);
      }
    }
    // std::cout << batch_ptr << ", " << batch_end << ", " << *batch_ptr << ", "
    //           << last_written << std::endl;
    // std::cout << temp_ptr - temp_arr << std::endl;

    // write rest of the batch if it exists
    while (batch_ptr < batch_end && batch_ptr[0] < end_val) {
      if (*batch_ptr == last_written) {
        batch_ptr++;
        continue;
      }
      T new_difference = *batch_ptr - last_written;
      // std::cout << "writting " << *batch_ptr << std::endl;
      int64_t er_size = EncodeResult::write_encoded(new_difference, temp_ptr);
      last_written = *batch_ptr;
      assert(last_written < end_val);
      batch_ptr++;
      distinct_batch_elts++;
      temp_ptr += er_size;
      assert(temp_ptr < temp_arr + temp_size);
    }

    // write the rest of the original leaf if it exist
    // first write the next element which needs to calculate a new difference
    if (leaf_ptr < leaf_end) {
      DecodeResult dr(leaf_ptr);
      if (dr.old_size != 0) {
        T leaf_element = last_in_leaf + dr.difference;
        // std::cout << "writing leftover leaf " << leaf_element << std::endl;
        int64_t er_size =
            EncodeResult::write_encoded(leaf_element - last_written, temp_ptr);
        last_written = leaf_element;
        assert(last_written < end_val);
        temp_ptr += er_size;
        leaf_ptr += dr.old_size;
        // then just copy over the rest
        while (leaf_ptr < leaf_end) {
          DecodeResult dr2(leaf_ptr);
          if (dr2.old_size == 0) {
            break;
          }
          small_memcpy(temp_ptr, leaf_ptr, dr2.old_size);
          last_written += dr2.difference;
          assert(last_written < end_val);
          temp_ptr += dr2.old_size;
          leaf_ptr += dr2.old_size;
          assert(temp_ptr < temp_arr + temp_size);
        }
      }
    }
    assert(temp_ptr < temp_arr + temp_size);
    // std::cout << temp_ptr - temp_arr << std::endl;
    int64_t used_bytes = temp_ptr - temp_arr;
    // write the byte after to zero so nothing else tries to read off the end
    if ((uint64_t)used_bytes < temp_size - 1) {
      temp_ptr[0] = 0;
      temp_ptr[1] = 0;
    }
    assert((uint64_t)used_bytes < temp_size);
    // check if you can fit in the leaf with some extra space at the end for
    // safety
    if (used_bytes <= length_in_bytes - 5) {
      // printf("\tNO OVERFLOW\n");
      // dest, src, size
      head = *((T *)temp_arr);
      memcpy(array, temp_arr + sizeof(T), used_bytes - sizeof(T));
      free(temp_arr);
    } else { // special write for when you don't fit
      // printf("\tYES OVERFLOW\n");
      head = 0;
      set_out_of_place_used_bytes(used_bytes);
      set_out_of_place_pointer((T *)temp_arr);
      set_out_of_place_last_written(last_written);
      set_out_of_place_temp_size(temp_size);
      assert(last_written < end_val);
    }
    // std::cout << "ending leaf: \n";
    // print();
    return {batch_ptr, distinct_batch_elts,
            used_bytes - ((head_in_place) ? 0 : sizeof(T)) /* for the head*/};
  }
  template <bool head_in_place>
  std::tuple<T *, uint64_t, uint64_t> strip_from_leaf(T *batch_start,
                                                      T *batch_end, T end_val) {
    // std::cout << "end_val = " << end_val << std::endl;
    // std::cout << "batch to add\n";
    // for (T *e = batch_start; e < batch_end && *e < end_val; e++) {
    //   std::cout << *e << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "starting leaf: \n";
    // print();

    // case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      // printf("case 1, batch start %p, end %p\n", batch_start, batch_end);
      // if (batch_start + 1 < batch_end) {
      //   printf("next val %u\n", batch_start[1]);
      // }
      auto [removed, byte_count] = remove<head_in_place>(*batch_start);
      return {batch_start + 1, removed, byte_count};
    }

    // case 2: more than 1 elt from batch
    // two-finger merge into extra space, might overflow leaf
    // printf("case 2\n");

    T *batch_ptr = batch_start;
    uint8_t *front_pointer = array;
    uint8_t *back_pointer = array;

    uint64_t distinct_batch_elts = 0;
    const uint8_t *leaf_end = array + length_in_bytes;
    T last_written = 0;
    // deal with the head
    const T old_head = head;
    T last_in_leaf = old_head;
    // anything from the batch before the old head is skipped
    while (batch_ptr < batch_end && batch_ptr[0] < old_head) {
      batch_ptr++;
    }
    bool head_written = false;

    // if we are not removing the old leaf head
    if ((batch_ptr < batch_end && *batch_ptr != old_head) ||
        batch_ptr == batch_end) {
      last_written = old_head;
      assert(last_written < end_val);
      head_written = true;
    } else {
      distinct_batch_elts++;
    }

    // deal with the rest of the elemnts
    while (batch_ptr < batch_end && batch_ptr[0] < end_val &&
           front_pointer < leaf_end) {
      DecodeResult dr(front_pointer);
      if (dr.old_size == 0) {
        // std::cout << "break\n";
        break;
      }
      T leaf_element = last_in_leaf + dr.difference;
      // std::cout << last_written << ", " << leaf_element << ", " << *batch_ptr
      //           << ", " << temp_ptr - temp_arr << std::endl;
      assert(leaf_element > last_written);
      // otherwise, do a step of the merge
      if (leaf_element == *batch_ptr) {
        // std::cout << leaf_element << " from both\n";
        front_pointer += dr.old_size;
        last_in_leaf = leaf_element;
        batch_ptr++;
        distinct_batch_elts++;
      } else if (leaf_element > *batch_ptr) {
        // std::cout << *batch_ptr << " from batch\n";
        batch_ptr++;
      } else {
        if (head_written) {
          // std::cout << leaf_element << " from leaf\n";
          EncodeResult er(leaf_element - last_written);
          front_pointer += dr.old_size;
          small_memcpy(back_pointer, er.data, er.size);
          last_written = leaf_element;
          assert(last_written < end_val);
          last_in_leaf = leaf_element;
          back_pointer += er.size;
        } else {
          assert(back_pointer == array);
          head = leaf_element;
          front_pointer += dr.old_size;
          last_written = leaf_element;
          assert(last_written < end_val);
          head_written = true;
          last_in_leaf = leaf_element;
        }
        assert(back_pointer <= front_pointer);
      }
    }
    // std::cout << batch_ptr << ", " << batch_end << ", " << *batch_ptr << ", "
    //           << last_written << std::endl;
    // std::cout << temp_ptr - temp_arr << std::endl;

    // write the rest of the original leaf if it exist
    // first write the next element which needs to calculate a new difference
    if (front_pointer < leaf_end) {
      DecodeResult dr(front_pointer);
      if (dr.old_size != 0) {
        T leaf_element = last_in_leaf + dr.difference;
        // std::cout << "writing leftover leaf " << leaf_element << std::endl;
        if (head_written) {
          EncodeResult er(leaf_element - last_written);
          small_memcpy(back_pointer, er.data, er.size);
          back_pointer += er.size;
        } else {
          head = leaf_element;
          head_written = true;
        }
        last_written = leaf_element;
        assert(last_written < end_val);

        front_pointer += dr.old_size;
        assert(back_pointer <= front_pointer);
        // then just copy over the rest
        while (front_pointer < leaf_end) {
          DecodeResult dr2(front_pointer);
          if (dr2.old_size == 0) {
            break;
          }
          small_memcpy(back_pointer, front_pointer, dr2.old_size);
          last_written += dr2.difference;
          assert(last_written < end_val);
          back_pointer += dr2.old_size;
          front_pointer += dr2.old_size;
          assert(back_pointer <= front_pointer);
        }
      }
    }
    // std::cout << temp_ptr - temp_arr << std::endl;
    int64_t used_bytes = back_pointer - array;
    if (!head_written) {
      head = 0;
    }
    memset(back_pointer, 0, front_pointer - back_pointer);
    if (head != 0) {
      if constexpr (head_in_place) {
        used_bytes += sizeof(T);
      }
    }

    // std::cout << "ending leaf: \n";
    // print();
    return {batch_ptr, distinct_batch_elts, used_bytes};
  }

  // Inputs: start of PMA node , number of leaves we want to merge, size of
  // leaf in bytes, number of nonempty bytes in range

  // returns: merged leaf, number of full bytes in leaf
  template <bool head_in_place, bool have_densities, typename F,
            typename density_array_type>
  static std::pair<delta_compressed_leaf<T>, uint64_t>
  parallel_merge(T *start, uint64_t num_leaves, uint64_t leaf_size,
                 uint64_t leaf_start_index, F index_to_head,
                 [[maybe_unused]] density_array_type density_array) {
    std::vector<uint64_t> bytes_per_leaf(num_leaves);
    std::vector<T> last_per_leaf(num_leaves);
    std::vector<T *> leaf_start(num_leaves);

    ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
      delta_compressed_leaf l(index_to_head(leaf_start_index + i),
                              start + i * leaf_size / sizeof(T) +
                                  ((head_in_place) ? 1 : 0),
                              leaf_size - ((head_in_place) ? sizeof(T) : 0));

      auto last_size_start = l.last_and_size_in_bytes();
      last_per_leaf[i] = std::get<0>(last_size_start);
      bytes_per_leaf[i] = std::get<1>(last_size_start);
      if (l.head == 0) {
        if (l.get_out_of_place_used_bytes() == 0) {
          // its empty, but we may as well put a valid pointer here
          leaf_start[i] = nullptr;
        } else {
          index_to_head(leaf_start_index + i) = *l.get_out_of_place_pointer();
          leaf_start[i] = l.get_out_of_place_pointer() + 1;
        }
      } else {
        leaf_start[i] = l.T_array();
      }
    });

    // check to make sure the leaf we bring the head from has data
    uint64_t start_leaf = std::numeric_limits<uint64_t>::max();
    for (uint64_t i = 0; i < num_leaves; i++) {
      if (leaf_start[i] != nullptr) {
        start_leaf = i;
        break;
      }
    }

    ParallelTools::parallel_for(start_leaf + 1, num_leaves, [&](uint64_t i) {
      T head = index_to_head(leaf_start_index + i);
      if (head != 0) {
        T last = last_per_leaf[i - 1];
        if (last == 0) {
          if (i >= 2) {
            uint64_t j = i - 2;
            while (j < i) {
              last = last_per_leaf[j];
              if (last) {
                break;
              }
              j -= 1;
            }
          }
        }
        T difference = head - last;
        assert(difference < head);
        index_to_head(leaf_start_index + i) = difference;
        bytes_per_leaf[i] += EncodeResult(difference).size;
      }
      // printf("i = %lu, bytes_per_leaf = %lu\n", i, bytes_per_leaf[i]);
    });

    uint64_t total_size;
#if CILK == 0
    total_size = prefix_sum_inclusive(bytes_per_leaf);
#else
    if (num_leaves > 1 << 15) {
      total_size = parlay::scan_inclusive_inplace(bytes_per_leaf);
    } else {
      total_size = prefix_sum_inclusive(bytes_per_leaf);
    }
#endif
    uint64_t memory_size =
        ((total_size + (num_leaves * sizeof(T)) + 31) / 32) * 32 + 32;
    // printf("memory_size = %lu\n", memory_size);
    uint8_t *merged_arr = (uint8_t *)(aligned_alloc(32, memory_size));
    // going to place the head 1 before the data
    merged_arr += sizeof(T);

    // first loop not in parallel due to weird compiler behavior
    if (start_leaf < std::numeric_limits<uint64_t>::max()) {
      uint64_t i = start_leaf;
      *(reinterpret_cast<T *>(merged_arr) - 1) =
          index_to_head(leaf_start_index);
      memcpy(merged_arr, leaf_start[i], bytes_per_leaf[start_leaf]);
      if (leaf_start[i] !=
          start + i * leaf_size / sizeof(T) + ((head_in_place) ? 1 : 0)) {
        // -1 since in the external leaves we store the head one before the data
        free(leaf_start[i] - 1);
      }
    }

    ParallelTools::parallel_for(start_leaf + 1, num_leaves, [&](uint64_t i) {
      if (index_to_head(leaf_start_index + i) != 0) {
        EncodeResult head(index_to_head(leaf_start_index + i));
        uint8_t *dest = merged_arr + bytes_per_leaf[i - 1];
        small_memcpy(dest, head.data, head.size);
        dest += head.size;
        uint8_t *source = (uint8_t *)(leaf_start[i]);
        memcpy(dest, source,
               bytes_per_leaf[i] - bytes_per_leaf[i - 1] - head.size);
        // printf("i = %lu, %u, %u, %u\n", i, bytes_per_leaf[i],
        //        bytes_per_leaf[i - 1], head.size);
        if (leaf_start[i] !=
            start + i * leaf_size / sizeof(T) + ((head_in_place) ? 1 : 0)) {
          // -1 since in the external leaves we store the head one before the
          // data
          free(leaf_start[i] - 1);
        }
      }
    });
    for (uint64_t i = total_size; i < memory_size - sizeof(T); i++) {
      merged_arr[i] = 0;
    }

    delta_compressed_leaf result(*(reinterpret_cast<T *>(merged_arr) - 1),
                                 (void *)merged_arr, memory_size);
    // result.print();
    //+sizeof(T) to include the head
    return {result, total_size + sizeof(T)};
  }

  template <bool head_in_place, bool have_densities, typename F,
            typename density_array_type>
  static std::pair<delta_compressed_leaf<T>, uint64_t>
  merge(T *start, uint64_t num_leaves, uint64_t leaf_size,
        uint64_t leaf_start_index, F index_to_head,
        [[maybe_unused]] density_array_type density_array) {

#if CILK == 1
    if (num_leaves > ParallelTools::getWorkers() * 100U) {
      return parallel_merge<head_in_place, have_densities>(
          start, num_leaves, leaf_size, leaf_start_index, index_to_head,
          density_array);
    }
#endif
    uint64_t dest_size = (max_element_size - sizeof(T)) * num_leaves;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * (leaf_size / sizeof(T));
      if (index_to_head(leaf_start_index + i) == 0 &&
          get_out_of_place_used_bytes(start + src_idx +
                                      ((head_in_place) ? 1 : 0)) != 0) {
        // +(max_element_size - sizeof(T)) to account for any extra space the
        // head might need
        dest_size += get_out_of_place_used_bytes(start + src_idx +
                                                 ((head_in_place) ? 1 : 0));
      } else {
        dest_size += leaf_size;
      }
    }

    if constexpr (head_in_place) {
      start += 1;
    }

    // printf("mallocing size %u\n", dest_size);
    uint64_t memory_size = ((dest_size + 31) / 32) * 32;
    T *merged_arr = (T *)(aligned_alloc(32, memory_size));

    uint8_t *dest_byte_position = (uint8_t *)merged_arr;

    uint8_t *src = (uint8_t *)start;
    T prev_elt = 0;
    uint8_t *leaf_start = (uint8_t *)start;

    // deal with first leaf separately
    // copy head uncompressed
    bool done_head = false;
    if (index_to_head(leaf_start_index) != 0) {
      done_head = true;
      merged_arr[0] = index_to_head(leaf_start_index);
      prev_elt = index_to_head(leaf_start_index);
      dest_byte_position += sizeof(T);

      // copy rest of leaf
      // T current_elem = start[0];
      DecodeResult dr(src);
      while (dr.old_size != 0 && (uint64_t)(src - leaf_start) < leaf_size) {
        small_memcpy(dest_byte_position, src, dr.old_size);
        dest_byte_position += dr.old_size;
        src += dr.old_size;
        prev_elt += dr.difference;
        if ((uint64_t)(src - leaf_start) >= leaf_size) {
          break;
        }
        dr = DecodeResult(src);
      }
    } else if (get_out_of_place_used_bytes(start) != 0) {
      // leaf is in extra storage
      done_head = true;
      memcpy(merged_arr, get_out_of_place_pointer(start),
             get_out_of_place_used_bytes(start));
      dest_byte_position += get_out_of_place_used_bytes(start);
      prev_elt = get_out_of_place_last_written(start);
      // std::cout << prev_elt << std::endl;
      free(get_out_of_place_pointer(start)); // release temp storage
    }

    // prev_elt should be the end of this node at this point
    uint64_t leaves_so_far = 1;
    while (leaves_so_far < num_leaves) {
      // deal with head
      leaf_start = (uint8_t *)start + leaves_so_far * leaf_size;
      T *leaf_start_array_pointer = (T *)leaf_start;
      T head = index_to_head(leaf_start_index + leaves_so_far);
      if (head != 0) {
        ASSERT(head > prev_elt, "head = %lu, prev_elt = %lu\n", (uint64_t)head,
               (uint64_t)prev_elt);
        if (done_head) {
          T diff = head - prev_elt;
          // copy in encoded head with diff from end of previous block
          EncodeResult er(diff);
          small_memcpy(dest_byte_position, er.data, er.size);
          dest_byte_position += er.size;
        } else {
          done_head = true;
          dest_byte_position += sizeof(T);
          // we know since we haven't written the head yet that we are at the
          // start
          *merged_arr = head;
        }

        src = leaf_start;
        prev_elt = head;

        // copy body
        DecodeResult dr(src);
        while (dr.old_size != 0 && (uint64_t)(src - leaf_start) < leaf_size) {
          small_memcpy(dest_byte_position, src, dr.old_size);
          dest_byte_position += dr.old_size;
          src += dr.old_size;
          prev_elt += dr.difference;
          if ((uint64_t)(src - leaf_start) + 1 >= leaf_size) {
            break;
          }
          dr = DecodeResult(src);
        }
      } else if (get_out_of_place_used_bytes(leaf_start_array_pointer) != 0) {
        T *extrenal_leaf = get_out_of_place_pointer(leaf_start_array_pointer);
        T external_leaf_head = *extrenal_leaf;
        EncodeResult er(external_leaf_head - prev_elt);
        small_memcpy(dest_byte_position, er.data, er.size);
        dest_byte_position += er.size;
        memcpy(dest_byte_position, extrenal_leaf + 1,
               get_out_of_place_used_bytes(leaf_start_array_pointer) -
                   sizeof(T));
        dest_byte_position +=
            get_out_of_place_used_bytes(leaf_start_array_pointer) - sizeof(T);
        prev_elt = get_out_of_place_last_written(leaf_start_array_pointer);
        // std::cout << prev_elt << std::endl;
        free(extrenal_leaf); // release temp storage
      }
      leaves_so_far++;
    }

    // how many bytes were filled in the dest?
    uint64_t num_full_bytes = dest_byte_position - (uint8_t *)merged_arr;
    assert(num_full_bytes < dest_size);
    std::memset(dest_byte_position, 0, memory_size - num_full_bytes);

    delta_compressed_leaf result(merged_arr[0], (void *)(merged_arr + 1),
                                 memory_size - sizeof(T));
    // result.print();

    // comment back in to debug, also need to comment out the freeing of extra
    // space
    // if (num_leaves >= 2) {
    //   auto parallel_check = parallel_merge<head_in_place, have_densities>(
    //       start - head_in_place, num_leaves, leaf_size, leaf_start_index,
    //       index_to_head, density_array);
    //   if (num_full_bytes != parallel_check.second) {
    //     printf("%lu != %lu\n", num_full_bytes, parallel_check.second);
    //   }
    //   ASSERT(num_full_bytes == parallel_check.second, "%lu != %lu\n",
    //          num_full_bytes, parallel_check.second);
    //   for (uint64_t i = 0; i < num_full_bytes; i++) {
    //     uint8_t *arr = result.array;
    //     if (arr[i] != parallel_check.first.array[i]) {
    //       printf("%u != %u for i = %lu, len was %lu\n", arr[i],
    //              parallel_check.first.array[i], i, num_full_bytes);
    //       result.print();
    //       parallel_check.first.print();
    //       abort();
    //     }

    //     ASSERT(arr[i] == parallel_check.first.array[i],
    //            "%u != %u for i = %lu, len was %lu\n", arr[i],
    //            parallel_check.first.array[i], i, num_full_bytes);
    //   }
    // }

    return {result, num_full_bytes};
  }

  // input: a merged leaf in delta-compressed format
  // input: number of leaves to split into
  // input: number of occupied bytes in the input leaf
  // input: number of bytes per output leaf
  // input: pointer to the start of the output area to write to (requires that
  // you have num_leaves * num_bytes bytes available here to write to)
  // output: split input leaf into num_leaves leaves, each with
  // num_output_bytes bytes
  template <bool head_in_place, bool store_densities, typename F,
            typename density_array_type>
  void parallel_split(const uint64_t num_leaves,
                      const uint64_t num_occupied_bytes,
                      const uint64_t bytes_per_leaf, T *dest_region,
                      uint64_t leaf_start_index, F index_to_head,
                      density_array_type density_array) {
    // std::cout << num_leaves << ", " << num_occupied_bytes << ", "
    //           << bytes_per_leaf << ", " << dest_region << std::endl;
    std::vector<uint8_t *> start_points(num_leaves + 1);
    start_points[0] = array;
    // - sizeof(T) is becuase num_occupied_bytes counts the head, but its not in
    // the array
    start_points[num_leaves] = array + num_occupied_bytes - sizeof(T);
    uint64_t count_per_leaf = num_occupied_bytes / num_leaves;
    uint64_t extra = num_occupied_bytes % num_leaves;

    ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
      uint64_t start_guess_index =
          count_per_leaf * i + std::min(i, extra) - max_element_size / 2;
      // we are looking for the end of the previous element
      uint8_t *start_guess = array + start_guess_index - 1;
      if ((*start_guess & 0x80U) == 0) {
        start_points[i] = start_guess + 1;
      }
      for (uint8_t j = 1; j < max_element_size; j++) {
        if ((start_guess[j] & 0x80U) == 0) {
          start_points[i] = start_guess + j + 1;
          break;
        }
        // if ((start_guess[-j] & 0x80U) == 0) {
        //   start_points[i] = start_guess - j + 1;
        //   break;
        // }
      }
      // printf("leaf = %lu, index = %lu\n", i, start_points[i] - array);
      assert(start_points[i] > array);
      assert(start_points[i] <= array + (bytes_per_leaf * num_leaves));
    });
    std::vector<T> difference_accross_leaf(num_leaves);
    // first loop not in parallel due to weird compiler behavior
    {
      uint64_t i = 0;
      uint8_t *start = start_points[i];
      uint8_t *end = start_points[i + 1];
      assert(end > start);
      assert(start != end);
      T *dest = dest_region + i * bytes_per_leaf / sizeof(T);
      if constexpr (head_in_place) {
        dest += 1;
      }
      index_to_head(leaf_start_index) = head;
      memcpy(dest, start, end - start);
      if constexpr (store_densities) {
        density_array[leaf_start_index] = end - start;
        if constexpr (head_in_place) {
          if (index_to_head(leaf_start_index) != 0) {
            density_array[leaf_start_index] += sizeof(T);
          }
        }
      }
      ASSERT(uint64_t(end - start) + max_element_size < bytes_per_leaf,
             "end - start = %lu, bytes_per_leaf = %lu\n", uint64_t(end - start),
             bytes_per_leaf);
      memset(((uint8_t *)dest) + (end - start), 0,
             bytes_per_leaf - (end - start) -
                 ((head_in_place) ? sizeof(T) : 0));
      delta_compressed_leaf<T> l(index_to_head(leaf_start_index + i), dest,
                                 bytes_per_leaf -
                                     ((head_in_place) ? sizeof(T) : 0));
      // l.print();
      difference_accross_leaf[i] = l.last();
      assert(difference_accross_leaf[i] != 0);
    }

    ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
      uint8_t *start = start_points[i];
      uint8_t *end = start_points[i + 1];
      assert(end > start);
      assert(start != end);
      T *dest = dest_region + i * bytes_per_leaf / sizeof(T);
      if constexpr (head_in_place) {
        dest += 1;
      }
      DecodeResult head(start);
      assert(head.difference != 0);
      index_to_head(leaf_start_index + i) = head.difference;
      memcpy(dest, start + head.old_size, (end - start) - head.old_size);
      if constexpr (store_densities) {
        density_array[leaf_start_index + i] = (end - start) - head.old_size;
        if constexpr (head_in_place) {
          // for head
          if (index_to_head(leaf_start_index + i) != 0) {
            density_array[leaf_start_index + i] += sizeof(T);
          }
        }
      }
      assert((end - start) - head.old_size + max_element_size <
             bytes_per_leaf - ((head_in_place) ? sizeof(T) : 0));
      for (uint8_t *it = ((uint8_t *)dest) + (end - start) - head.old_size;
           it < ((uint8_t *)dest_region) + (i + 1) * bytes_per_leaf; ++it) {
        *it = 0;
      }

      delta_compressed_leaf<T> l(index_to_head(leaf_start_index + i), dest,
                                 bytes_per_leaf -
                                     ((head_in_place) ? sizeof(T) : 0));
      // l.print();
      difference_accross_leaf[i] = l.last();
      assert(difference_accross_leaf[i] != 0);
    });
    // for (auto e : difference_accross_leaf) {
    //   std::cout << e << ",";
    // }
    // std::cout << std::endl;

#if CILK == 0
    prefix_sum_inclusive(difference_accross_leaf);
#else
    if (num_leaves > 1 << 15) {
      parlay::scan_inclusive_inplace(difference_accross_leaf);
    } else {
      prefix_sum_inclusive(difference_accross_leaf);
    }
#endif
    // for (auto e : difference_accross_leaf) {
    //   std::cout << e << ",";
    // }
    // std::cout << std::endl;
    ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
      index_to_head(leaf_start_index + i) += difference_accross_leaf[i - 1];
      assert(index_to_head(leaf_start_index + i) != 0);
    });
  }

  template <bool head_in_place, bool store_densities, typename F,
            typename density_array_type>
  void split(const uint64_t num_leaves, const uint64_t num_occupied_bytes,
             const uint64_t bytes_per_leaf, T *dest_region,
             uint64_t leaf_start_index, F index_to_head,
             density_array_type density_array) {
    ASSERT(used_size_simple<head_in_place>() ==
               num_occupied_bytes - ((head_in_place) ? 0 : sizeof(T)),
           "used_size_simple() == %lu, num_occupied_bytes - ((head_in_place) ? "
           "0 : sizeof(T)) = %lu\n",
           used_size_simple<head_in_place>(),
           num_occupied_bytes - ((head_in_place) ? 0 : sizeof(T)));
    // std::cout << num_leaves << ", " << num_occupied_bytes << ", "
    //           << bytes_per_leaf << ", " << dest_region << std::endl;

    if (num_leaves == 1) {
      if constexpr (head_in_place) {
        dest_region += 1;
      }
      index_to_head(leaf_start_index) = head;
      memcpy((void *)dest_region, (void *)array, num_occupied_bytes);
      if constexpr (store_densities) {
        density_array[leaf_start_index] = num_occupied_bytes;
        if constexpr (head_in_place) {
          // for head
          if (index_to_head(leaf_start_index) != 0) {
            density_array[leaf_start_index] += sizeof(T);
          }
        }
      }
      memset(((uint8_t *)dest_region) + num_occupied_bytes, 0,
             bytes_per_leaf - num_occupied_bytes -
                 ((head_in_place) ? sizeof(T) : 0));
      return;
    }
#if CILK == 1
    if (num_leaves > ParallelTools::getWorkers() * 100U) {
      return parallel_split<head_in_place, store_densities>(
          num_leaves, num_occupied_bytes, bytes_per_leaf, dest_region,
          leaf_start_index, index_to_head, density_array);
    }
#endif
    if constexpr (head_in_place) {
      dest_region += 1;
    }
    // std::cout << "start: target = " << (num_occupied_bytes) / (num_leaves)
    //           << " remaining bytes = " << num_occupied_bytes
    //           << " leaves left = " << num_leaves << std::endl;

    // print();
    // approx occupied bytes per leaf
    // uint32_t count_per_leaf = num_occupied_bytes / num_leaves;
    // uint32_t extra = num_occupied_bytes % num_leaves;
    assert(num_occupied_bytes / num_leaves <= bytes_per_leaf);

    uint8_t *dest = (uint8_t *)dest_region;
    // get first elt
    uint8_t *src = array;
    T cur_elt = head;
    index_to_head(leaf_start_index) = head;
    uint64_t bytes_read = sizeof(T);
    // do intermediate leaves with heads
    for (uint64_t leaf_idx = 0; leaf_idx < num_leaves - 1; leaf_idx++) {
      uint64_t bytes_for_leaf =
          (num_occupied_bytes - bytes_read) / (num_leaves - leaf_idx);
      // std::cout << "trying to put about " << bytes_for_leaf << " in leaf"
      //           << std::endl;
      // copy leaf head
      if (leaf_idx > 0) {
        DecodeResult dr(src);
        cur_elt += dr.difference;
        src += dr.old_size;
        bytes_for_leaf -= dr.old_size;
        index_to_head(leaf_start_index + leaf_idx) = cur_elt;
        assert(cur_elt > 0);
        bytes_read += dr.old_size;
      }
      uint64_t bytes_so_far = 0;
      uint8_t *leaf_start = src; // start of differences in this leaf from src
      while (bytes_so_far < bytes_for_leaf) {
        DecodeResult dr(src);
        // printf("\tbytes read so far: %u\n", bytes_read);
        // dr.print();
        assert(dr.old_size > 0);
        // early exit if we would end up with too much data somewhere
        if (bytes_so_far + dr.old_size >= bytes_per_leaf - max_element_size) {
          break;
        }
        src += dr.old_size;
        bytes_so_far += dr.old_size;
        cur_elt += dr.difference;
        bytes_read += dr.old_size;
      }

      uint64_t num_bytes_filled = src - leaf_start;
      // std::cout << "ended up putting " << num_bytes_filled << " in leaf"
      //           << std::endl;

      ASSERT(num_bytes_filled < bytes_per_leaf - max_element_size,
             "num_bytes_filled = %lu, bytes_per_leaf = %lu, max_element_size = "
             "%lu\n",
             num_bytes_filled, bytes_per_leaf, max_element_size);
      memcpy(dest, leaf_start, num_bytes_filled);
      if constexpr (store_densities) {
        density_array[leaf_start_index + leaf_idx] = num_bytes_filled;
        if constexpr (head_in_place) {
          // for head
          if (index_to_head(leaf_start_index + leaf_idx) != 0) {
            density_array[leaf_start_index + leaf_idx] += sizeof(T);
          }
        }
      }
      memset(dest + num_bytes_filled, 0,
             bytes_per_leaf - num_bytes_filled -
                 ((head_in_place) ? sizeof(T) : 0));

      // jump to start of next leaf
      dest += bytes_per_leaf;
    }
    // std::cout << (void *)dest << std::endl;

    // handle last leaf
    // do the head
    DecodeResult dr(src);
    assert(dr.difference > 0);
    cur_elt += dr.difference;
    src += dr.old_size;
    bytes_read += dr.old_size;
    index_to_head(leaf_start_index + num_leaves - 1) = cur_elt;
    assert(cur_elt > 0);

    // copy the rest
    uint64_t leftover_bytes = num_occupied_bytes - bytes_read;
    // std::cout << "the last leaf gets " << leftover_bytes << std::endl;

    ASSERT(leftover_bytes <= bytes_per_leaf,
           "leftover_bytes = %lu, bytes_per_leaf = %lu\n", leftover_bytes,
           bytes_per_leaf);
    memcpy(dest, src, leftover_bytes);
    if constexpr (store_densities) {
      density_array[leaf_start_index + num_leaves - 1] = leftover_bytes;
      if constexpr (head_in_place) {
        // for head
        if (index_to_head(leaf_start_index + num_leaves - 1) != 0) {
          density_array[leaf_start_index + num_leaves - 1] += sizeof(T);
        }
      }
    }
    memset(dest + leftover_bytes, 0,
           bytes_per_leaf - leftover_bytes - ((head_in_place) ? sizeof(T) : 0));
  }

  // inserts an element
  // first return value indicates if something was inserted
  // if something was inserted the second value tells you the current size
  template <bool head_in_place> std::pair<bool, size_t> insert(T x) {
    if constexpr (head_in_place) {
      // used_size counts the head, length in bytes does not
      assert(used_size<head_in_place>() <
             length_in_bytes + sizeof(T) - max_element_size);
    } else {
      assert(used_size<head_in_place>() < length_in_bytes - max_element_size);
    }
    // std::cout << "leaf insert " << x << std::endl;
    if (x == head) {
      return {false, 0};
    }
    if (head == 0) {
      head = x;
      return {true, (head_in_place) ? sizeof(T) : 0};
    }
    if (x < head) {
      T temp = head;
      head = x;
      // since we just swapped the head we need to tell find to use the old head
      // when interpreting the bytes
      EncodeResult er(temp - head);
      slide_right(array, er.size);
      small_memcpy(array, er.data, er.size);
      return {true, used_size_with_start<head_in_place>(0)};
    }
    FindResult fr = find(x);

    if (fr.size != 0) {
      return {false, 0};
    }
    EncodeResult er(fr.difference);
    // fr.print();
    // er.print();
    DecodeResult next_difference(array + fr.loc);
    // next_difference.print();

    // we are inserting a new last element and don't need to slide
    if (next_difference.old_size == 0) {
      small_memcpy(array + fr.loc, er.data, er.size);
      return {true, fr.loc + er.size + ((head_in_place) ? sizeof(T) : 0)};
    }

    T old_difference = next_difference.difference;
    T new_difference = old_difference - fr.difference;
    // std::cout << new_difference << std::endl;
    EncodeResult new_er(new_difference);
    // new_er.print();

    size_t slide_size = er.size - (next_difference.old_size - new_er.size);

    // its possible that after adding the new element we don'y need to shift
    // anything over since the new and old difference together have the same
    // size as just the old difference
    if (slide_size > 0) {
      slide_right(array + fr.loc + next_difference.old_size, slide_size);
      // std::memmove(data + fr.loc + next_difference.old_size + slide_size,
      //              data + fr.loc + next_difference.old_size,
      //              length_in_bytes -
      //                  (fr.loc + next_difference.old_size + slide_size));
    }
    small_memcpy(array + fr.loc, er.data, er.size);
    small_memcpy(array + fr.loc + er.size, new_er.data, new_er.size);
    // print();
    return {true, used_size_with_start<head_in_place>(fr.loc + er.size +
                                                      new_er.size)};
  }

  // removes an element
  // first return value indicates if something was removed
  // if something was removed the second value tells you the current size
  template <bool head_in_place> std::pair<bool, size_t> remove(T x) {
    if (head == 0 || x < head) {
      return {false, 0};
    }
    if (x == head) {
      DecodeResult dr(array);
      T old_head = head;
      // before there was only a head
      if (dr.old_size == 0) {
        head = 0;
        set_out_of_place_used_bytes(0);
        return {true, 0};
      }
      head = old_head + dr.difference;
      slide_left(array + dr.old_size, dr.old_size);
      // std::memmove(data + sizeof(T), data + sizeof(T) + dr.old_size,
      //              length_in_bytes - (sizeof(T) + dr.old_size));
      // std::memset(data + length_in_bytes - dr.old_size, 0, dr.old_size);
      return {true, used_size<head_in_place>()};
    }
    FindResult fr = find(x);
    // fr.print();

    if (fr.size == 0) {
      return {false, 0};
    }

    DecodeResult dr(array + fr.loc);

    DecodeResult next_difference(array + fr.loc + dr.old_size);
    // we removed the last element
    if (next_difference.old_size == 0) {
      for (int64_t i = 0; i < dr.old_size; i++) {
        array[fr.loc + i] = 0;
      }
      size_t ret = fr.loc;
      if constexpr (head_in_place) {
        ret += sizeof(T);
      }
      return {true, ret};
    }

    T old_difference = next_difference.difference;
    T new_difference = old_difference + dr.difference;
    EncodeResult new_er(new_difference);

    size_t slide_size = dr.old_size - (new_er.size - next_difference.old_size);

    if (slide_size > 0) {
      slide_left(array + fr.loc + new_er.size + slide_size, slide_size);
      // std::memmove(data + fr.loc + new_er.size,
      //              data + fr.loc + new_er.size + slide_size,
      //              length_in_bytes - (fr.loc + new_er.size + slide_size));
    }
    small_memcpy(array + fr.loc, new_er.data, new_er.size);

    // std::memset(data + length_in_bytes - slide_size, 0, slide_size);
    return {true, used_size_with_start<head_in_place>(fr.loc)};
  }
  bool contains(T x) const {
    if (x < head) {
      return false;
    }
    if (x == head) {
      return true;
    }
    FindResult fr = find(x);
    return fr.size != 0;
  }

  bool debug_contains(T x) {
    if (head == 0) {
      T size_in_bytes = get_out_of_place_temp_size();
      T *ptr = get_out_of_place_pointer();
      auto leaf = delta_compressed_leaf(*ptr, ptr + 1, size_in_bytes);
      return leaf.contains(x);
    }
    return contains(x);
  }

  template <bool head_in_place = false> uint64_t sum() {
    T curr_elem = head;
    uint64_t curr_sum = head;
    int64_t curr_loc = 0;
    DecodeResult dr(array);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      curr_sum += curr_elem;
      while ((*(array + curr_loc) & 0x80U) == 0) {
        uint8_t a = *(array + curr_loc);
        if (!a) {
          return curr_sum;
        }
        curr_elem += a;
        curr_sum += curr_elem;
        curr_loc++;
      }
      dr = DecodeResult(array + curr_loc);
    }
    return curr_sum;
  }

  template <bool no_early_exit, class F> bool map(F f, T start, T end) {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      // printf("curr_elem = %lu, start = %lu, end = %lu\n", curr_elem, start,
      //        end);
      if (curr_elem >= end) {
        return false;
      }
      curr_loc += dr.old_size;
      if (curr_elem >= start) {
        if (f(curr_elem)) {
          if constexpr (!no_early_exit) {
            return true;
          }
        }
      }
      dr = DecodeResult(array + curr_loc);
    }
    return false;
  }

  template <bool no_early_exit, class F> bool map(F f) {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      // printf("curr_elem = %lu, start = %lu, end = %lu\n", curr_elem, start,
      //        end);
      curr_loc += dr.old_size;
      if constexpr (!no_early_exit) {
        if (f(curr_elem)) {
          return true;
        }
      } else {
        f(curr_elem);
      }
      dr = DecodeResult(array + curr_loc);
    }
    return false;
  }
  template <bool no_early_exit, class F> bool map_no_start(F f, T end) {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      // printf("curr_elem = %lu, start = %lu, end = %lu\n", curr_elem, start,
      //        end);
      if (curr_elem >= end) {
        return false;
      }
      curr_loc += dr.old_size;
      if (f(curr_elem)) {
        if constexpr (!no_early_exit) {
          return true;
        }
      }

      dr = DecodeResult(array + curr_loc);
    }
    return false;
  }
  template <bool no_early_exit, class F>
  static bool partial_map(F f, T end, const uint8_t *position, T curr_elem) {
    DecodeResult dr(position);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      if (curr_elem >= end) {
        return false;
      }

      if (f(curr_elem)) {
        if constexpr (!no_early_exit) {
          return true;
        }
      }

      position += dr.old_size;
      dr = DecodeResult(position);
    }
    return false;
  }
  static std::pair<uint8_t *, T>
  find_loc_and_difference_with_hint(T element, uint8_t *position, T curr_elem) {
    DecodeResult dr(position);
    while (dr.difference != 0) {
      T new_el = curr_elem + dr.difference;
      if (new_el >= element) {
        return {position, curr_elem};
      }
      position += dr.old_size;
      dr = DecodeResult(position);
      curr_elem = new_el;
    }
    return {position, curr_elem};
  }

  std::pair<uint8_t *, T> find_loc_and_difference(T element) {
    T curr_elem = head;
    int64_t curr_loc = 0;
    DecodeResult dr(array);
    while (dr.difference != 0) {
      T new_el = curr_elem + dr.difference;
      if (new_el >= element) {
        // this is wrong if its the head
        return {array + curr_loc, curr_elem};
      }
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
      curr_elem = new_el;
    }
    return {array + curr_loc, curr_elem};
  }

  T last() {
    T curr_elem = 0;
    int64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
    }
    return curr_elem;
  }

  std::tuple<T, uint64_t> last_and_size_in_bytes() {
    if (head == 0) {
      if (get_out_of_place_used_bytes() == 0) {
        return {0, 0};
      }
      return {get_out_of_place_last_written(),
              get_out_of_place_used_bytes() - sizeof(T)};
    }
    T curr_elem = 0;
    uint64_t curr_loc = 0;
    DecodeResult dr(head, 0);
    // printf("curr_loc = %lu\n", curr_loc);
    // dr.print();
    while (dr.difference != 0) {
      curr_elem += dr.difference;
      curr_loc += dr.old_size;
      dr = DecodeResult(array + curr_loc);
      // printf("curr_loc = %lu\n", curr_loc);
      // dr.print();
    }
    return {curr_elem, curr_loc};
  }
  template <bool head_in_place> size_t used_size_simple() {
    T curr_elem = head;
    if (curr_elem == 0) {
      if (get_out_of_place_used_bytes() > 0) {
        if constexpr (head_in_place) {
          return get_out_of_place_used_bytes();
        } else {
          return get_out_of_place_used_bytes() - sizeof(T);
        }
      }
      return 0;
    }
    int64_t curr_loc = 0;
    while (unaligned_load<uint16_t>(&array[curr_loc]) != 0) {
      curr_loc += 1;
    }
    if (curr_loc == length_in_bytes) {
      // only possibility is that just the very last byte is empty
      assert(array[curr_loc - 1] == 0);
      if constexpr (head_in_place) {
        curr_loc += sizeof(T);
      }
      return curr_loc - 1;
    }
    if constexpr (head_in_place) {
      curr_loc += sizeof(T);
    }
    return curr_loc;
  }
  // assumes that the head has data, and it is full up until start
  template <bool head_in_place>
  size_t used_size_simple_with_start(int64_t start) {
    int64_t curr_loc = start;
    while (unaligned_load<uint16_t>(&array[curr_loc]) != 0) {
      curr_loc += 1;
    }
    if (curr_loc == length_in_bytes) {
      // only possibility is that just the very last byte is empty
      assert(array[curr_loc - 1] == 0);
      curr_loc -= 1;
    }
    if constexpr (head_in_place) {
      curr_loc += sizeof(T);
    }
    ASSERT((uint64_t)curr_loc == used_size_simple<head_in_place>(),
           "got %lu, expected %lu\n", curr_loc, used_size<head_in_place>());
    return curr_loc;
  }
  template <bool head_in_place> size_t used_size_no_overflow() {
#ifdef __AVX2__
    {
      T curr_elem = head;
      if (curr_elem == 0) {
        return 0;
      }
      uint8_t *true_array = array;
      if constexpr (head_in_place) {
        true_array -= sizeof(T);
      }

      uint32_t data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
          _mm256_load_si256((__m256i *)true_array), _mm256_setzero_si256()));
      size_t curr_loc = 0;
      if constexpr (head_in_place) {
        data >>= sizeof(T);
        curr_loc = sizeof(T);
      }
      // bottom sizeof(T) bytes are the head
      uint32_t set_bit_mask = (data ^ (data >> 1U)) ^ (data | (data >> 1U));
      if (set_bit_mask) {
        size_t ret = curr_loc + bsf_word(set_bit_mask);
        ASSERT(ret == used_size_simple<head_in_place>(),
               "ret = %lu, correct = %lu\n", ret,
               used_size_simple<head_in_place>());
        return ret;
      }
      curr_loc = sizeof(__m256i);
      uint32_t last_bit;
      if constexpr (head_in_place) {
        last_bit = data >> (31U - sizeof(T));
      } else {
        last_bit = data >> 31U;
      }
      while (curr_loc < (size_t)length_in_bytes) {
        data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_load_si256((__m256i *)(true_array + curr_loc)),
            _mm256_setzero_si256()));
        uint32_t set_bit_mask = (data ^ (data >> 1U)) ^ (data | (data >> 1U));
        if (set_bit_mask) {
          size_t ret;
          if (data == 0xFFFFFFFFUL) {
            ret = curr_loc - last_bit;
          } else {
            ret = curr_loc + bsf_word(set_bit_mask);
          }
          if (ret != used_size_simple<head_in_place>()) {
            print();
          }
          ASSERT(ret == used_size_simple<head_in_place>(),
                 "ret = %lu, correct = %lu\n", ret,
                 used_size_simple<head_in_place>());
          return ret;
        }
        last_bit = data >> 31U;
        curr_loc += sizeof(__m256i);
      }
      // only possibility is that just the very last byte is empty
      ASSERT(true_array[curr_loc - 1] == 0, "true_array[curr_loc - 1] = %u\n",
             true_array[curr_loc - 1]);

      return length_in_bytes - 1;
    }
#endif
    return used_size_simple<head_in_place>();
  }

  template <bool head_in_place> size_t used_size() {
    T curr_elem = head;
    if (curr_elem == 0) {
      if (get_out_of_place_used_bytes() > 0) {
        if constexpr (head_in_place) {
          return get_out_of_place_used_bytes();
        } else {
          return get_out_of_place_used_bytes() - sizeof(T);
        }
      }
      return 0;
    }
    return used_size_no_overflow<head_in_place>();
  }

  template <bool head_in_place> size_t used_size_with_start(uint64_t start) {
#ifdef __AVX2__
    {
      if (start <= 32 - sizeof(T)) {
        return used_size<head_in_place>();
      }
      uint8_t *true_array = array;
      if constexpr (head_in_place) {
        true_array -= sizeof(T);
        start += sizeof(T);
      }
      uint64_t aligned_start = start & (~(0x1FU));
      uint32_t data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
          _mm256_load_si256((__m256i *)(true_array + aligned_start)),
          _mm256_setzero_si256()));
      uint32_t set_bit_mask = (data ^ (data >> 1U)) ^ (data | (data >> 1U));
      if (set_bit_mask) {
        size_t ret = aligned_start + bsf_word(set_bit_mask);
        ASSERT(ret == used_size_simple<head_in_place>(),
               "ret = %lu, correct = %lu\n", ret,
               used_size_simple<head_in_place>());
        return ret;
      }
      size_t curr_loc = aligned_start;
      uint32_t last_bit = data >> 31U;
      while (curr_loc < (size_t)length_in_bytes) {
        data = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_load_si256((__m256i *)(true_array + curr_loc)),
            _mm256_setzero_si256()));
        uint32_t set_bit_mask = (data ^ (data >> 1U)) ^ (data | (data >> 1U));
        if (set_bit_mask) {
          size_t ret;
          if (data == 0xFFFFFFFFUL) {
            ret = curr_loc - last_bit;
          } else {
            ret = curr_loc + bsf_word(set_bit_mask);
          }
          ASSERT(ret == used_size_simple<head_in_place>(),
                 "ret = %lu, correct = %lu\n", ret,
                 used_size_simple<head_in_place>());
          return ret;
        }
        last_bit = data >> 31U;
        curr_loc += sizeof(__m256i);
      }
      // only possibility is that just the very last byte is empty
      ASSERT(true_array[curr_loc - 1] == 0, "true_array[curr_loc - 1] = %u\n",
             true_array[curr_loc - 1]);
      size_t ret = length_in_bytes - 1;
      if constexpr (head_in_place) {
        ret += sizeof(T);
      }
      ASSERT(ret == used_size_simple<head_in_place>(),
             "got = %lu, correct = %lu\n", ret,
             used_size_simple<head_in_place>());
      return ret;
    }
#endif
    return used_size_simple_with_start<head_in_place>(start);
  }

  void print(bool external = false) {
    std::cout << "##############LEAF##############" << std::endl;

    T curr_elem = head;
    if (curr_elem == 0) {
      if (get_out_of_place_used_bytes() != 0) {
        if (external) {
          printf("*** EXTERNAL SHOULDNT BE HERE **\n");
          return;
        }
        T size_in_bytes = get_out_of_place_temp_size();
        T *ptr = get_out_of_place_pointer();
        auto leaf = delta_compressed_leaf(*ptr, ptr + 1, size_in_bytes);
        std::cout << "LEAF IN EXTERNAL STORAGE" << std::endl;
        std::cout << "used bytes = " << get_out_of_place_used_bytes()
                  << " last written = " << get_out_of_place_last_written()
                  << " temp size = " << get_out_of_place_temp_size()
                  << std::endl;

        leaf.print(true);
        return;
      }
    }
    uint8_t *p = array;
    std::cout << "head=" << curr_elem << std::endl;
    std::cout << "{ ";
    while (p - array < length_in_bytes && (*p != 0 || *(p + 1) != 0)) {
      std::cout << static_cast<uint32_t>(*p) << ", ";
      p += 1;
    }
    std::cout << " }" << std::endl;
    std::cout << "remaining_bytes {";
    while (p - array < length_in_bytes) {
      std::cout << static_cast<uint32_t>(*p) << ", ";
      p += 1;
    }
    std::cout << " }" << std::endl;
    std::cout << "leaf has: ";
    if (curr_elem != 0) {
      std::cout << curr_elem << ", ";
    }
    int64_t curr_loc = 0;
    while (curr_loc < length_in_bytes) {
      DecodeResult dr(array + curr_loc, length_in_bytes - curr_loc);
      if (dr.old_size == 0) {
        break;
      }
      curr_elem += dr.difference;
      std::cout << curr_elem << ", ";
      curr_loc += dr.old_size;
    }
    std::cout << std::endl;
    std::cout << "used bytes = " << curr_loc << " out of " << length_in_bytes
              << std::endl;
    // assert(*head_p != 0);
  }
};

template <class T> class uncompressed_leaf {

public:
  static constexpr size_t max_element_size = sizeof(T);
  static constexpr bool compressed = false;
  using value_type = T;
  T &head;
  T *array;

private:
  const uint64_t length_in_elements;

  uint64_t find(T x) const {
    // heads are dealt with seprately
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array[i] == 0 || array[i] >= x) {
        return i;
      }
    }
    return -1;
  }

  static T get_out_of_place_used_bytes(T *data) { return data[0]; }
  static void set_out_of_place_used_bytes(T *data, T bytes) { data[0] = bytes; }

  static T *get_out_of_place_pointer(T *data) {
    T *pointer = nullptr;
    memcpy(&pointer, data + 1, sizeof(T *));
    return pointer;
  }
  static void set_out_of_place_pointer(T *data, T *pointer) {
    memcpy(data + 1, &pointer, sizeof(T *));
  }
  T get_out_of_place_used_bytes() const {
    return get_out_of_place_used_bytes(array);
  }
  void set_out_of_place_used_bytes(T bytes) const {
    set_out_of_place_used_bytes(array, bytes);
  }
  T *get_out_of_place_pointer() const {
    return get_out_of_place_pointer(array);
  }
  void set_out_of_place_pointer(T *pointer) const {
    set_out_of_place_pointer(array, pointer);
  }

public:
  uncompressed_leaf(T &head, void *array_, uint64_t length)
      : head(head), array(static_cast<T *>(array_)),
        length_in_elements(length / sizeof(T)) {}

  // Input: pointer to the start of this merge in the batch, end of batch,
  // value in the PMA at the next head (so we know when to stop merging)
  // Output: returns a tuple (ptr to where this merge stopped in the batch,
  // number of distinct elements merged in, and number of bytes used in this
  // leaf)
  template <bool head_in_place>
  std::tuple<T *, uint64_t, uint64_t> merge_into_leaf(T *batch_start,
                                                      T *batch_end, T end_val) {
    // case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      // printf("case 1, batch start %p, end %p, end val %u\n", batch_start,
      //        batch_end, end_val);
      // if (batch_start + 1 < batch_end) {
      //   printf("next val %u\n", batch_start[1]);
      // }
      auto [inserted, byte_count] = insert<head_in_place>(*batch_start);

      return {batch_start + 1, inserted, byte_count};
    }
    // case 2: more than 1 elt from batch
    // two-finger merge into extra space, might overflow leaf
    // printf("case 2\n");
    uint64_t temp_size = length_in_elements;

    // get enough size for the batch
    while (batch_start + temp_size < batch_end &&
           batch_start[temp_size] < end_val) {
      temp_size *= 2;
    }
    // get enough size for the leaf
    temp_size += length_in_elements;

    T *temp_arr = (T *)malloc(sizeof(T) * (temp_size + 1));
    T *batch_ptr = batch_start;
    T *leaf_ptr = array;
    T *temp_ptr = temp_arr;

    uint64_t distinct_batch_elts = 0;
    T *leaf_end = array + length_in_elements;
    T last_written = 0;

    // merge into temp space
    // everything that needs to go before the head
    while (batch_ptr < batch_end && *batch_ptr < head) {
      // if duplicates in batch, skip
      if (*batch_ptr == last_written) {
        batch_ptr++;
        continue;
      }
      *temp_ptr = *batch_ptr;
      batch_ptr++;
      distinct_batch_elts++;
      last_written = *temp_ptr;
      temp_ptr++;
    }

    // deal with the head
    *temp_ptr = head;
    last_written = head;
    temp_ptr++;

    // the standard merge
    while (batch_ptr < batch_end && batch_ptr[0] < end_val &&
           leaf_ptr < leaf_end && leaf_ptr[0] > 0) {
      // if duplicates in batch, skip
      if (*batch_ptr == last_written) {
        batch_ptr++;
        continue;
      }

      // otherwise, do a step of the merge
      // bool batch_element_smaller = *batch_ptr < *leaf_ptr;
      // bool elements_equal = *batch_ptr == *leaf_ptr;
      // *temp_ptr = (batch_element_smaller) ? *batch_ptr : *leaf_ptr;
      // distinct_batch_elts += batch_element_smaller;
      // leaf_ptr += !batch_element_smaller;
      // batch_ptr += elements_equal + batch_element_smaller;
      if (*leaf_ptr == *batch_ptr) {
        *temp_ptr = *leaf_ptr;
        leaf_ptr++;
        batch_ptr++;
      } else if (*leaf_ptr > *batch_ptr) {
        *temp_ptr = *batch_ptr;
        batch_ptr++;
        distinct_batch_elts++;
      } else {
        *temp_ptr = *leaf_ptr;
        leaf_ptr++;
      }

      last_written = *temp_ptr;
      temp_ptr++;
    }

    // write rest of the batch if it exists
    while (batch_ptr < batch_end && batch_ptr[0] < end_val) {
      if (*batch_ptr == last_written) {
        batch_ptr++;
        continue;
      }
      *temp_ptr = *batch_ptr;
      batch_ptr++;
      distinct_batch_elts++;

      last_written = *temp_ptr;
      temp_ptr++;
    }
    // write rest of the leaf it exists
    while (leaf_ptr < leaf_end && leaf_ptr[0] > 0) {
      *temp_ptr = *leaf_ptr;
      leaf_ptr++;
      temp_ptr++;
    }

    uint64_t used_elts = temp_ptr - temp_arr;
    // check if you can fit in the leaf
    if (used_elts <= length_in_elements) {
      // printf("\tNO OVERFLOW\n");
      // dest, src, size
      head = temp_arr[0];
      memcpy(array, temp_arr + 1, (used_elts - 1) * sizeof(T));
      free(temp_arr);
    } else { // special write for when you don't fit
      // printf("\tYES OVERFLOW\n");
      head = 0; // special case head
      set_out_of_place_used_bytes(used_elts * sizeof(T));
      set_out_of_place_pointer(temp_arr);
    }
#if DEBUG == 1
    if (!check_increasing_or_zero()) {
      print();
      assert(false);
    }
#endif

    return {batch_ptr, distinct_batch_elts,
            used_elts * sizeof(T) -
                ((head_in_place) ? 0 : sizeof(T)) /* for the head*/};
  }

  // Input: pointer to the start of this merge in the batch, end of batch,
  // value in the PMA at the next head (so we know when to stop merging)
  // Output: returns a tuple (ptr to where this merge stopped in the batch,
  // number of distinct elements striped_from, and number of bytes used in
  // this leaf)
  template <bool head_in_place>
  std::tuple<T *, uint64_t, uint64_t> strip_from_leaf(T *batch_start,
                                                      T *batch_end, T end_val) {
    // case 1: only one element from the batch goes into the leaf
    if (batch_start + 1 == batch_end || batch_start[1] >= end_val) {
      // printf("case 1, batch start %p, end %p, end val %u\n", batch_start,
      //        batch_end, end_val);
      // if (batch_start + 1 < batch_end) {
      //   printf("next val %u\n", batch_start[1]);
      // }
      auto [removed, byte_count] = remove<head_in_place>(*batch_start);
      // TODO(AUTHOR) get more info from insert so we don't need to call
      // used_size here
      return {batch_start + 1, removed, byte_count};
    }
    // case 2: more than 1 elt from batch
    // two-finger merge into side array for ease
    // printf("case 2\n");
    T *batch_ptr = batch_start;
    T *front_pointer = array;
    T *back_pointer = array;

    uint64_t distinct_batch_elts = 0;
    T *leaf_end = array + length_in_elements;

    // everything that needs to go before the head
    while (batch_ptr < batch_end && *batch_ptr < head) {
      batch_ptr++;
    }
    bool head_done = false;
    if ((batch_ptr < batch_end && *batch_ptr != head) ||
        batch_ptr == batch_end) {
      head_done = true;
    } else {
      distinct_batch_elts++;
    }

    // merge into temp space
    while (batch_ptr < batch_end && batch_ptr[0] < end_val &&
           front_pointer<leaf_end && * front_pointer> 0) {

      // otherwise, do a step of the merge
      if (*front_pointer == *batch_ptr) {
        front_pointer++;
        batch_ptr++;
        distinct_batch_elts++;
      } else if (*front_pointer > *batch_ptr) {
        batch_ptr++;
      } else {
        if (!head_done) {
          head = *front_pointer;
          head_done = true;
        } else {
          *back_pointer = *front_pointer;
          back_pointer++;
        }
        front_pointer++;
      }
    }
    // write rest of the leaf it exists
    while (front_pointer<leaf_end && * front_pointer> 0) {
      if (!head_done) {
        head = *front_pointer;
        head_done = true;
      } else {
        *back_pointer = *front_pointer;
        back_pointer++;
      }
      front_pointer++;
    }

    uint64_t used_elts = back_pointer - array;
    if (!head_done) {
      head = 0;
      set_out_of_place_used_bytes(0);
    }

    // clear the rest of the space
    memset(back_pointer, 0, (front_pointer - back_pointer) * sizeof(T));
    uint64_t used_bytes = used_elts * sizeof(T);
    if (head != 0) {
      if constexpr (head_in_place) {
        used_bytes += sizeof(T);
      }
    }

    return {batch_ptr, distinct_batch_elts, used_bytes};
  }

  template <bool head_in_place, typename F>
  static std::pair<uncompressed_leaf<T>, uint64_t>
  merge_debug(T *array, uint64_t num_leaves, uint64_t leaf_size_in_bytes,
              uint64_t leaf_start_index, F index_to_head) {
    uint64_t leaf_size = leaf_size_in_bytes / sizeof(T);

    // giving it an extra leaf_size_in_bytes to ensure we don't write off the
    // end without needed to be exact about what we are writing for some extra
    // performance
    uint64_t dest_size = leaf_size_in_bytes;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      if (index_to_head(leaf_start_index + i) == 0) {
        dest_size += get_out_of_place_used_bytes(array + src_idx +
                                                 ((head_in_place) ? 1 : 0));
      } else {
        dest_size += leaf_size_in_bytes;
      }
    }

    // printf("mallocing size %u\n", dest_size);
    uint64_t memory_size = ((dest_size + 31) / 32) * 32;
    T *merged_arr = (T *)(aligned_alloc(32, memory_size));

    uint64_t start = 0;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      T *leaf_data_start = array + src_idx + ((head_in_place) ? 1 : 0);

      // if reading overflowed leaf, copy in the temporary storage
      if (index_to_head(leaf_start_index + i) == 0 &&
          get_out_of_place_used_bytes(leaf_data_start) != 0) {
        // dest, src, size
        T *ptr_to_temp = get_out_of_place_pointer(leaf_data_start);
        memcpy(merged_arr + start, ptr_to_temp,
               get_out_of_place_used_bytes(leaf_data_start));
        start += get_out_of_place_used_bytes(leaf_data_start) / sizeof(T);
      } else { // otherwise, reading regular leaf
        merged_arr[start] = index_to_head(leaf_start_index + i);
        start += (index_to_head(leaf_start_index + i) != 0);
        uint64_t local_start = start;
        if constexpr (head_in_place) {
          for (uint64_t j = 0; j < leaf_size - 1; j++) {
            merged_arr[local_start + j] = array[j + src_idx + 1];
            start += (array[j + src_idx + 1] != 0);
          }
        } else {
          for (uint64_t j = 0; j < leaf_size; j++) {
            merged_arr[local_start + j] = array[j + src_idx];
            start += (array[j + src_idx] != 0);
          }
        }
      }
    }

    // fill in the rest of the leaf with 0
    for (uint64_t i = start; i < memory_size / sizeof(T); i++) {
      merged_arr[i] = 0;
    }

    uncompressed_leaf result(merged_arr[0], merged_arr + 1,
                             memory_size - sizeof(T));

    // +1 for head
    return {result, start};
  }

  // Inputs: start of PMA node , number of leaves we want to merge, size of
  // leaf in bytes, number of nonempty bytes in range
  // if we re after an insert that means head being zero means data is stored
  // in an extra storage, if we are after a delete then heads can just be
  // empty from having no data

  // returns: merged leaf, number of full elements in leaf
  template <bool head_in_place, bool have_densities, typename F,
            typename density_array_type>
  static std::pair<uncompressed_leaf<T>, uint64_t>
  parallel_merge(T *array, uint64_t num_leaves, uint64_t leaf_size_in_bytes,
                 uint64_t leaf_start_index, F index_to_head,
                 density_array_type density_array) {
#if DEBUG == 1
    auto checker = merge_debug<head_in_place>(
        array, num_leaves, leaf_size_in_bytes, leaf_start_index, index_to_head);

#endif
    uint64_t leaf_size = leaf_size_in_bytes / sizeof(T);
    std::vector<uint64_t> bytes_per_leaf(num_leaves);
    ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
      uncompressed_leaf l(index_to_head(leaf_start_index + i),
                          array + i * leaf_size + ((head_in_place) ? 1 : 0),
                          leaf_size_in_bytes -
                              ((head_in_place) ? sizeof(T) : 0));
      if constexpr (have_densities) {
        if (density_array[leaf_start_index + i] ==
            std::numeric_limits<uint16_t>::max()) {
          bytes_per_leaf[i] = l.template used_size<head_in_place>();
        } else {
          bytes_per_leaf[i] = density_array[leaf_start_index + i];
        }
      } else {
        bytes_per_leaf[i] = l.template used_size<head_in_place>();
      }
      // for the head
      if constexpr (!head_in_place) {
        if (l.head != 0 || l.get_out_of_place_used_bytes() != 0) {
          bytes_per_leaf[i] += sizeof(T);
        }
      }
    });
    uint64_t total_size;
#if CILK == 0
    total_size = prefix_sum_inclusive(bytes_per_leaf);
#else
    if (num_leaves > 1 << 15) {
      total_size = parlay::scan_inclusive_inplace(bytes_per_leaf);
    } else {
      total_size = prefix_sum_inclusive(bytes_per_leaf);
    }
#endif
    uint64_t memory_size = ((total_size + 31) / 32) * 32;
    T *merged_arr = (T *)(aligned_alloc(32, memory_size));
    ParallelTools::parallel_for(0, num_leaves, [&](uint64_t i) {
      uint64_t start = 0;
      if (i > 0) {
        start = bytes_per_leaf[i - 1] / sizeof(T);
      }
      uint64_t end = bytes_per_leaf[i] / sizeof(T);
      T *source = array + i * leaf_size + ((head_in_place) ? 1 : 0);
      if (index_to_head(leaf_start_index + i) == 0 &&
          get_out_of_place_used_bytes(source) != 0) {
        // get the point from extra storage
        memcpy(merged_arr + start, get_out_of_place_pointer(source),
               get_out_of_place_used_bytes(source));
        free(get_out_of_place_pointer(source));
      } else if (index_to_head(leaf_start_index + i) != 0) {
        if constexpr (!head_in_place) {
          merged_arr[start] = index_to_head(leaf_start_index + i);
          start += 1;
        }
        std::memcpy(merged_arr + start, source - ((head_in_place) ? 1 : 0),
                    (end - start) * sizeof(T));
      }
    });
    // ParallelTools::parallel_for((total_size / sizeof(T)),
    //                             memory_size / sizeof(T),
    //                             [&](uint64_t i) { merged_arr[i] = 0; });
    uncompressed_leaf result(merged_arr[0], merged_arr + 1,
                             memory_size - sizeof(T));
#if DEBUG == 1
    bool good_shape = true;
    if (checker.second != total_size / sizeof(T)) {
      printf("bad total size, got %lu, expected %lu\n", total_size / sizeof(T),
             checker.second);
      good_shape = false;
    }
    if (checker.first.head != result.head) {
      printf("bad head, got %lu, expected %lu\n", (uint64_t)result.head,
             (uint64_t)checker.first.head);
      good_shape = false;
    }
    for (uint64_t i = 0; i < result.length_in_elements; i++) {
      if (checker.first.array[i] != result.array[i]) {
        printf("got bad value in array in position %lu, got %lu, expected %lu, "
               "check length = %lu, got length = %lu\n",
               i, (uint64_t)result.array[i], (uint64_t)checker.first.array[i],
               checker.second, total_size / sizeof(T));
        good_shape = false;
      }
    }
    free(checker.first.array - 1);
    assert(good_shape);

#endif
    return {result, total_size / sizeof(T)};
  }

  template <bool head_in_place, bool have_densities, typename F,
            typename density_array_type>
  static std::pair<uncompressed_leaf<T>, uint64_t>
  merge(T *array, uint64_t num_leaves, uint64_t leaf_size_in_bytes,
        uint64_t leaf_start_index, F index_to_head,
        density_array_type density_array) {
#if CILK == 1
    if (num_leaves > ParallelTools::getWorkers() * 100U) {
      return parallel_merge<head_in_place, have_densities>(
          array, num_leaves, leaf_size_in_bytes, leaf_start_index,
          index_to_head, density_array);
    }
#endif

#if DEBUG == 1
    auto checker = merge_debug<head_in_place>(
        array, num_leaves, leaf_size_in_bytes, leaf_start_index, index_to_head);

#endif
    uint64_t leaf_size = leaf_size_in_bytes / sizeof(T);

    // giving it an extra leaf_size_in_bytes to ensure we don't write off the
    // end without needed to be eact about what we are writing for some extra
    // performance
    uint64_t dest_size = leaf_size_in_bytes;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      if (index_to_head(leaf_start_index + i) == 0) {
        dest_size += get_out_of_place_used_bytes(array + src_idx +
                                                 ((head_in_place) ? 1 : 0));
      } else {
        dest_size += leaf_size_in_bytes;
      }
    }

    // printf("mallocing size %u\n", dest_size);
    uint64_t memory_size = ((dest_size + 31) / 32) * 32;
    T *merged_arr = (T *)(aligned_alloc(32, memory_size));

    uint64_t start = 0;
    for (uint64_t i = 0; i < num_leaves; i++) {
      uint64_t src_idx = i * leaf_size;
      T *leaf_data_start = array + src_idx + ((head_in_place) ? 1 : 0);

      // if reading overflowed leaf, copy in the temporary storage
      if (index_to_head(leaf_start_index + i) == 0 &&
          get_out_of_place_used_bytes(leaf_data_start) != 0) {
        // dest, src, size
        T *ptr_to_temp = get_out_of_place_pointer(leaf_data_start);
        memcpy(merged_arr + start, ptr_to_temp,
               get_out_of_place_used_bytes(leaf_data_start));

        free(ptr_to_temp); // release temp storage

        start += get_out_of_place_used_bytes(leaf_data_start) / sizeof(T);
      } else { // otherwise, reading regular leaf
        merged_arr[start] = index_to_head(leaf_start_index + i);
        start += (index_to_head(leaf_start_index + i) != 0);
        uint64_t local_start = start;
        if constexpr (head_in_place) {
          for (uint64_t j = 0; j < leaf_size - 1; j++) {
            merged_arr[local_start + j] = array[j + src_idx + 1];
            if constexpr (!have_densities) {
              start += (array[j + src_idx + 1] != 0);
            }
          }
        } else {
          for (uint64_t j = 0; j < leaf_size; j++) {
            merged_arr[local_start + j] = array[j + src_idx];
            if constexpr (!have_densities) {
              start += (array[j + src_idx] != 0);
            }
          }
        }
        if constexpr (have_densities) {
          start += density_array[leaf_start_index + i] / sizeof(T);
          if constexpr (head_in_place) {
            start -= (index_to_head(leaf_start_index + i) != 0);
          }
        }
      }
    }

    // fill in the rest of the leaf with 0
    for (uint64_t i = start; i < memory_size / sizeof(T); i++) {
      merged_arr[i] = 0;
    }

    uncompressed_leaf result(merged_arr[0], merged_arr + 1,
                             memory_size - sizeof(T));
#if DEBUG == 1
    bool good_shape = true;
    if (start != checker.second) {
      printf("got bad num elements, got %lu, expected %lu\n", start,
             checker.second);
      good_shape = false;
    }
    if (!result.check_increasing_or_zero()) {
      result.print();
      good_shape = false;
    }
    if (checker.first.head != result.head) {
      printf("bad head, got %lu, expected %lu\n", (uint64_t)result.head,
             (uint64_t)checker.first.head);
      good_shape = false;
    }
    for (uint64_t i = 0; i < result.length_in_elements; i++) {
      if (checker.first.array[i] != result.array[i]) {
        printf("got bad value in array in position %lu, got %lu, expected %lu, "
               "check length = %lu, got length = %lu\n",
               i, (uint64_t)result.array[i], (uint64_t)checker.first.array[i],
               checker.second, memory_size - sizeof(T));
        good_shape = false;
      }
    }
    assert(good_shape);
    free(checker.first.array - 1);

#endif
    // +1 for head
    return {result, start};
  }

  // input: a merged leaf in delta-compressed format
  // input: number of leaves to split into
  // input: number of elements in the input leaf
  // input: number of elements per output leaf
  // input: pointer to the start of the output area to write to (requires that
  // you have num_leaves * num_bytes bytes available here to write to)
  // output: split input leaf into num_leaves leaves, each with
  // num_output_bytes bytes
  template <bool head_in_place, bool store_densities, typename F,
            typename density_array_type>
  void split(uint64_t num_leaves, uint64_t num_elements,
             uint64_t bytes_per_leaf, T *dest_region, uint64_t leaf_start_index,
             F index_to_head, density_array_type density_array) const {
    uint64_t elements_per_leaf = bytes_per_leaf / sizeof(T);

    // printf("num leaves %lu, num elements %lu, bytes per leaf %lu\n",
    // num_leaves,
    //        num_elements, bytes_per_leaf);
    // print();
    // approx occupied bytes per leaf
    uint64_t count_per_leaf = num_elements / num_leaves;
    uint64_t extra = num_elements % num_leaves;
    // printf("count_per_leaf = %lu, extra = %lu\n", count_per_leaf, extra);

    assert(count_per_leaf + (extra > 0) <= elements_per_leaf);

    if ((CILK == 0) || num_leaves <= 1UL << 12UL) {
      for (uint64_t i = 0; i < num_leaves; i++) {
        uint64_t j3 = count_per_leaf * i + std::min(i, extra) - 1;
        if (i == 0) {
          index_to_head(leaf_start_index) = head;
          j3 = 0;
        } else {
          // printf("index = %lu, element = %lu\n", j3, (uint64_t)array[j3]);
          index_to_head(leaf_start_index + i) = array[j3];
          j3 += 1;
        }
        uint64_t out = i * elements_per_leaf;
        // -1 for head
        uint64_t count_for_leaf = count_per_leaf + (i < extra) - 1;
        if (count_for_leaf == std::numeric_limits<uint64_t>::max()) {
          count_for_leaf = 0;
        }
        ASSERT(j3 + count_for_leaf <= length_in_elements,
               "j3 = %lu, count_for_leaf = %lu, length_in_elements = %lu\n", j3,
               count_for_leaf, length_in_elements);
        if constexpr (head_in_place) {
          out += 1;
        }
        memcpy(dest_region + out, array + j3, count_for_leaf * sizeof(T));
        if constexpr (store_densities) {
          density_array[leaf_start_index + i] = count_for_leaf * sizeof(T);
          if constexpr (head_in_place) {
            // for head
            if (index_to_head(leaf_start_index + i) != 0) {
              density_array[leaf_start_index + i] += sizeof(T);
            }
          }
        }
        if constexpr (head_in_place) {
          memset(dest_region + out + count_for_leaf, 0,
                 (elements_per_leaf - count_for_leaf - 1) * sizeof(T));
        } else {
          memset(dest_region + out + count_for_leaf, 0,
                 (elements_per_leaf - count_for_leaf) * sizeof(T));
        }
      }
    } else {
      {
        // first loop not in parallel due to weird compiler behavior
        uint64_t i = 0;
        uint64_t j3 = 0;
        index_to_head(leaf_start_index) = head;
        uint64_t out = 0;
        // -1 for head
        uint64_t count_for_leaf = count_per_leaf + (i < extra) - 1;
        if (count_for_leaf == std::numeric_limits<uint64_t>::max()) {
          count_for_leaf = 0;
        }
        ASSERT(j3 + count_for_leaf <= length_in_elements,
               "j3 = %lu, count_for_leaf = %lu, length_in_elements = %lu\n", j3,
               count_for_leaf, length_in_elements);
        if constexpr (head_in_place) {
          out += 1;
        }
        memcpy(dest_region + out, array + j3, count_for_leaf * sizeof(T));
        if constexpr (store_densities) {
          density_array[leaf_start_index + i] = count_for_leaf * sizeof(T);
          if constexpr (head_in_place) {
            // for head
            if (index_to_head(leaf_start_index) != 0) {
              density_array[leaf_start_index + i] += sizeof(T);
            }
          }
        }
        if constexpr (head_in_place) {
          memset(dest_region + out + count_for_leaf, 0,
                 (elements_per_leaf - count_for_leaf - 1) * sizeof(T));
        } else {
          memset(dest_region + out + count_for_leaf, 0,
                 (elements_per_leaf - count_for_leaf) * sizeof(T));
        }
      }
      // seperate loops due to more weird compiler behavior
      ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
        uint64_t j3 = count_per_leaf * i + std::min(i, extra);
        index_to_head(leaf_start_index + i) = array[j3 - 1];
      });
      ParallelTools::parallel_for(1, num_leaves, [&](uint64_t i) {
        uint64_t j3 = count_per_leaf * i + std::min(i, extra);
        uint64_t out = i * elements_per_leaf;
        // -1 for head
        uint64_t count_for_leaf = count_per_leaf + (i < extra) - 1;
        if (count_for_leaf == std::numeric_limits<uint64_t>::max()) {
          count_for_leaf = 0;
        }
        ASSERT(j3 + count_for_leaf <= length_in_elements,
               "j3 = %lu, count_for_leaf = %lu, length_in_elements = %lu\n", j3,
               count_for_leaf, length_in_elements);
        if constexpr (head_in_place) {
          out += 1;
        }
        memcpy(dest_region + out, array + j3, count_for_leaf * sizeof(T));
        if constexpr (store_densities) {
          density_array[leaf_start_index + i] = count_for_leaf * sizeof(T);
          if constexpr (head_in_place) {
            // for head
            if (index_to_head(leaf_start_index + i) != 0) {
              density_array[leaf_start_index + i] += sizeof(T);
            }
          }
        }
        if constexpr (head_in_place) {
          memset(dest_region + out + count_for_leaf, 0,
                 (elements_per_leaf - count_for_leaf - 1) * sizeof(T));
        } else {
          memset(dest_region + out + count_for_leaf, 0,
                 (elements_per_leaf - count_for_leaf) * sizeof(T));
        }
      });
    }
  }
  // inserts an element
  // first return value indicates if something was inserted
  // if something was inserted the second value tells you the current size
  template <bool head_in_place> std::pair<bool, size_t> insert(T x) {
    assert(array[length_in_elements - 1] == 0);
    if (x == head) {
      return {false, 0};
    }
    if (x < head) {
      T temp = head;
      head = x;
      x = temp;
    }
    if (head == 0) {
      head = x;
      return {true, (head_in_place) ? sizeof(T) : 0};
    }
    uint64_t fr = find(x);
    if (array[fr] == x) {
      return {false, 0};
    }
    size_t num_elements = fr + 1;
    for (uint64_t i = length_in_elements - 1; i >= fr + 1; i--) {
      num_elements += (array[i - 1] != 0);
      array[i] = array[i - 1];
    }
    array[fr] = x;
    return {true, num_elements * sizeof(T) + ((head_in_place) ? sizeof(T) : 0)};
  }

  // removes an element
  // first return value indicates if something was removed
  // if something was removed the second value tells you the current size
  template <bool head_in_place> std::pair<bool, size_t> remove(T x) {
    if (head == 0 || x < head) {
      return {false, 0};
    }
    if (x == head) {
      head = array[0];
      x = array[0];
      if (x == 0) {
        return {true, 0};
      }
    }
    uint64_t fr = find(x);
    if (array[fr] != x) {
      return {false, 0};
    }
    std::memmove(array + fr, array + fr + 1,
                 (length_in_elements - fr - 1) * sizeof(T));

    array[length_in_elements - 1] = 0;
    size_t num_elements = fr;
    for (uint64_t i = fr; i < length_in_elements; i++) {
      if (array[i] == 0) {
        break;
      }
      num_elements += 1;
    }
    return {true, num_elements * sizeof(T) + ((head_in_place) ? sizeof(T) : 0)};
  }

  bool contains(T x) const {
    if (x < head) {
      return false;
    }
    if (x == head) {
      return true;
    }
    uint64_t fr = find(x);
    return array[fr] == x;
  }

  bool debug_contains(T x) const {
    if (head == 0) {
      T size_in_bytes = get_out_of_place_used_bytes();
      T *ptr = get_out_of_place_pointer();
      auto leaf = uncompressed_leaf(*ptr, ptr + 1, size_in_bytes);
      return leaf.contains(x);
    }
    return contains(x);
  }

  template <bool head_in_place = false> [[nodiscard]] uint64_t sum() const {
    if constexpr (head_in_place) {
      uint64_t curr_sum = 0;
      curr_sum += array[-1];
      for (uint64_t i = 0; i < length_in_elements; i++) {
        curr_sum += array[i];
      }
      return curr_sum;
    } else {
      uint64_t curr_sum = 0;
      for (uint64_t i = 0; i < length_in_elements; i++) {
        curr_sum += array[i];
      }
      return head + curr_sum;
    }
  }

  template <bool no_early_exit, class F> bool map(F &f) const {
    if (head == 0) {
      return false;
    }
    if constexpr (!no_early_exit) {
      if (f(head)) {
        return true;
      }
    } else {
      f(head);
    }
    for (uint64_t i = 0; i < length_in_elements; i++) {
      T curr_elem = array[i];
      if (curr_elem == 0) {
        return false;
      }
      if constexpr (!no_early_exit) {
        if (f(curr_elem)) {
          return true;
        }
      } else {
        f(curr_elem);
      }
    }
    return false;
  }

  template <class F> bool map(F &f, T start, T end) const {
    if (head >= end || head == 0) {
      return false;
    }
    if (head >= start && f(head)) {
      return true;
    }
    for (uint64_t i = 0; i < length_in_elements; i++) {
      T curr_elem = array[i];
      if (curr_elem >= end || curr_elem == 0) {
        return false;
      }
      if (curr_elem >= start && f(curr_elem)) {
        return true;
      }
    }
    return false;
  }

  T last() const {
    T last_elem = head;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array[i] == 0) {
        break;
      }
      last_elem = array[i];
    }
    return last_elem;
  }

  template <bool head_in_place>
  [[nodiscard]] size_t used_size_no_overflow() const {
    size_t num_elements = 0;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      num_elements += (array[i] != 0);
    }
    if constexpr (head_in_place) {
      // for the head
      num_elements += 1;
    }
    return num_elements * sizeof(T);
  }

  template <bool head_in_place> [[nodiscard]] size_t used_size() const {
    // if using temp elsewhere
    if (head == 0) {
      if (get_out_of_place_used_bytes() > 0) {
        if constexpr (head_in_place) {
          return get_out_of_place_used_bytes();
        } else {
          return get_out_of_place_used_bytes() - sizeof(T);
        }
      }
      return 0;
    }
    // otherwise not
    return used_size_no_overflow<head_in_place>();
  }

  void print(bool external = false) const {
    std::cout << "##############LEAF##############" << std::endl;
    std::cout << "length_in_elements = " << length_in_elements << std::endl;

    if (head == 0 && get_out_of_place_used_bytes() > 0) {
      if (external) {
        printf("*** EXTERNAL SHOULDNT BE HERE **\n");
        return;
      }
      T size_in_bytes = get_out_of_place_used_bytes();
      T *ptr = get_out_of_place_pointer();
      auto leaf = uncompressed_leaf(*ptr, ptr + 1, size_in_bytes - sizeof(T));
      std::cout << "LEAF IN EXTERNAL STORAGE" << std::endl;
      leaf.print(true);
      return;
    }
    std::cout << "head=" << head << std::endl;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array[i] == 0) {
        std::cout << " - ";
      } else {
        std::cout << array[i] << ", ";
      }
    }
    std::cout << std::endl;
  }

  [[nodiscard]] bool check_increasing_or_zero(bool external = false) const {
    // if using temp elsewhere
    if (head == 0) {
      if (get_out_of_place_used_bytes() > 0) {
        if (external) {
          printf("*** EXTERNAL SHOULDNT BE HERE **\n");
          return false;
        }
        T size_in_bytes = get_out_of_place_used_bytes();
        T *ptr = get_out_of_place_pointer();
        auto leaf = uncompressed_leaf(*ptr, ptr + 1, size_in_bytes - sizeof(T));
        return leaf.check_increasing_or_zero(true);
      }
      return true;
    }

    T check = head;
    for (uint64_t i = 0; i < length_in_elements; i++) {
      if (array[i] == 0) {
        continue;
      }
      if (array[i] <= check) {
        return false;
      }
      check = array[i];
    }
    // otherwise not
    return true;
  }
};
