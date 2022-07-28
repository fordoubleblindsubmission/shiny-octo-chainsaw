/*
 * ============================================================================
 *
 *       Filename:  btree.h
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#ifndef _BTREE_H_
#define _BTREE_H_

#include <string.h>

#include <iostream>
#include <limits>
#include <stack>
#include <utility>
#include <vector>
// #include "parallel.h"
#ifndef DEBUG
#define DEBUG 0
#endif

#define WEIGHTED 0

#define MIN_KEYS 64
#define MAX_KEYS (2 * MIN_KEYS - 1)
#define MAX_CHILDREN (2 * MIN_KEYS)

template <class T, class W> class BTreeNode {
public:
  BTreeNode(bool is_leaf);

  // override via leaf and internal
  virtual BTreeNode<T, W> *get_children(int i) const = 0;
  virtual void set_children(int i, BTreeNode<T, W> *a) = 0;
  virtual void move_children(int i, BTreeNode<T, W> *c) = 0;
  virtual void copy_children(int i) = 0;

  const uint32_t find_index(T k) const;
  const uint32_t find_index_branchless(T k) const;
  const uint32_t find_index_branchless_fixedsize1(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize2(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize4(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize8(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize16(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize32(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize64(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize128(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize256(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize512(T k, intptr_t pos) const;
  const uint32_t find_index_branchless_fixedsize(T k) const;
  const BTreeNode<T, W> *find(T k) const;
#if WEIGHTED
  W get_val(T k) const;
#endif
#if WEIGHTED
  bool insertNonFull(T k, W w);
#else
  bool insertNonFull(T k);
#endif
  // The Child c must be full when this function is called
  void splitChild(uint32_t i, BTreeNode<T, W> *c);
  void traverse() const;
  uint64_t sum() const;
  uint64_t psum() const;
  uint32_t get_num_nodes() const;

  class NodeIterator {
  public:
    NodeIterator(){};
    NodeIterator(const BTreeNode<T, W> *node);
#if WEIGHTED
    std::pair<T, W> operator*(void) const;
#else
    T operator*(void) const;
#endif
    void operator++(void);
    bool done(void) const;

  private:
    std::stack<std::pair<const BTreeNode<T, W> *, uint32_t>> stack;
    const BTreeNode<T, W> *cur_node;
    uint32_t cur_idx;
  };

  NodeIterator begin(void) const { return NodeIterator(this); }

  template <typename C, typename D> friend class BTree;

protected:
  T keys[MAX_KEYS];
#if WEIGHTED
  W weights[MAX_KEYS];
#endif
  // BTreeNode<T, W> *children[MAX_CHILDREN];
  uint32_t num_keys;
  bool is_leaf;
};

// -------- NEW NODE TYPES ---------
template <class T, class W> class BTreeNodeInternal : public BTreeNode<T, W> {

public:
  BTreeNodeInternal(); // { children = new BTreeNode<T,W>[MAX_CHILDREN]; };

  BTreeNode<T, W> *get_children(int i) const { return children[i]; }
  void set_children(int i, BTreeNode<T, W> *a) { children[i] = a; }

  // starting from position i, move the children
  void move_children(int i, BTreeNode<T, W> *c) {
    memmove(&children[i], &(((BTreeNodeInternal *)c)->children)[MIN_KEYS],
            (MAX_KEYS) * sizeof(children[0]));
  }
  void copy_children(int i) {
    memcpy(&children[i + 2], &children[i + 1],
           (MAX_CHILDREN - i - 2) * sizeof(children[0]));
  };

  BTreeNode<T, W> *children[MAX_CHILDREN];
};

template <class T, class W> class BTreeNodeLeaf : public BTreeNode<T, W> {
public:
  BTreeNodeLeaf();
  BTreeNode<T, W> *get_children(int i) const { return nullptr; }
  void set_children(int i, BTreeNode<T, W> *a) {}
  void move_children(int i, BTreeNode<T, W> *c) {}
  void copy_children(int i) {}
};

template <class T, class W> class BTree {
public:
  BTree() : root(nullptr) {}

#if WEIGHTED
  W get_val(T k) const { return (root == nullptr) ? 0 : root->get_val(k); }
#endif
  const BTreeNode<T, W> *find(T k) const {
    return (root == nullptr) ? nullptr : root->find(k);
  }
#if WEIGHTED
  bool insert(T k, W w);
#else
  bool insert(T k);
#endif
  void traverse() const {
    if (root != nullptr)
      root->traverse();
  }
  uint64_t sum() const {
    if (root != nullptr)
      return root->psum();
    return 0;
  }

  uint32_t get_num_nodes() const {
    if (root != nullptr)
      return root->get_num_nodes();
    return 0;
  }

  uint64_t get_size(void) const {
    // ASK: will this work with two different derived nodes?
    return get_num_nodes() * sizeof(BTreeNode<T, W>);
  }

  const BTreeNode<T, W> *get_root(void) const { return root; }

  class Iterator {
  public:
    Iterator(){};
    Iterator(const BTree<T, W> *tree) {
      // assert(tree->root != nullptr);
      it = tree->root->begin();
    }
#if WEIGHTED
    std::pair<T, W> operator*(void) const { return *it; }
#else
    T operator*(void) const { return *it; }
#endif
    void operator++(void) { ++it; }
    bool done(void) const { return it.done(); }

  private:
    typename BTreeNode<T, W>::NodeIterator it;
  };

  Iterator begin(void) const { return Iterator(this); }

private:
  BTreeNode<T, W> *root; // Pointer to root node
};

template <class T, class W>
BTreeNode<T, W>::BTreeNode(bool is_leaf) : num_keys(0), is_leaf(is_leaf) {
  memset(keys, 0, MAX_KEYS * sizeof(keys[0]));
#if WEIGHTED
  memset(weights, 0, MAX_KEYS * sizeof(weights[0]));
#endif
}

template <class T, class W>
BTreeNodeInternal<T, W>::BTreeNodeInternal() : BTreeNode<T, W>(0) {
  memset(children, 0, MAX_CHILDREN * sizeof(children[0]));
}

template <class T, class W>
BTreeNodeLeaf<T, W>::BTreeNodeLeaf() : BTreeNode<T, W>(1) {}

#if WEIGHTED
template <class T, class W> W BTreeNode<T, W>::get_val(T k) const {
  uint32_t i = 0;
  while (i < num_keys && k > keys[i])
    i++;

  if (keys[i] == k)
    return weights[i];

  if (is_leaf)
    return 0;

  return get_children(i)->get_val(k);
}
#endif

template <class T, class W>
const uint32_t BTreeNode<T, W>::find_index(T k) const {
  uint32_t left = 0;
  uint32_t right = num_keys - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (keys[mid] == k) {
      left = -1;
      break;
    } else if (keys[mid] < k) {
      left = mid + 1;
    } else {
      if (mid == 0) {
        break;
      }
      right = mid - 1;
    }
  }
  return left;
}

template <class T, class W>
const uint32_t BTreeNode<T, W>::find_index_branchless(T k) const {

  intptr_t pos = -1;
  intptr_t logstep = 32 - __builtin_clz(num_keys) - 1; // bsr

  // special first iteration to make power of 2 intervals
  intptr_t step = num_keys + 1 - uint32_t(intptr_t(1) << logstep);
  pos = (keys[pos + step] < k ? pos + step : pos);
  step = uint32_t(intptr_t(1) << (logstep - 1));

  while (step > 0) {
    pos = (keys[pos + step] < k ? pos + step : pos);
    step >>= 1;
  }
  return pos + 1;
}

// FIXED SIZE FUNCTIONS
template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize512(T k, intptr_t pos) const {
  pos += (keys[pos + 512] < k ? 512 : 0);
  pos += (keys[pos + 256] < k ? 256 : 0);
  pos += (keys[pos + 128] < k ? 128 : 0);
  pos += (keys[pos + 64] < k ? 64 : 0);
  pos += (keys[pos + 32] < k ? 32 : 0);
  pos += (keys[pos + 16] < k ? 16 : 0);
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize256(T k, intptr_t pos) const {
  pos += (keys[pos + 256] < k ? 256 : 0);
  pos += (keys[pos + 128] < k ? 128 : 0);
  pos += (keys[pos + 64] < k ? 64 : 0);
  pos += (keys[pos + 32] < k ? 32 : 0);
  pos += (keys[pos + 16] < k ? 16 : 0);
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize128(T k, intptr_t pos) const {
  pos += (keys[pos + 128] < k ? 128 : 0);
  pos += (keys[pos + 64] < k ? 64 : 0);
  pos += (keys[pos + 32] < k ? 32 : 0);
  pos += (keys[pos + 16] < k ? 16 : 0);
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize64(T k, intptr_t pos) const {
  pos += (keys[pos + 64] < k ? 64 : 0);
  pos += (keys[pos + 32] < k ? 32 : 0);
  pos += (keys[pos + 16] < k ? 16 : 0);
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize32(T k, intptr_t pos) const {
  pos += (keys[pos + 32] < k ? 32 : 0);
  pos += (keys[pos + 16] < k ? 16 : 0);
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize16(T k, intptr_t pos) const {
  pos += (keys[pos + 16] < k ? 16 : 0);
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize8(T k, intptr_t pos) const {
  pos += (keys[pos + 8] < k ? 8 : 0);
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize4(T k, intptr_t pos) const {
  pos += (keys[pos + 4] < k ? 4 : 0);
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize2(T k, intptr_t pos) const {
  pos += (keys[pos + 2] < k ? 2 : 0);
  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t
BTreeNode<T, W>::find_index_branchless_fixedsize1(T k, intptr_t pos) const {

  pos += (keys[pos + 1] < k ? 1 : 0);
  return pos + 1;
}

template <class T, class W>
const uint32_t BTreeNode<T, W>::find_index_branchless_fixedsize(T k) const {

  intptr_t pos = -1;
  intptr_t logstep = 32 - __builtin_clz(num_keys) - 1; // bsr

  // special first iteration to make power of 2 intervals
  intptr_t step = num_keys + 1 - uint32_t(intptr_t(1) << logstep);
  pos = (keys[pos + step] < k ? pos + step : pos);
  // step = uint32_t(intptr_t(1) << (logstep - 1));
  step = logstep - 1;

  // say its 64 - 128 keys
  if (step < 0) {
    return pos + 1;
  } else if (step == 0) {
    return find_index_branchless_fixedsize1(k, pos);
  } else if (step == 1) {
    return find_index_branchless_fixedsize2(k, pos);
  } else if (step == 2) {
    return find_index_branchless_fixedsize4(k, pos);
  } else if (step == 3) {
    return find_index_branchless_fixedsize8(k, pos);
  } else if (step == 4) {
    return find_index_branchless_fixedsize16(k, pos);
  } else if (step == 5) {
    return find_index_branchless_fixedsize32(k, pos);
  } else if (step == 6) {
    return find_index_branchless_fixedsize64(k, pos);
  } else if (step == 7) {
    return find_index_branchless_fixedsize128(k, pos);
  } else if (step == 8) {
    return find_index_branchless_fixedsize256(k, pos);
  } else if (step == 9) {
    return find_index_branchless_fixedsize512(k, pos);
  } else {
    printf("\nfixed size bad, step = %ld \n", step);
    exit(0);
  }
}

template <class T, class W>
const BTreeNode<T, W> *BTreeNode<T, W>::find(T k) const {

  // uint32_t idx = find_index(k);
  // uint32_t idx_branchless = find_index_branchless(k);
  uint32_t idx = find_index_branchless_fixedsize(k);

#if DEBUG
  uint32_t i = 0;
  while (i < num_keys && k > keys[i])
    i++;

  if (keys[i] == k) {
    if (idx != i) {
      printf("\nbad binary search! i = %u, bs_i = %u", i, idx);
    }
    return this;
  }
  if (idx != i) {
    printf("\nbad binary search! i = %u, bs_i = %u", i, idx);
  }

  if (is_leaf)
    return nullptr;

  return get_children(i)->find(k);
#endif

  // check for equality if using linear or branchless bs
  if (keys[idx] == k) {
    return this;
  }

  // std::cout << "num keys " << num_keys << " found i " << i << std::endl;
  if (is_leaf)
    return nullptr;

  return get_children(idx)->find(k);
}

template <class T, class W> void BTreeNode<T, W>::traverse() const {
  uint32_t i;
  for (i = 0; i < num_keys; i++) {
    // If this is not leaf, then before printing key[i],
    // traverse the subtree rooted with child C[i].
    if (!is_leaf)
      get_children(i)->traverse();
    std::cout << keys[i] << " ";
  }

  // Print the subtree rooted with last child
  if (!is_leaf)
    get_children(i)->traverse();
}

template <class T, class W> uint64_t BTreeNode<T, W>::psum() const {
  std::vector<uint64_t> partial_sums(num_keys * 8);
  for (uint32_t i = 0; i < num_keys; i++) {
    // If this is not leaf, then before printing key[i],
    // traverse the subtree rooted with child C[i].
    if (!is_leaf)
      partial_sums[i * 8] += get_children(i)->sum();
    partial_sums[i * 8] += keys[i];
  }

  uint64_t count{0};
  for (uint32_t i = 0; i < num_keys; i++) {
    count += partial_sums[i * 8];
  }

  // Print the subtree rooted with last child
  if (!is_leaf)
    count += get_children(num_keys)->sum();
  return count;
}

template <class T, class W> uint64_t BTreeNode<T, W>::sum() const {
  uint32_t i;
  uint64_t count{0};
  for (i = 0; i < num_keys; i++) {
    // If this is not leaf, then before printing key[i],
    // traverse the subtree rooted with child C[i].
    if (!is_leaf)
      count += get_children(i)->sum();
    count += keys[i];
  }

  // Print the subtree rooted with last child
  if (!is_leaf)
    count += get_children(i)->sum();
  return count;
}

template <class T, class W> uint32_t BTreeNode<T, W>::get_num_nodes() const {
  uint32_t count{1};
  uint32_t i;
  for (i = 0; i < num_keys; i++) {
    if (!is_leaf)
      count += get_children(i)->get_num_nodes();
  }
  // Print the subtree rooted with last child
  if (!is_leaf)
    count += get_children(i)->get_num_nodes();
  return count;
}

template <class T, class W>
#if WEIGHTED
bool BTreeNode<T, W>::insertNonFull(T k, W w) {
#else
bool BTreeNode<T, W>::insertNonFull(T k) {
#endif

  uint32_t idx = find_index_branchless_fixedsize(k);

#if DEBUG
  uint32_t i;
  for (i = 0; i < num_keys; i++) {
    if (keys[i] < k)
      continue;
    else if (k == keys[i]) {
      if (idx != i) {
        printf("\nbad binary search! i = %u, bs_i = %u", i, idx);
      }
#if WEIGHTED
      weights[i] = w;
#endif
      return false;
    } else
      break;
  }

  if (idx != i) {
    printf("\nbad binary search! i = %u, bs_i = %u", i, idx);
  }
#endif

  // check for equality for linear or branchless bs
  if (keys[idx] == k) {
    return false;
  }

  if (is_leaf) { // If this is a leaf node
#if DEBUG
    printf("\tinsert-non-full in leaf\n");
#endif
    memmove(&keys[idx + 1], &keys[idx], (MAX_KEYS - idx - 1) * sizeof(keys[0]));
#if WEIGHTED
    memmove(&weights[idx + 1], &weights[idx],
            (MAX_KEYS - idx - 1) * sizeof(weights[0]));
#endif
    // Insert the new key at found location
#if DEBUG
    printf("\tput key %lu at idx %u\n", k, idx);
#endif
    keys[idx] = k;
#if WEIGHTED
    weights[idx] = w;
#endif
    num_keys = num_keys + 1;
    return true;
  } else { // If this node is not leaf
#if DEBUG
    printf("\tinsert-non-full in internal node\n");
#endif
    if (get_children(idx)->num_keys ==
        MAX_KEYS) { // See if the found child is full
      // If the child is full, then split it
      splitChild(idx, get_children(idx));

      // After split, the middle key of children[idx] goes up and
      // children[idx] is splitted into two. See which of the two
      // is going to have the new key
      if (keys[idx] == k)
        return false;
      else if (keys[idx] < k)
        idx++;
    }
#if WEIGHTED
    return get_children(idx)->insertNonFull(k, w);
#else
    return get_children(idx)->insertNonFull(k);
#endif
  }
}

template <class T, class W>
void BTreeNode<T, W>::splitChild(uint32_t i, BTreeNode<T, W> *c) {
  // Create a new node which is going to store (MIN_KEYS-1) keys
  // of c

  // only memset children if c is an internal node
  if (c->is_leaf) {
#if DEBUG
    printf("\tsplitting from leaf\n");
#endif
    BTreeNodeLeaf<T, W> *z = new BTreeNodeLeaf<T, W>();
    z->num_keys = MIN_KEYS - 1;

    // Copy the last (MIN_KEYS-1) keys of c to z
    memmove(&z->keys[0], &c->keys[MIN_KEYS], (MIN_KEYS - 1) * sizeof(keys[0]));
#if WEIGHTED
    memmove(&z->weights[0], &c->weights[MIN_KEYS],
            (MIN_KEYS - 1) * sizeof(weights[0]));
#endif

#if DEBUG
    printf("\tz keys: ");
    for (int i = 0; i < z->num_keys; i++) {
      printf("%lu\t", z->keys[i]);
    }
    printf("\n");
    printf("set child at idx %u as z\n", i + 1);
#endif

    this->copy_children(i);
    set_children(i + 1, z);
  } else { // c is an internal node
    BTreeNodeInternal<T, W> *z = new BTreeNodeInternal<T, W>();
    z->num_keys = MIN_KEYS - 1;

    // Copy the last (MIN_KEYS-1) keys of c to z
    memmove(&z->keys[0], &c->keys[MIN_KEYS], (MIN_KEYS - 1) * sizeof(keys[0]));
#if WEIGHTED
    memmove(&z->weights[0], &c->weights[MIN_KEYS],
            (MIN_KEYS - 1) * sizeof(weights[0]));
#endif

    z->move_children(0, c);
    // memmove(z->get_children(0), c->get_children(MIN_KEYS),
    //         (MAX_KEYS) * sizeof(get_children(0)));

    this->copy_children(i);
    // memcpy(get_children(i + 2), get_children(i + 1),
    //     (MAX_CHILDREN - i - 2) * sizeof(get_children(0)));

    // Link the new child to this node
    set_children(i + 1, z);
  }

  // Reduce the number of keys in y
  c->num_keys = MIN_KEYS - 1;

  // A key of y will move to this node. Find the location of
  // new key and move all greater keys one space ahead
  memcpy(&keys[i + 1], &keys[i], (MAX_KEYS - i - 1) * sizeof(keys[0]));
#if WEIGHTED
  memcpy(&weights[i + 1], &weights[i], (MAX_KEYS - i - 1) * sizeof(weights[0]));
#endif

  // Copy the middle key of y to this node
  keys[i] = c->keys[MIN_KEYS - 1];
#if WEIGHTED
  weights[i] = c->weights[MIN_KEYS - 1];
#endif

  // Increment count of keys in this node
  num_keys = num_keys + 1;
}

template <class T, class W>
BTreeNode<T, W>::NodeIterator::NodeIterator(const BTreeNode<T, W> *node) {
  // assert(node != nullptr);
  const BTreeNode<T, W> *cur = node;
  while (cur != nullptr) {
    stack.push(std::make_pair(cur, 0));

    // ASK: this might be wrong, b/c previously children[0] of leaves would be
    // set to 0, not nullptr I think
    cur = cur->get_children(0);
  }
  cur_node = stack.top().first;
  cur_idx = 0;
}

template <class T, class W>
#if WEIGHTED
std::pair<T, W> BTreeNode<T, W>::NodeIterator::operator*(void) const {
  return std::make_pair(cur_node->keys[cur_idx], cur_node->weights[cur_idx]);
#else
T BTreeNode<T, W>::NodeIterator::operator*(void) const {
  return cur_node->keys[cur_idx];
#endif
}

template <class T, class W>
void BTreeNode<T, W>::NodeIterator::operator++(void) {
  if (cur_node->is_leaf) { // if we are traversing a leaf node
    if (cur_idx < cur_node->num_keys - 1)
      cur_idx++;
    else { // leaf node is done. Go to parent
      stack.pop();
      if (stack.size()) {
        cur_node = stack.top().first;
        cur_idx = stack.top().second;
        stack.top().second++;
      }
    }
  } else { // if we are traversing an internal node
    if (cur_idx ==
        cur_node->num_keys - 1) { // internal node is done. Go to parent
      stack.pop();
    }
    BTreeNode<T, W> *cur = cur_node->get_children(cur_idx + 1);
    while (cur != nullptr) {
      stack.push(std::make_pair(cur, 0));

      // ASK: same thing-- this might be wrong, b/c previously children[0] of
      // leaves would be set to 0, not nullptr I think
      cur = cur->get_children(0);
    }
    cur_node = stack.top().first;
    cur_idx = 0;
  }
}

template <class T, class W>
bool BTreeNode<T, W>::NodeIterator::done(void) const {
  return !stack.size();
}

template <class T, class W>
#if WEIGHTED
bool BTree<T, W>::insert(T k, W w) {
#else
bool BTree<T, W>::insert(T k) {
#endif
#if DEBUG
  printf("inserting key %lu\n", k);
#endif
  // If tree is empty
  if (root == nullptr) {
    // Allocate memory for root
    root = new BTreeNodeLeaf<T, W>();
    root->keys[0] = k; // Insert key
#if WEIGHTED
    root->weights[0] = w; // Insert weight
#endif
    root->num_keys = 1; // Update number of keys in root
    return true;
  } else { // If tree is not empty
    // If root is full, then tree grows in height
    if (root->num_keys == MAX_KEYS) {
#if DEBUG
      printf("\tsplit root\n");
#endif
      // Allocate memory for new root
      BTreeNode<T, W> *s = new BTreeNodeInternal<T, W>();

      // Make old root as child of new root
      s->set_children(0, root);

      // Split the old root and move 1 key to the new root
      s->splitChild(0, root);
      // printf("\tafter split, root has %lu keys, prev root has %lu keys, right
      // child has %lu keys\n", s->num_keys, root->num_keys,
      // s->get_children(1)->num_keys);

      // New root has two children now.  Decide which of the
      // two children is going to have new key
      int i = 0;
      if (s->keys[0] < k)
        i++;

      // printf("\tsplit pivot is %lu, i = %d\n", s->keys[0], i);
      // Change root
      root = s;
#if WEIGHTED
      return s->get_children(i)->insertNonFull(k, w);
#else
      return s->get_children(i)->insertNonFull(k);
#endif
    } else // If root is not full, call insertNonFull for root
#if WEIGHTED
      return root->insertNonFull(k, w);
#else
           // printf("\tinsert into root with %lu keys\n", root->num_keys);
      return root->insertNonFull(k);
#endif
  }
}

#endif