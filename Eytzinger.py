import sys

from numpy import size


def nextPowerOf2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def index_to_head(index, length):
    if length != nextPowerOf2(length):
        length = nextPowerOf2(length) - 1
    else:
        length = length*2-1

    value_to_check = length >> 1
    length_to_add = (length >> 2) + 1
    current_check = 1
    while length_to_add > 0:
        # print(value_to_check)
        # print(length_to_add)
        # print(current_check)
        if value_to_check == index:
            return current_check - 1

        if index < value_to_check:
            current_check *= 2
            value_to_check -= length_to_add
        else:
            current_check *= 2
            current_check += 1
            value_to_check += length_to_add

        length_to_add >>= 1
    return current_check - 1


def trailing_zeros(num):
    count = 0
    while num & 1 == 0:
        count += 1
        num >>= 1
    return count


def bsr(num):
    count = 0
    while num != 0:
        count += 1
        num >>= 1
    return count


def pop_count(num):
    count = 0
    while num != 0:
        count += num & 1
        num >>= 1
    return count


def index_to_head_test(index, length):
    # print()
    if length != nextPowerOf2(length):
        length = nextPowerOf2(length) - 1
    else:
        length = length*2-1

    pos_0 = trailing_zeros(~index) + 1
    num_bits = pop_count(length)
    # print("length", length, "pos_0", pos_0, "num_bits", num_bits, end="\t")
    # print()
    return (1 << (num_bits - pos_0)) + (index >> pos_0) - 1


num_leaves = int(sys.argv[1])
# for i in range(num_leaves):
#     print(i, end="\t")
#     print(index_to_head(i, num_leaves), end="\t")
#     print(format(i, "04b"), end="\t")
#     print(index_to_head_test(i, num_leaves), end="\t")
#     print()


def Bnary_correct(length, B):
    length_rounded = B
    while length_rounded <= length:
        length_rounded *= B
    vec = [-1 for i in range(length_rounded)]
    index = 0
    magnitude = length_rounded//B
    while magnitude > 0:
        for i in range(magnitude, length_rounded, magnitude):
            if vec[i-1] == -1:
                vec[i-1] = index
                index += 1
        magnitude //= B
    return vec[:length]


# def Bnary_level(index, length_rounded, B):
#     length_rounded //= B
#     level = 0
#     while index % length_rounded != 0:
#         length_rounded //= B
#         level += 1
#     return level


# def Bnary_level_to_start(level, length_rounded, B):
#     num = 1
#     for i in range(level):
#         num *= B
#     return num

# def Bnary(index, length, B, p=False):
#     length_rounded = B
#     while length_rounded < length:
#         length_rounded *= B
#     current_check = 0
#     index += 1
#     level = Bnary_level(index, length_rounded, B)
#     start = Bnary_level_to_start(level, length_rounded, B)
#     size_to_consider = length_rounded//start
#     if p:
#         print("\tindex = {}, length_rounded = {}, B = {}, level = {}, start = {}, size_to_consider = {}".format(
#             index, length_rounded, B, level, start, size_to_consider))
#     return start + (B-1)*((index-1)//(size_to_consider)) + ((index-1) % size_to_consider)//(size_to_consider//B) - 1

def Bnary_level(index, length_rounded, B):
    length_rounded //= B
    level = 0
    while index % length_rounded != 0:
        length_rounded //= B
        level += 1
    return level


def Bnary_level_start(index, length_rounded, B):
    num = 1
    length_rounded //= B
    while index % length_rounded != 0:
        length_rounded //= B
        num *= B
    return num


def Bnary(index, length, B, p=False):
    length_rounded = B
    while length_rounded <= length:
        length_rounded *= B
    index += 1
    start = Bnary_level_start(index, length_rounded, B)
    size_to_consider = length_rounded//start
    return start + (B-1)*((index-1)//(size_to_consider)) + ((index-1) % size_to_consider)//(size_to_consider//B) - 1


B = 5
p = False
correct = Bnary_correct(num_leaves, B)
order_helper = sorted([(correct[i], i) for i in range(num_leaves)])

for i in range(num_leaves):
    if Bnary(i, num_leaves, B) == correct[i]:
        print("correct at i = {}, B = {} got {}".format(
            i, B, Bnary(i, num_leaves, B)))
for c, i in order_helper:
    if Bnary(i, num_leaves, B) != c:
        print("different at i = {}, B = {}, correct was {}, got {}".format(
            i, B, c, Bnary(i, num_leaves, B)))
        Bnary(i, num_leaves, B, True)

if [index_to_head(i, num_leaves) for i in range(num_leaves)] != [index_to_head_test(i, num_leaves) for i in range(num_leaves)]:
    print("issue")


def correct_e_range(start, end, length):
    return [(index_to_head(i, length), i) for i in range(start, end)]


def e_range(start, end, length, value_to_check=-1, length_to_add=-1, current_check=-1):
    # print("start", start, "end", end, "length", length, "value_to_check",
    #       value_to_check, "length_to_add", length_to_add, "current_check", current_check)

    if length != nextPowerOf2(length):
        length = nextPowerOf2(length) - 1
    else:
        length = length*2-1
    if value_to_check == -1:
        value_to_check = length >> 1
    if length_to_add == -1:
        length_to_add = (length >> 2) + 1
    if current_check == -1:
        current_check = 1
    vector = []
    if length_to_add > 0:
        if start <= value_to_check and value_to_check < end:
            vector.append((current_check-1, value_to_check))
        if start < value_to_check:
            vector += e_range(start, min(value_to_check, end), length, value_to_check -
                              length_to_add, length_to_add >> 1, current_check*2)
        if end > value_to_check:
            vector += e_range(max(value_to_check+1, start), end, length, value_to_check +
                              length_to_add, length_to_add >> 1, current_check*2+1)
    else:
        if start <= value_to_check and value_to_check < end:
            vector.append((current_check-1, value_to_check))
    return vector


if len(sys.argv) < 4:
    exit()
start = int(sys.argv[2])
end = int(sys.argv[3])
print("got")
print(sorted(e_range(start, end, num_leaves)))
print("expected")
print(sorted(correct_e_range(start, end, num_leaves)))


if (sorted(e_range(start, end, num_leaves)) != sorted(correct_e_range(start, end, num_leaves))):
    for i in range(num_leaves):
        print(i, end="\t")
        print(index_to_head(i, num_leaves), end="\n")
    print("got")
    print(sorted(e_range(start, end, num_leaves)))
    print("expected")
    print(sorted(correct_e_range(start, end, num_leaves)))


if [index_to_head(i, num_leaves) for i in range(num_leaves)] != [index_to_head_test(i, num_leaves) for i in range(num_leaves)]:
    print("issue")
