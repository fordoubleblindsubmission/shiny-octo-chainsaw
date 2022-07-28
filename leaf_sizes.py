
def bsr(num):
    count = 0
    while num != 0:
        count += 1
        num >>= 1
    return count


def leaf_size(n, leaf_blow_up_factor=16):
    loglogN = bsr(bsr(n))
    return leaf_blow_up_factor*(1 << loglogN)


n = 256
while n <= 1 << 64:
    print(n, leaf_size(n))
    n *= 2
