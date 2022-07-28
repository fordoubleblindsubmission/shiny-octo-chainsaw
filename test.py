def check_sequntial_bits(i):
    return (i ^ (i >> 1)) != (i | (i >> 1))


def first_sequntial_bit(i):
    return (i ^ (i >> 1)) ^ (i | (i >> 1))


for i in range(100):
    print("{0:b}: ".format(i), check_sequntial_bits(
        i), "{0:b}: ".format(first_sequntial_bit(i)))
