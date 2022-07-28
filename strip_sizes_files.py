import sys

def point(d):
    if (d[1] > d[0] and d[1] > d[2]) or (d[1]<d[0] and d[1] < d[2]):
        return True
    return False 

data = []


with open(sys.argv[1]) as input:
    numbers = input.readlines()
    numbers = map(lambda x: float(x.strip()), numbers)
    first = numbers[0]
    second = numbers[1]
    third = numbers[2]
    data.append((0, first))
    curr = [first, second, third]
    if point(curr):
        data.append((1, curr[1]))
    index = 2
    for num in numbers[2:]:
        if num == curr[2]:
            index+=1
            continue
        curr.pop(0)
        curr.append(num)
        if point(curr):
            data.append((index, curr[1]))
        index+=1
    data.append((index, numbers[-1]))

with open(sys.argv[2], "w") as output:
    for index, value in data:
        output.write(str(index)+","+str(value)+"\n")
