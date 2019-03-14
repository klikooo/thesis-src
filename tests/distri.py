from old.ascad import HW

p = [x for x in range(256)]
r = [x for x in range(256)]


res = [0 for x in range(9)]

for x in p:
    for y in r:
        if HW[y] == 3:
            res[HW[x ^ y]] += 1
            print('{} xor {} = {}'.format(HW[x], HW[y], HW[x^y]))
print('res: {}'.format(res))



