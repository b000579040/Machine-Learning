# leetcode 771 宝石与石头 Jewels and Stones
import time

start_time = time.time()
S = "aAAbbbb"
J = "aA"

count = 0
for i in S:
    for j in J:
        if ord(i) == ord(j):
            count += 1

print(count)

# count = 0
# for i in S:
#     for k in J:
#         if i == k:
#             count += 1
#             break

print(count)
print('%f' % ((time.time() - start_time) * 1000), "ms")
