import time

start_time = time.time()

s = "132-(2+2)+12+34-1"

a = []
k = []

for i in s:
    if i != " ":
        a.append(i)


def find(s):
    a = []
    for i in s:
        if i != "(":
            a.append(i)
        elif i == "(":
            find(s[int(s[i]):])
        elif i == ")":
            return a
    return a

print(find(s))

# lastFuHao = 0
# j = []
# last = 0
# re = 0
# for i in a:
#     if i == "+" or i == "-":
#         # last = int(''.join(j))
#         if lastFuHao == 1:
#             re = re + int(''.join(j))
#         elif lastFuHao == 2:
#             re = re - int(''.join(j))
#         else:
#             re = int(''.join(j))
#         j = []
#
#         if i == "+":
#             lastFuHao = 1
#         else:
#             lastFuHao = 2
#     else:
#         j.append(i)
#
# if lastFuHao == 1:
#     re = re + int(''.join(j))
# elif lastFuHao == 2:
#     re = re - int(''.join(j))
# else:
#     re = int(''.join(j))

# print(re)
print('%f' % ((time.time() - start_time) * 1000), "ms")
