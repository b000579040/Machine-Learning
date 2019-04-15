# 2. 两数相加

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """


r1 = [2, 3, 4]
r2 = [9, 2, 7]

if len(r1) < len(r2):
    for i in range(len(r2) - len(r1)):
        r1.append(0)
elif len(r1) > len(r2):
    for i in range(len(r1) - len(r2)):
        r2.append(0)

r = []
for i in range(len(r2)):
    r.append(r1[i] + r2[i])

for i in range(len(r)):
    if r[i] >= 10:
        if i == len(r) - 1:
            r.append(r[i] % 10)
        else:
            r[i + 1] = r[i + 1] + r[i] % 10
        r[i] = r[i] // 10
print(r)
