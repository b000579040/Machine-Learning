class ListNode(object):
    def __init__(self):
        self.val = None
        self.next = None


class ListNode_Handle:
    def __init__(self):
        self.cur_node = None

    def add(self, data):
        node = ListNode()
        node.val = data
        node.next = self.cur_node
        self.cur_node = node
        return node

    def print_ListNode(self, node):
        while node:
            print('\nnode:', node, 'value:', node.val, 'next', node.next)
            node = node.next


l1 = ListNode()
lh = ListNode_Handle()
a = [1, 2, 3]
b = [2, 4, 6]

for i in a:
    l1 = lh.add(i)

lh.print_ListNode(l1)
