# 729. 我的日程安排表 I

class MyCalendarTwo(object):

    def __init__(self):
        self.calendar = [[47, 50], [33, 41], [25, 32]]

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """

        rtype = False

        for i in self.calendar:  # 遍历所有日期
            if not ((start >= i[1]) or (end <= i[0])):  # 如果新日期区间在已经安排的时间非交集部分
                return rtype
        self.calendar.append([start, end])
        rtype = True

        return rtype


# Your MyCalendarTwo object will be instantiated and called as such:
obj = MyCalendarTwo()
start = 19
end = 25
print(obj.book(start, end))
