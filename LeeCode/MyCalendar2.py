# 731. 我的日程安排表 II

class MyCalendarTwo(object):

    def __init__(self):
        # self.calendar = [[47, 50], [1, 10], [27, 36], [40, 47], [20, 27], [15, 23], [10, 18], [27, 36], [17, 25],
        #                  [8, 17], [24, 33],
        #                  [23, 28], [21, 27], [47, 50], [14, 21], [26, 32], [16, 21], [2, 7], [24, 33]]
        self.calendar = []

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """

        rtype = False

        for i in self.calendar:  # 遍历所有日期
            if (start >= i[1]) or (end <= i[0]):  # 如果新日期区间在已经安排的时间非交集部分
                pass
            elif (start >= i[0]) and (end <= i[1]):
                doubleCalendar = self.calendar[self.calendar.index(i) + 1:]
                for j in doubleCalendar:
                    if not ((start >= j[1]) or (end <= j[0])):
                        return rtype

            elif (start >= i[0]) and (end >= i[1]):
                doubleCalendar = self.calendar[self.calendar.index(i) + 1:]
                # newStart = start
                newEnd = i[1]
                for j in doubleCalendar:
                    if not ((start >= j[1]) or (newEnd <= j[0])):
                        return rtype

            elif (start <= i[0]) and (end <= i[1]):
                doubleCalendar = self.calendar[self.calendar.index(i) + 1:]
                newStart = i[0]
                # newEnd = end
                for j in doubleCalendar:
                    if not ((newStart >= j[1]) or (end <= j[0])):
                        return rtype

            elif (start <= i[0]) and (end >= i[1]):
                doubleCalendar = self.calendar[self.calendar.index(i) + 1:]
                newStart = i[0]
                newEnd = i[1]
                for j in doubleCalendar:
                    if not ((newStart >= j[1]) or (newEnd <= j[0])):
                        return rtype

        self.calendar.append([start, end])
        rtype = True

        return rtype


# Your MyCalendarTwo object will be instantiated and called as such:
obj = MyCalendarTwo()
# start = 6
# end = 13
# print(obj.book(start, end))


book = [[22, 29], [12, 17], [20, 27], [27, 36], [24, 31], [23, 28], [47, 50], [23, 30], [24, 29], [19, 25], [19, 27],
        [3, 9], [34, 41], [22, 27], [3, 9], [29, 38], [34, 40], [49, 50], [42, 48], [43, 50], [39, 44], [30, 38],
        [42, 50], [31, 39], [9, 16], [10, 18], [31, 39], [30, 39], [48, 50], [36, 42]]
for i in book:
    start = i[0]
    end = i[1]
    re = obj.book(start, end)
    print(book.index(i) + 1, ": ", re)
