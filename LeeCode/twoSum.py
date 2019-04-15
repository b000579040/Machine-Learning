# 1. 两数之和

nums = [2, 7, 11, 15]
target = 9


def twoSum1(nums, target):
    result = []
    for i in range(0, nums.__len__() - 1):
        for j in range(1, nums.__len__() - 1):
            if nums[i] + nums[j] == target:
                result.append(i)
                result.append(j)
                return result


def twoSum2(nums, target):
    dict = {}
    for index, num in enumerate(nums):  # index是序号 num是nums里面的数值
        hash_item = target - num  # 另外一个目标值
        if hash_item in dict:
            return [dict[hash_item], index]  # 存在就返回这两个下标
        dict[num] = index  # 把不符合条件的值存入字典中


a = twoSum1(nums, target)
b = twoSum2(nums, target)
print(twoSum1(nums, target))
print(twoSum2(nums, target))

print("1")
