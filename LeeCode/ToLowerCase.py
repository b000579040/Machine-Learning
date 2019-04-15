# 709	转换成小写字母
s = ""
str = "Hello"
for i in str:
    if (ord(i) <= 90) & (ord(i) >= 65):
        s += chr(ord(i) + 32)
    else:
        s += i
print(s)
