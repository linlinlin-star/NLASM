定义函数 长度(arr):
    返回 len(arr)

定义函数 排序(arr):
    返回 sorted(arr)

定义函数 反转(arr):
    返回 reversed(arr)

定义函数 映射(arr, func):
    返回 map(func, arr)

定义函数 过滤(arr, func):
    返回 filter(func, arr)

定义函数 归约(arr, func, init):
    定义 acc = init
    对于 arr 中的每个元素 x:
        acc = func(acc, x)
    返回 acc

定义函数 包含(arr, val):
    对于 arr 中的每个元素 x:
        如果 x == val:
            返回 真
    返回 假

定义函数 去重(arr):
    返回 __array_unique__(arr)

定义函数 拼接(arr, sep):
    定义 result = ""
    定义 i = 0
    对于 arr 中的每个元素 x:
        如果 i > 0:
            result = result + sep
        result = result + str(x)
        i = i + 1
    返回 result

定义函数 查找(arr, val):
    返回 __array_find__(arr, val)

定义函数 计数(arr, val):
    返回 __array_count__(arr, val)

定义函数 展平(arr):
    返回 __array_flat__(arr)

定义函数 拉链(*arrays):
    返回 __array_zip__(*arrays)

定义函数 分块(arr, size):
    返回 __array_chunk__(arr, size)

定义函数 差集(arr1, arr2):
    返回 __array_difference__(arr1, arr2)

定义函数 交集(arr1, arr2):
    返回 __array_intersection__(arr1, arr2)

定义函数 并集(arr1, arr2):
    返回 __array_union__(arr1, arr2)

定义函数 第一个(arr):
    如果 len(arr) > 0:
        返回 arr[0]
    返回 空

定义函数 最后一个(arr):
    如果 len(arr) > 0:
        返回 arr[len(arr) - 1]
    返回 空

定义函数 取前n个(arr, n):
    定义 result = []
    定义 i = 0
    当 i < n 且 i < len(arr):
        result.追加(arr[i])
        i = i + 1
    返回 result

定义函数 取后n个(arr, n):
    定义 start = len(arr) - n
    如果 start < 0:
        start = 0
    定义 result = []
    定义 i = start
    当 i < len(arr):
        result.追加(arr[i])
        i = i + 1
    返回 result
