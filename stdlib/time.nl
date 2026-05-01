定义函数 当前时间():
    返回 __time_now__()

定义函数 格式化时间(timestamp, fmt):
    返回 __time_format__(timestamp, fmt)

定义函数 解析时间(time_str, fmt):
    返回 __time_parse__(time_str, fmt)

定义函数 睡眠(seconds):
    __time_sleep__(seconds)

定义函数 本地时间(timestamp):
    返回 __time_localtime__(timestamp)

定义函数 当前日期时间():
    返回 __time_localtime__(空)

定义函数 年():
    返回 __time_localtime__(空)["年"]

定义函数 月():
    返回 __time_localtime__(空)["月"]

定义函数 日():
    返回 __time_localtime__(空)["日"]

定义函数 时():
    返回 __time_localtime__(空)["时"]

定义函数 分():
    返回 __time_localtime__(空)["分"]

定义函数 秒():
    返回 __time_localtime__(空)["秒"]

定义函数 格式化当前时间(fmt):
    返回 __time_format__(__time_now__(), fmt)

定义函数 时间戳():
    返回 int(__time_now__())
