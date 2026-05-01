定义函数 当前目录():
    返回 __os_getcwd__()

定义函数 列出目录(path):
    返回 __os_listdir__(path)

定义函数 创建目录(path):
    __os_mkdir__(path)

定义函数 删除文件(path):
    __os_remove__(path)

定义函数 重命名(old, new):
    __os_rename__(old, new)

定义函数 获取环境变量(name):
    返回 __os_getenv__(name)

定义函数 设置环境变量(name, value):
    __os_setenv__(name, value)

定义函数 路径拼接(*parts):
    返回 __os_path_join__(*parts)

定义函数 路径存在(path):
    返回 __os_path_exists__(path)

定义函数 是文件(path):
    返回 __os_path_isfile__(path)

定义函数 是目录(path):
    返回 __os_path_isdir__(path)

定义函数 文件名(path):
    返回 __os_path_basename__(path)

定义函数 目录名(path):
    返回 __os_path_dirname__(path)

定义函数 扩展名(path):
    返回 __os_path_splitext__(path)

定义函数 文件大小(path):
    返回 __os_path_size__(path)
