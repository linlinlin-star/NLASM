定义函数 读取文件(path):
    返回 __read_file__(path)

定义函数 写入文件(path, content):
    __write_file__(path, content)

定义函数 追加文件(path, content):
    __append_file__(path, content)

定义函数 文件存在(path):
    返回 __file_exists__(path)

定义函数 读取行(path):
    返回 __file_readlines__(path)

定义函数 写入行(path, lines):
    __file_writelines__(path, lines)

定义函数 复制文件(src, dst):
    __file_copy__(src, dst)

定义函数 移动文件(src, dst):
    __file_move__(src, dst)

定义函数 删除文件(path):
    __file_delete__(path)

定义函数 文件大小(path):
    返回 __file_size__(path)

定义函数 修改时间(path):
    返回 __file_mtime__(path)
