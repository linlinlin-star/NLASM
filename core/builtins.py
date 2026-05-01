from __future__ import annotations

from typing import Any, Callable


class BuiltinRegistry:
    """内置函数注册表 - 管理NLASM可用的内置函数 / Builtin function registry - manages NLASM available built-in functions.

    注册Python标准函数供NLASM程序调用，支持动态扩展。
    Registers Python standard functions for NLASM programs to call, supports dynamic extension.
    """

    def __init__(self) -> None:
        self.functions: dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.functions["abs"] = abs
        self.functions["max"] = max
        self.functions["min"] = min
        self.functions["sum"] = sum
        self.functions["len"] = len
        self.functions["int"] = int
        self.functions["float"] = float
        self.functions["str"] = str
        self.functions["bool"] = bool
        self.functions["type"] = lambda x: type(x).__name__
        self.functions["range"] = lambda *args: list(range(*[int(a) for a in args]))
        self.functions["print"] = print
        self.functions["input"] = input
        self.functions["sorted"] = sorted
        self.functions["reversed"] = lambda x: list(reversed(x))
        self.functions["enumerate"] = lambda x: list(enumerate(x))
        self.functions["zip"] = lambda *args: list(zip(*args))
        self.functions["next"] = lambda it, *default: next(it) if default == () else next(it, default[0])
        self.functions["isinstance"] = isinstance
        self.functions["list"] = list
        self.functions["dict"] = dict
        self.functions["tuple"] = tuple
        self.functions["set"] = set
        self.functions["round"] = round
        self.functions["pow"] = pow
        self.functions["divmod"] = divmod
        self.functions["hex"] = hex
        self.functions["oct"] = oct
        self.functions["bin"] = bin
        self.functions["chr"] = chr
        self.functions["ord"] = ord
        self.functions["id"] = id
        self.functions["hash"] = hash
        self.functions["callable"] = callable
        self.functions["hasattr"] = hasattr
        self.functions["getattr"] = getattr
        self.functions["setattr"] = setattr
        self._register_math()
        self._register_string()
        self._register_array()
        self._register_dict()
        self._register_file()
        self._register_os()
        self._register_time()
        self._register_random()
        self._register_json()
        self._register_concurrency()

    def _register_math(self) -> None:
        import math
        self.functions["__math_sqrt__"] = lambda x: math.sqrt(float(x))
        self.functions["__math_pow__"] = lambda x, y: math.pow(float(x), float(y))
        self.functions["__math_sin__"] = lambda x: math.sin(float(x))
        self.functions["__math_cos__"] = lambda x: math.cos(float(x))
        self.functions["__math_tan__"] = lambda x: math.tan(float(x))
        self.functions["__math_asin__"] = lambda x: math.asin(float(x))
        self.functions["__math_acos__"] = lambda x: math.acos(float(x))
        self.functions["__math_atan__"] = lambda x: math.atan(float(x))
        self.functions["__math_atan2__"] = lambda y, x: math.atan2(float(y), float(x))
        self.functions["__math_hypot__"] = lambda x, y: math.hypot(float(x), float(y))
        self.functions["__math_log__"] = _math_log
        self.functions["__math_log2__"] = lambda x: math.log2(float(x))
        self.functions["__math_log10__"] = lambda x: math.log10(float(x))
        self.functions["__math_ceil__"] = lambda x: math.ceil(float(x))
        self.functions["__math_floor__"] = lambda x: math.floor(float(x))
        self.functions["__math_pi__"] = lambda: math.pi
        self.functions["__math_e__"] = lambda: math.e
        self.functions["__math_radians__"] = lambda x: math.radians(float(x))
        self.functions["__math_degrees__"] = lambda x: math.degrees(float(x))

    def _register_string(self) -> None:
        self.functions["__str_split__"] = lambda s, sep=None: s.split(sep)
        self.functions["__str_join__"] = lambda arr, sep: sep.join(str(x) for x in arr)
        self.functions["__str_replace__"] = lambda s, old, new: s.replace(old, new)
        self.functions["__str_strip__"] = lambda s: s.strip()
        self.functions["__str_lstrip__"] = lambda s, chars=None: s.lstrip(chars)
        self.functions["__str_rstrip__"] = lambda s, chars=None: s.rstrip(chars)
        self.functions["__str_upper__"] = lambda s: s.upper()
        self.functions["__str_lower__"] = lambda s: s.lower()
        self.functions["__str_startswith__"] = lambda s, prefix: s.startswith(prefix)
        self.functions["__str_endswith__"] = lambda s, suffix: s.endswith(suffix)
        self.functions["__str_contains__"] = lambda s, sub: sub in s
        self.functions["__str_find__"] = lambda s, sub, start=0: s.find(sub, int(start))
        self.functions["__str_rfind__"] = lambda s, sub, start=0: s.rfind(sub, int(start))
        self.functions["__str_count__"] = lambda s, sub: s.count(sub)
        self.functions["__str_center__"] = lambda s, width, fillchar=" ": s.center(int(width), fillchar)
        self.functions["__str_ljust__"] = lambda s, width, fillchar=" ": s.ljust(int(width), fillchar)
        self.functions["__str_rjust__"] = lambda s, width, fillchar=" ": s.rjust(int(width), fillchar)
        self.functions["__str_zfill__"] = lambda s, width: s.zfill(int(width))
        self.functions["__str_repeat__"] = lambda s, n: s * int(n)
        self.functions["__str_reverse__"] = lambda s: s[::-1]
        self.functions["__str_capitalize__"] = lambda s: s.capitalize()
        self.functions["__str_title__"] = lambda s: s.title()
        self.functions["__str_isdigit__"] = lambda s: s.isdigit()
        self.functions["__str_isalpha__"] = lambda s: s.isalpha()
        self.functions["__str_isalnum__"] = lambda s: s.isalnum()
        self.functions["__str_isspace__"] = lambda s: s.isspace()
        self.functions["__str_partition__"] = lambda s, sep: list(s.partition(sep))
        self.functions["__str_rpartition__"] = lambda s, sep: list(s.rpartition(sep))
        self.functions["__str_splitlines__"] = lambda s: s.splitlines()
        self.functions["__str_swapcase__"] = lambda s: s.swapcase()

    def _register_array(self) -> None:
        self.functions["__array_find__"] = _array_find
        self.functions["__array_count__"] = lambda arr, val: arr.count(val)
        self.functions["__array_flat__"] = _array_flat
        self.functions["__array_zip__"] = lambda *arrays: [list(group) for group in zip(*arrays)]
        self.functions["__array_chunk__"] = lambda arr, size: [arr[i:i+int(size)] for i in range(0, len(arr), int(size))]
        self.functions["__array_unique__"] = _array_unique
        self.functions["__array_difference__"] = lambda arr1, arr2: [x for x in arr1 if x not in arr2]
        self.functions["__array_intersection__"] = lambda arr1, arr2: [x for x in arr1 if x in arr2]
        self.functions["__array_union__"] = lambda arr1, arr2: _array_unique(arr1 + arr2)

    def _register_dict(self) -> None:
        self.functions["__dict_keys__"] = lambda d: list(d.keys())
        self.functions["__dict_values__"] = lambda d: list(d.values())
        self.functions["__dict_items__"] = lambda d: list(d.items())
        self.functions["__dict_get__"] = lambda d, key, default=None: d.get(key, default)
        self.functions["__dict_has__"] = lambda d, key: key in d
        self.functions["__dict_merge__"] = lambda d1, d2: {**d1, **d2}
        self.functions["__dict_from_list__"] = _dict_from_list
        self.functions["__dict_invert__"] = lambda d: {v: k for k, v in d.items()}

    def _register_file(self) -> None:
        self.functions["__read_file__"] = lambda path: open(path, "r", encoding="utf-8").read()
        self.functions["__write_file__"] = _write_file
        self.functions["__append_file__"] = _append_file
        self.functions["__file_exists__"] = lambda path: __import__("os").path.exists(path)
        self.functions["__file_readlines__"] = lambda path: open(path, "r", encoding="utf-8").readlines()
        self.functions["__file_writelines__"] = _writelines
        self.functions["__file_copy__"] = lambda src, dst: __import__("shutil").copy2(src, dst)
        self.functions["__file_move__"] = lambda src, dst: __import__("shutil").move(src, dst)
        self.functions["__file_delete__"] = lambda path: __import__("os").remove(path)
        self.functions["__file_size__"] = lambda path: __import__("os").path.getsize(path)
        self.functions["__file_mtime__"] = lambda path: __import__("os").path.getmtime(path)

    def _register_os(self) -> None:
        import os
        self.functions["__os_getcwd__"] = lambda: os.getcwd()
        self.functions["__os_listdir__"] = lambda path=".": os.listdir(path)
        self.functions["__os_mkdir__"] = lambda path: os.mkdir(path)
        self.functions["__os_remove__"] = lambda path: os.remove(path)
        self.functions["__os_rename__"] = lambda old, new: os.rename(old, new)
        self.functions["__os_getenv__"] = lambda name, default=None: os.environ.get(name, default)
        self.functions["__os_setenv__"] = lambda name, value: os.environ.__setitem__(name, str(value))
        self.functions["__os_path_join__"] = lambda *parts: os.path.join(*parts)
        self.functions["__os_path_exists__"] = lambda path: os.path.exists(path)
        self.functions["__os_path_isfile__"] = lambda path: os.path.isfile(path)
        self.functions["__os_path_isdir__"] = lambda path: os.path.isdir(path)
        self.functions["__os_path_basename__"] = lambda path: os.path.basename(path)
        self.functions["__os_path_dirname__"] = lambda path: os.path.dirname(path)
        self.functions["__os_path_splitext__"] = lambda path: list(os.path.splitext(path))
        self.functions["__os_path_size__"] = lambda path: os.path.getsize(path)

    def _register_time(self) -> None:
        import time
        self.functions["__time_now__"] = lambda: time.time()
        self.functions["__time_format__"] = lambda timestamp, fmt: time.strftime(fmt, time.localtime(timestamp))
        self.functions["__time_parse__"] = lambda time_str, fmt: time.mktime(time.strptime(time_str, fmt))
        self.functions["__time_sleep__"] = lambda seconds: time.sleep(seconds)
        self.functions["__time_localtime__"] = _time_localtime

    def _register_random(self) -> None:
        import random
        self.functions["__random_int__"] = lambda a, b: random.randint(int(a), int(b))
        self.functions["__random_float__"] = lambda: random.random()
        self.functions["__random_choice__"] = lambda arr: random.choice(arr)
        self.functions["__random_shuffle__"] = lambda arr: random.sample(arr, len(arr))
        self.functions["__random_sample__"] = lambda arr, k: random.sample(list(arr), int(k))
        self.functions["__random_seed__"] = lambda seed: random.seed(seed)

    def _register_json(self) -> None:
        import json
        self.functions["__json_parse__"] = lambda json_str: json.loads(json_str)
        self.functions["__json_stringify__"] = lambda obj: json.dumps(obj, ensure_ascii=False)

    def _register_concurrency(self) -> None:
        import time as _time
        self.functions["__concurrent_sleep__"] = lambda seconds: _time.sleep(float(seconds))
        self.functions["__concurrent_thread_count__"] = lambda: __import__("threading").active_count()
        self.functions["__concurrent_cpu_count__"] = lambda: __import__("os").cpu_count() or 1

    def register(self, name: str, func: Callable) -> None:
        self.functions[name] = func

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self.functions:
            raise NameError(f"未定义的内置函数: {name}")
        return self.functions[name](*args, **kwargs)

    def has(self, name: str) -> bool:
        return name in self.functions


def _math_log(x, base=None):
    import math
    if base is None:
        return math.log(float(x))
    return math.log(float(x), float(base))


def _array_find(arr, val):
    try:
        return arr.index(val)
    except ValueError:
        return -1


def _array_flat(arr):
    result = []
    for item in arr:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def _array_unique(arr):
    seen = []
    result = []
    for item in arr:
        if item not in seen:
            seen.append(item)
            result.append(item)
    return result


def _dict_from_list(keys, values=None):
    if values is None:
        return dict.fromkeys(keys)
    return dict(zip(keys, values))


def _write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _append_file(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)


def _writelines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _time_localtime(timestamp=None):
    import time
    if timestamp is None:
        t = time.localtime()
    else:
        t = time.localtime(timestamp)
    return {"年": t.tm_year, "月": t.tm_mon, "日": t.tm_mday,
            "时": t.tm_hour, "分": t.tm_min, "秒": t.tm_sec,
            "周几": t.tm_wday, "年中第几天": t.tm_yday}


BUILTINS = BuiltinRegistry()
