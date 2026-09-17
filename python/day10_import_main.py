# print("main: before import")

# import day10_import_helper

# print("main: after import")

# print("value =", day10_import_helper.value)
# print("add =", day10_import_helper.add(3, 4))

# print("main: before first import")

# import day10_import_helper

# print("main: after first import")

# print("main: before second import")

# import day10_import_helper

# print("main: after second import")


import sys
import os

print("current working directory:")
print(os.getcwd()) # 终端当前所在目录


print("\nscript path:")
print(__file__) # 当前脚本文件本身的位置

print("\nsys.path[0]:")
print(sys.path[0]) # 当前这种启动方式下，import 搜索的首要目录


# for path in sys.path:
#     print(path)



# print("before:")
# print("day10_import_helper" in sys.modules)

# import day10_import_helper

# print("\nafter first import:")
# print("day10_import_helper" in sys.modules)

# import day10_import_helper

# print("\nafter second import:")
# print("day10_import_helper" in sys.modules)

# print(
#     "same object:",
#     day10_import_helper is sys.modules["day10_import_helper"]
# )