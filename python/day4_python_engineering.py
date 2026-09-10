# print("start")

# try:
#     x = int(input("number:"))

#     print(x)

# except ValueError:
#     print("input error")

# print("end")

# def divide(a,b):

#     try:
#         return a / b

#     except ZeroDivisionError:
#         print("cannot divide by zero")
#         return None

# print(divide(10,2))
# print(divide(10,0))

# print("\nfile test")

# f = open("test.txt", "w")

# f.write("hello ai infra\n")

# f.close()

print("\nwith test")


with open("test_with.txt", "w") as f:
    f.write("hello with\n")


print("file closed automatically")