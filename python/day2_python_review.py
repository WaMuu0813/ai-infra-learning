from model_utils import tokens_per_second

tokens_per_second(10000, 2.5)

# memory = model_utils.estimate_memory(32, 0.5)
# print(memory)

# nums = [1,2,3,4,5]

# for x in nums:
#     print(x * x) 

# model = {
#     "name": "llama",
#     "layers": 32,
#     "hidden_size": 4096
# }

# for key,value in model.items():
#     print(key,value)


# def get_even(nums):
#     result = []
#     for x in nums:
#         if x % 2 == 0:
#             result.append(x)
#     return result

# evens = get_even(nums)
# print(evens)

# class GPU:
#     def __init__(self,name,memory):
#         self.name = name
#         self.memory = memory

#     def show(self):
#         print(self.name,self.memory)

#     def can_run(self,required_memory):
#         if self.memory >= required_memory:
#             return True
#         else:
#             return False

# gpu = GPU("A100",80)
# gpu.show()

# print(gpu.can_run(40))
# print(gpu.can_run(100))

# nums1 = [1,2,3,4,5]
# squares = [x*x for x in nums1]
# print(squares)

# nums2 = [3,15,7,22,9,18]
# ret = [y for y in nums2 if y > 10]
# print(ret)

# request_lengths = [128, 512, 64, 1024, 256]
# ret1 = [z for z in request_lengths if z >= 256]
# print(ret1)

# def launch_model(model_name, *args, **kwargs):
#     print("model_name:", model_name)
#     print("args:", args)
#     print("kwargs:", kwargs)

# launch_model(
#     "aplusx",
#     64,128,256,
#     x = "south",
#     y = "north",
#     z = "east")

# config = {
#     "model_name": "llama",
#     "dtype": "float16",
#     "max_tokens": 4096
# }

# def start(model_name, dtype, max_tokens):
#     print(model_name)
#     print(dtype)
#     print(max_tokens)

# start(**config)