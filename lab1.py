import random
import numpy as np
import statistics

def count_pairs(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if nums[i] + nums[j] == 10:
                count += 1
    return count

def list_range(nums):
    range_ = max(nums) - min(nums)
    return range_

def mult_matrices(a,b):
    return np.dot(a,b)

def mat_pow(a,m):
    result = mult_matrices(a,a)
    for i in range(m-2):
        result = mult_matrices(result,a)
    return result

def highchar(st):
    ch = ""
    count = 0
    for i in st:
        if count < st.count(i):
            count = st.count(i)
            ch = i
    return ch

def charcount(st):
    ch = ""
    count = 0
    for i in st:
        if count < st.count(i):
            count = st.count(i)
            ch = i
    return count

def mean(x):
    li = np.array(x)
    return np.mean(li)

def median(x):
    li = np.array(x)
    return np.median(li)

def mode(x):
    li = np.array(x)
    return np.mode(li)

nums1 = [2,7,4,1,3,6]
print("1. Number of pairs with sum equal to 10:",count_pairs(nums1))

nums2 = list(map(float,input("Enter real numbers:").split()))
if len(nums2) < 3:
    print("2. Range determination not possible")
else:
    r = list_range(nums2)
    print("2. Range of the list:",r)

n = int(input("Enter order of square matrix:"))
matrix = []
print("Enter matrix elements row-wise:")
for _ in range(n):
    matrix.append(list(map(int,input().split())))

matrix = np.array(matrix)
m = int(input("Enter power m:"))

matrix_result = mat_pow(matrix,m)
print("3. Matrix A^m:")
print(matrix_result)

string_input = input("Enter a string:")

print("4. Highest occurring character:", highchar(string_input))
print("Occurrence count:", charcount(string_input))

random_numbers = [random.randint(1, 10) for _ in range(1,25)]

mean_value = mean(random_numbers)
median_value = median(random_numbers)
mode_value = statistics.mode(random_numbers)

print("\n5. Random numbers:", random_numbers)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
