import time

def constant_time(arr):  # O(1)
    print("First element:", arr[0])  # Accessing an element = constant time
    print("Iterations: 1")

def logarithmic_time(n):  # O(log n)
    i = 1
    iterations = 0
    while i < n:
        print(i)
        i *= 2  
        iterations += 1
    print("Iterations:", iterations)

def linear_time(arr):  # O(n)
    iterations = 0
    for num in arr:
        print(num)  # Iterates over all elements
        iterations += 1
    print("Iterations:", iterations)

def quadratic_time(n):  # O(n^2)
    iterations = 0
    for i in range(n):
        for j in range(n):
            print("*")  # Nested loops = quadratic 
            iterations += 1
        print()
    print("Iterations:", iterations)


arr = [1, 2, 3, 4, 5]

print("O(1) - Constant Time:")
constant_time(arr)

print("\nO(log n) - Logarithmic Time:")
logarithmic_time(32)

print("\nO(n) - Linear Time:")
linear_time(arr)

print("\nO(n^2) - Quadratic Time:")
quadratic_time(6)
