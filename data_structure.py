
arr = [10, 20, 30, 40, 50]

def copy_array(arr, n):
    new_arr = [0] * n
    for i in range(n):  # Copying element by element
        new_arr[i] = arr[i]
    return new_arr

# Example Usage
copied_array = copy_array(arr, 5)
print(copied_array)  


def search_by_value(arr, value):
    for i in range(len(arr)):  # Loop through the array
        if arr[i] == value:  # Check if value matches
            return i  # Return the index
    return -1  # Return -1 if value is not found

print(search_by_value(arr, 30))  # Output: 2
print(search_by_value(arr, 60))  # Output: -1 (Not found)


#OPERATION 1: Remove element at beginning
arr_1 = [2, 3, 4, 5]  
size = len(arr_1) 

if size > 0:
    for i in range(1, size):
        arr_1[i - 1] = arr_1[i]  # Move elements left

size -=1
for i in range(size):
    print(arr_1[i])


#OPERATION 2: Add element at beginning
arr_2 = [2, 3, 4, 5, None]
new_element = 1
size = 4
if size > 0: 
    for i in range(size-1, -1, -1):
        arr_2[i + 1] = arr_2[i]

arr_2[0] = new_element
size += 1

print(arr_2[:size]) 