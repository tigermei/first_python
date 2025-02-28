import random
import copy

def quick_sort(arr, left, right):
    if left < right:
        pivot_pos = partition(arr, left, right)
        quick_sort(arr, left, pivot_pos - 1)
        quick_sort(arr, pivot_pos + 1, right)

def partition(arr, left, right):
    pivot = arr[right]
    i = left - 1
    
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def main():
    # 生成随机数据
    data = [random.randint(1, 100) for _ in range(10)]
    
    # 为每种排序算法创建数据副本
    data_quick = data.copy()
    data_bubble = data.copy()
    data_selection = data.copy()
    data_insertion = data.copy()
    data_merge = data.copy()
    
    print("原始数据:", data)
    print("\n各种排序算法结果:")
    
    # 快速排序
    quick_sort(data_quick, 0, len(data_quick) - 1)
    print("快速排序:", data_quick)
    
    # 冒泡排序
    bubble_sort(data_bubble)
    print("冒泡排序:", data_bubble)
    
    # 选择排序
    selection_sort(data_selection)
    print("选择排序:", data_selection)
    
    # 插入排序
    insertion_sort(data_insertion)
    print("插入排序:", data_insertion)
    
    # 归并排序
    data_merge = merge_sort(data_merge)
    print("归并排序:", data_merge)

if __name__ == "__main__":
    main()
