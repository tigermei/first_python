import random

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

def main():
    # 生成随机数据
    data = [random.randint(1, 100) for _ in range(10)]
    
    # 打印排序前的数据
    print("排序前的数据:", data)
    
    # 进行快速排序
    quick_sort(data, 0, len(data) - 1)
    
    # 打印排序后的数据
    print("排序后的数据:", data)

if __name__ == "__main__":
    main()
