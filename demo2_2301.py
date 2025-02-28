import random

def quicksort(arr):
    """
    快速排序算法的简单实现
    参数：
        arr: 需要排序的列表
    返回：
        排序后的新列表
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准值
    left = [x for x in arr if x < pivot]  # 小于基准值的元素
    middle = [x for x in arr if x == pivot]  # 等于基准值的元素
    right = [x for x in arr if x > pivot]  # 大于基准值的元素
    
    return quicksort(left) + middle + quicksort(right)


# 帮我用冒泡排序再实现一遍上面的算法
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 调用上面的冒泡算法实现一个demo
def bubble_sort_demo():
    # 生成20个1到100之间的随机整数
    random_numbers = [random.randint(1, 100) for _ in range(20)]
    
    # 打印排序前的数据
    print("排序前的随机数据:")
    print(random_numbers)
    
    # 使用冒泡排序算法进行排序
    sorted_numbers = bubble_sort(random_numbers.copy())  # 使用copy()避免修改原始数据
    
    # 打印排序后的数据
    print("\n冒泡排序后的数据:")
    print(sorted_numbers)

def main():
    # 生成20个1到100之间的随机整数
    random_numbers = [random.randint(1, 100) for _ in range(20)]
    
    # 打印排序前的数据
    print("排序前的随机数据:")
    print(random_numbers)
    
    # 使用快速排序算法进行排序
    sorted_numbers = quicksort(random_numbers)
    
    # 打印排序后的数据
    print("\n快速排序后的数据:")
    print(sorted_numbers)
    
    # 打印统计信息
    print(f"\n数据统计:")
    print(f"最小值: {sorted_numbers[0]}")
    print(f"最大值: {sorted_numbers[-1]}")
    print(f"元素个数: {len(sorted_numbers)}")
    
    # 调用冒泡排序算法
    bubble_sorted_numbers = bubble_sort(random_numbers)
    print("\n冒泡排序后的数据:")
    print(bubble_sorted_numbers)

if __name__ == "__main__":
    # main()  # 注释掉原来的main调用
    bubble_sort_demo()  # 使用新的demo函数
