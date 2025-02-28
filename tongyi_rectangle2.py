def count_possible_rectangles(n, m, grid):
    count = 0
    
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '*':  # 假设'*'表示不能形成矩形的点，'.'是可以形成矩形的部分
                continue
            # 简化的逻辑可能只是检查每个点上下左右能否延伸形成矩形，但实际问题细节不明确
            # 这里需要具体的规则来判断如何构成一个“合法”的矩形，以下仅为示意
            # 例如，检查以当前点为右下角，左边和上边是否都是 '*' 形成的直边
            width = 0
            while j - width >= 0 and grid[i][j - width] == '.':
                width += 1
                
            height = 0
            while i - height >= 0 and all(grid[k][j] == '.' for k in range(i-height, i)):
                height += 1
                
            # 对于每个点，其能参与构成的矩形数量取决于它可以向左和向上延伸的长度
            # 实际上，这一步可能需要更精细的逻辑来准确计算能形成的矩形数量
            count += width * height  # 简单累加可能的矩形数量
            
    return count

def count_rectangles(n, m, paper):
    dp = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if paper[i][j] == '*':
                dp[i][j] = 0
            else:
                left = dp[i][j-1] if j > 0 else 0
                top = dp[i-1][j] if i > 0 else 0
                topleft = dp[i-1][j-1] if i > 0 and j > 0 else 0
                dp[i][j] = (left + top - topleft + 1) % 1000000007

    total_rectangles = sum(sum(row) for row in dp)
    return total_rectangles % 1000000007



# 使用之前假设的测试样例（但需注意，这个逻辑可能不符合原问题的实际意图）
grid_example = [
    ['.', '.', '.', '.'],
    ['.', '*', '*', '*'],
    ['.', '*', '.', '.'],
    ['.', '*', '*', '*'],
    ['.', '.', '.', '*'],
    ['.', '*', '*', '*']
]

grid_example1 = [
    ['.', '.', '*'],
    ['.', '.', '.'],
    ['*', '.', '.']
]


def count_dp_example(grid):
    # Example usage:
    n, m = len(grid), len(grid[0])
    print(count_rectangles(n, m, grid))
    
    
def count_simple_example(grid):
    # Example usage:
    n, m = len(grid), len(grid[0])
    print(count_possible_rectangles(n, m, grid))
    
# Example usage:
def main_program():
    n, m = map(int, input().split())
    paper = [input() for _ in range(n)]
    print(count_possible_rectangles(n, m, grid_example1))
    count_simple_example(grid_example1)
    count_dp_example(grid_example1)


if __name__ == '__main__':
    c = 'python'
    b = '3 c'
    a = 2*b
    print(a)
    main_program()