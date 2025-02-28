def count_rectangles(n, m, grid):
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if grid[i - 1][j - 1] == '.':
                dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
            else:
                dp[i][j] = 0
    
    return sum(sum(row) for row in dp)

# 测试样例
n = 6
m = 4
grid = [
    ['.', '.', '*', '*'],
    ['.', '.', '*', '*'],
    ['*', '*', '.', '.'],
    ['*', '*', '.', '.'],
    ['.', '.', '*', '*'],
    ['.', '.', '*', '*']
]


if __name__ == '__main__':
    c = 'python'
    b = '3 c'
    a = 2*b
    print(a)
    print(count_rectangles(n, m, grid)) # 输出38
  