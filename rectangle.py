import argparse

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

# Example usage:
def main_program():
    n, m = map(int, input().split())
    paper = [input() for _ in range(n)]
    print(count_rectangles(n, m, paper))


if __name__ == '__main__':
    c = 'python'
    b = '3 c'
    a = 2*b
    print(a)
    main_program()
    
  