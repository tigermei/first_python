def count_possible_rectangles(n, m, grid):
    count = 0
    
    #以当前方块为右下角的方块数量
    rect_num = [[0 for j in range(m)] for i in range(n)]
    
    # 当前方块上方最高值。
    high_dp = [[0 for j in range(m)] for i in range(n)]
    
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '*':  
                high_dp[i][j] = 0
            else:
                if i == 0:
                    high_dp[i][j] = 1
                else :
                    high_dp[i][j] = high_dp[i-1][j] + 1 
                    
    # print(high_dp)
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '*': 
                rect_num[i][j] = 0
            else:
                if high_dp[i][j-1] <= high_dp[i][j]:
                    rect_num[i][j] = rect_num[i][j-1] + high_dp[i][j] 
                else:
                    # 找到左侧第一个不高于high_dp[i][j]
                    width = 1
                    while 0 <= j - width and high_dp[i][j] < high_dp[i][j - width]:
                        width += 1
                    if j - width < 0:
                        rect_num[i][j] = high_dp[i][j] * width
                    else:
                        rect_num[i][j] = rect_num[i][j - width] + (high_dp[i][j] * width)
                                 
    # print(rect_num)                    
    for i in range(n):
        for j in range(m):
            count += rect_num[i][j]                
    return count

def main_program():
    n, m = map(int, input().split())
    paper = [input() for _ in range(n)]
    print(count_possible_rectangles(n, m, paper))

if __name__ == '__main__':
    main_program()