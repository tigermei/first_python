def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    up = [0] * (m + 1)
    lef = [0] * (m + 1)
    s = [{'h': 0, 'pos': 0}] * (n * m + 5)
    top = 0
    ans = 0
    idx = 2

    for i in range(1, n + 1):
        top = 0
        s[top] = {'h': 0, 'pos': 0}
        for j in range(1, m + 1):
            ch = data[idx]
            idx += 1
            if ch == '.':
                up[j] += 1
                while s[top]['h'] >= up[j]:
                    top -= 1
                lef[j] = s[top]['pos']
                top += 1
                s[top] = {'h': up[j], 'pos': j}
                for k in range(top):
                    ans += (s[k + 1]['h'] - s[k]['h']) * (j - s[k]['pos'])
            else:
                up[j] = 0
                top = 1
                s[top] = {'h': 0, 'pos': j}
    
    print(ans)

if __name__ == "__main__":
    main()