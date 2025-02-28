import argparse
import cv2 as cv

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，
                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    help = "The path of address"
    parser.add_argument('--addresses',help = help)    # 步骤三，后面的help是我的描述
    parser.add_argument('--model', help="model here!! ")
    parser.add_argument("--checkpoint_path", help="check point path please!!")
    args = parser.parse_args()                                       # 步骤四          
    return args


def test_oneline_if():
    dx = 0
    sx = dx/abs(dx) if dx!=0 else 0
    print(sx)
  

# 快速排序
def sort(num:int, nLength: int):
     return
  
def test__():
    for x in range(1000):
        y = (x-50)//40 + 1
        print(y)


def test_number():
    percentage = 3
    print("{:.3f}".format(percentage))

def test_opencv():
    img = cv.imread('mess.png', 0)
    print(img)

def test_str():
    w = "15englisheeee23"
    for x in w:
        if '0' <= x <= '9':
            continue
        else:
            str1 = w.replace(x, '')
            print(str1)
    print(w)

if __name__ == '__main__':
    args = parse_args()
    print('Hello, World!!')
    print(args.addresses)
    
    test_number()
    test_opencv()
    test_str()
    
    
