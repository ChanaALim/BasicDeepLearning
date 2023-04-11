
def AND(x1, x2):
    # NAND : -0.5, -0.5, -0.7
    # OR : 0.5, 0.5, 0.4
    w1, w2, theta = 0.5, 0.5, 0.7
    y = (x1 * w1) + (x2 * w2)
    if y <= theta:
        return 0
    elif y > theta:
        return 1

while True:
    n1, n2 = map(int, input().split())
    # print(type(n1))
    if str(n1) == "q":
        break
    print(AND(n1, n2))
