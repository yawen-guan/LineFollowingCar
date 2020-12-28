

def checkListValue(lst, value, goal):
    cnt = 0
    for v in lst:
        if v == value:
            cnt += 1
    return cnt >= goal


def getListValue(lst, goal):
    cnt = {}
    for v in lst:
        if v not in cnt.keys():
            cnt[v] = 1
        else:
            cnt[v] += 1
        if cnt[v] >= goal:
            return v
    return None


def allTrue(lst):
    return checkListValue(lst, True, len(lst))


def allFalse(lst):
    return checkListValue(lst, False, len(lst))


def mostTrue(lst):
    return checkListValue(lst, True, (len(lst) // 2) + 1)


def getMost(lst):
    return getListValue(lst, (len(lst) // 2) + 1)


def getTurnDirection(pre_labels, pre_size):
    print("isTurn = ", pre_labels["isTurn"])
    print("turnDirection = ", pre_labels["turnDirection"])
    cnt = {}
    maxcnt = 0
    turnDirection = None
    for i in range(len(pre_labels["isTurn"])):
        if pre_labels["isTurn"][i] == True:
            v = pre_labels["turnDirection"][i]
            if v not in cnt.keys():
                cnt[v] = 1
            else:
                cnt[v] += 1
            if cnt[v] >= maxcnt:
                maxcnt = cnt[v]
                turnDirection = v
    return turnDirection


def getLabel(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1
