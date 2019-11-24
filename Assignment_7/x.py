def f(x, jin):
    l = []
    for i in x:
        l.append(i)

    for i in x:
        l.append(i+2*jin)
        l.append(i+4*jin)
    return list(set(l))
