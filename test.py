with open("wayfair_out.txt")as f:
    a = f.readlines()
    den = len(a)
    num = len([b for b in a if len(b.strip()) > 0])
    print(num / den)