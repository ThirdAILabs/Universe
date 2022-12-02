h = "9adbfd881d363d31c156c348996892dd0183a7e4540dc001cb7c2eec5a7966ae"
for i in range(0, len(h), 2):
    print(f'0x{h[i: i+2].upper()}', end=", ")
    # print("'" + str(int(h[i: i+2], base=16)),  end = "', ")