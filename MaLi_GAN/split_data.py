

with open('save/real_data.txt', 'r') as f:
    lines = f.read().splitlines()

with open('save/real_data.txt', 'w') as f:
    f.writelines([line+'\n' for line in lines[:6657]])

with open('save/eval_data.txt', 'w') as f:
    f.writelines([line+'\n' for line in lines[6657:]])


