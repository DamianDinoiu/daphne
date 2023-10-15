# Open a file for writing
with open('numbers.txt', 'w') as file:
    # Write numbers from 1 to 10000, one on each line
    g1 = 0
    g2 = 0
    g3 = 0
    g4 = 0
    g5 = 0
    for number in range(1, 100, 8):
        if (number < 20):
            g1 += 1
        if (number > 20 and number < 40):
            g2 += 1
        if (number > 40 and number < 60):
            g3 += 1
        if (number > 60 and number < 80):
            g4 += 1
        if (number > 80 and number < 100):
            g5 += 1

        
        file.write(str(number) + '\n')
    print(g1)
    print(g2)
    print(g3)
    print(g4)
    print(g5)
