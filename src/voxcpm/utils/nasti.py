def dec2bit(num):
    return bin(num).replace("0b", "").zfill(7)

def bit2dec(bit):
    return int(bit, 2)

def ascii2num(char):
    return ord(char)

def num2ascii(num):
    if num == 10:
        return "\\n"
    return chr(num)

def main():
    for c in "nasti":
        print(f"Character: {c}, ASCII: {ascii2num(c):3d}, Binary: {dec2bit(ascii2num(c))[:4]} {dec2bit(ascii2num(c))[4:]}")
    print(num2ascii(bit2dec("0001010")))
    
if __name__ == "__main__":
    main()