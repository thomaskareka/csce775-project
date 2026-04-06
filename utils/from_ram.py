def get_x_pos_adjusted(ram) -> int:
    # The x position is stored in two bytes at 0x006D and 0x0086, and the current level is stored in 0x075F and 0x0760
    return (int(ram[0x006D]) << 8 | int(ram[0x0086])) + (int(ram[0x075F]) * 4 + int(ram[0x0760])) * 10000
