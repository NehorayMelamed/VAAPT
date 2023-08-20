

merger_commands = []


def get_ith_filename(i: int):
    i_str = str(i)
    i_len = len(i_str)
    if i_len == 1:
        return f"slow_mode00{i}.raw"
    elif i_len == 2:
        return f"slow_mode0{i}.raw"
    else:
        return f"slow_mode{i}.raw"


for i in range(400):
    next_filename = get_ith_filename(i)
    merger_commands.append(f"cat {next_filename} >> combined_video.raw")

with open('merger.sh', 'w') as merger:
    merger.write("\n".join(merger_commands))
    merger.write("\n")