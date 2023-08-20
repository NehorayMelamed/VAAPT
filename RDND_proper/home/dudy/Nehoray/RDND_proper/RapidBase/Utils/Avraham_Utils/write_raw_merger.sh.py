import os


def format_num(i: int) -> str:
    s = str(i)
    return "0"*(4-len(s))+s


if __name__ == '__main__':
    head_directory = "/media/avraham/dudy/fast_frames"
    os.chdir(head_directory)
    lines_sh = ["touch combined_fast_video.raw"]
    for i in range(10000):
        lines_sh.append(f"cat {head_directory}/fasf{format_num(i)}.raw >> combined_fast_video.raw")
    with open("merger.sh", 'w+') as merger:
        merger.write("\n".join(lines_sh))
        merger.write("\n")
