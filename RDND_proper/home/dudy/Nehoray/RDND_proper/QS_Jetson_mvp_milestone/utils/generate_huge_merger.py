merger_commands = []

for i in range(18681):
    next_filename = f"fast_mode{i}.raw"
    merger_commands.append(f"cat {next_filename} >> combined_video.raw")

with open('merger.sh', 'w') as merger:
    merger.write("\n".join(merger_commands))
    merger.write("\n")