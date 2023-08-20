import scipy.io, argparse, sys, os


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-m", "--mat_file", help="RAW video file", required=True, type=str)
    parser.add_argument("-n", "--name", help="Name of .txt file", required=True, type=str)
    return parser


def exit_on(message: str):
    sys.stderr.write(message)
    sys.exit(1)


def check_input(args: argparse.Namespace):
    if not os.path.exists(args.mat_file):
        exit_on("Given MAT file does not exist\n")
    if os.path.exists(args.name + ".txt"):
        exit_on("Given name for txt file already exists\n")


def parse_1d_vector(vector):
    substrs = ['[']
    for i in range(vector.shape[0]-1):
        substrs.append(str(vector[i]) + ",")
    substrs.append(str(vector[vector.shape[0]-1]))
    substrs.append(']')
    return ''.join(substrs)


def parse_multidimensional_vector(vector):
    if len(vector.shape) == 1:
        return parse_1d_vector(vector)
    else:
        substrs = ['[']
        for i in range(vector.shape[0]):
            substrs.append(parse_multidimensional_vector(vector[i,]))
        substrs.append(']')
        return ''.join(substrs)


def convert_mat_to_str(mat_name: str):
    loaded_mat = scipy.io.loadmat(mat_name)
    loaded_mat_key = list(loaded_mat.keys())[-1]
    vector = loaded_mat[loaded_mat_key]
    dims = len(vector.shape)
    if dims > 3:
        exit_on("Too many dimensions to parse\n")
    elif dims == 1:
        return parse_1d_vector(vector)
    elif dims == 2:
        return parse_multidimensional_vector(vector)
    else:
        return parse_multidimensional_vector(vector)


def main():
    parser = create_parser()
    args = parser.parse_args()
    check_input(args)
    mat_str = convert_mat_to_str(args.mat_file)
    with open(args.name+".txt", 'w+') as txt_file:
        txt_file.write(mat_str)


if __name__ == '__main__' : # to avoid namespace stupidity  wrapped in main function
    main()
