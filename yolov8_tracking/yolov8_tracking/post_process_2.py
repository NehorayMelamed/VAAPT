from tkinter import simpledialog, Tk
from RDND_proper.RapidBase.import_all import *
from collections import defaultdict

# Define global variables
cls_dict = {0: 'car', 1: 'jeep', 2: 'taxi', 3: 'bus', 4: 'truck', 5: 'minibus', 6: 'motorcycle', 7: 'bicycle',
            8: 'minivan', 9: 'person', 10: 'tender'}
drawing = False
mode = 0
ix, iy = -1, -1
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
lanes = {}
frame_size = (960, 540)
lanes_cluster = {}
class Lane:
    def __init__(self):
        self.id_list_closest_to_lane = []
        self.counter_dict = defaultdict(int)

# Define drawing function
def draw_lane(event, x, y, flags, param):
    global ix, iy, drawing, mode, lanes, colors, frame_original_size

    # Calculate the scale factor based on the original and resized dimensions
    scale_x = frame_original_size[1] / frame_size[0]
    scale_y = frame_original_size[0] / frame_size[1]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        lanes['lane'+str(mode)] = [(ix, iy)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (ix, iy), (x, y), colors[mode % len(colors)], 2)
            ix, iy = x, y
            lanes['lane'+str(mode)].append((int(x * scale_x), int(y * scale_y)))

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False
        rename_lane()
        root = Tk()
        root.withdraw()
        lanes_cluster[lane_name] = simpledialog.askstring("Input", "What is the cluster of the lane?")
        root.destroy()


# Function to rename lane
def rename_lane():
    global mode
    global lane_name
    root = Tk()
    root.withdraw()
    lane_name = simpledialog.askstring("Input", "What is the name of the lane?")
    lanes[lane_name] = lanes.pop('lane'+str(mode))
    root.destroy()
    mode += 1


def find_indexes(big_list, numbers):
    return [i for i, num in enumerate(big_list) if num in numbers[1]]


def get_cls(bounding_box_list, cls_list):
    heights_list = bounding_box_list[:, 1]
    widths_list = bounding_box_list[:, 0]
    if len(heights_list) != len(widths_list):
        raise ValueError("The number of heights and widths should be the same.")

    # Combine heights and widths into a list of tuples (height, width)
    boxes = list(zip(heights_list, widths_list))

    # Sort the list of boxes based on their area (height * width)
    sorted_boxes = sorted(boxes, key=lambda box: box[0] * box[1], reverse=True)

    # Get the five biggest bounding boxes
    indexes = find_indexes(widths_list, sorted_boxes[:5])
    bb_cls_list = []

    for i in indexes:
        bb_cls_list.append(cls_list[i])

    return np.bincount(bb_cls_list).argmax()
# Function to convert lanes to fixed number of points
def convert_lanes(lanes, number_of_samples):
    new_lanes = {}
    for lane_name, lane in lanes.items():
        if len(lane) > number_of_samples:
            new_lane = [lane[i*len(lane)//number_of_samples] for i in range(number_of_samples)]
        else:
            new_lane = lane + [lane[-1]]*(number_of_samples-len(lane))
        new_lanes[lane_name] = new_lane
    return new_lanes


def dictionary_to_arrays(dictionary):
    arrays = []
    for key, value in dictionary.items():
        arrays.append(np.array(value))
    return arrays

def draw_lanes_on_image(image, lane_dict):
    lanes_array = dictionary_to_arrays(lane_dict)

from scipy.spatial import distance_matrix
def get_stats(base_dir):
    # Load video
    cap = cv2.VideoCapture('video_path.mp4')

    # Take first frame of the video
    ret, frame = cap.read()

    frame = cv2.imread(f"{base_dir}/first_frame.jpg")
    global frame_original_size
    frame_original_size = frame.shape
    frame = cv2.resize(frame, (960, 540))
    # Create a black image and a window
    global img
    img = frame

    # np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    cv2.namedWindow('image')

    # Bind the function to window
    cv2.setMouseCallback('image', draw_lane)

    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    # Print all lanes with their names
    for lane_name, lane in lanes.items():
        print(lane_name, lane)



    # Convert and print the new lanes
    GT_lanes = convert_lanes(lanes, number_of_samples=100)

    trajectories_list = []
    txt_dir = f"{base_dir}/txt_files"
    # lane_dir = f"{base_dir}/lanes"
    txt_files_list = os.listdir(txt_dir)
    # lane_files_list = os.listdir(lane_dir)
    txt_files_list = sorted(txt_files_list, key=lambda x: int(os.path.split(x)[1].split("_")[0]))
    cls_list = []
    id_list = []
    for file in txt_files_list:
        data = np.loadtxt(os.path.join(txt_dir, file))
        if len(data.shape) == 1 or len(data) < 10:
            continue
        else:
            classes = np.array(data[:, 3], dtype=np.uint8)
            most_common_number = get_cls(data[:, 4:], classes)
            # time_stamps = data[:, 0]
            if not most_common_number == 9:
                data = data[:, 1:3]
                if any(data):
                    # data = get_rid_of_redlight_frames(data)
                    cls_list.append(most_common_number)
                    id_list.append(int(file.split("_")[0]))
                    trajectories_list.append(data)
    # list_of_lane_counters = [Lane() for _ in range(len(lanes_cluster))]
    list_of_lane_counters = [Lane() for _ in range(len(set(lanes_cluster.values())))]
    tic()

    ### Find closest GT lane to each trajectory: ###
    ### Loop over trajectories: ###
    for trajectory_idx, current_trajectory in enumerate(trajectories_list):
        closest_lane = float('inf')
        # closest_lane_idx = -1
        ### Loop over GT lanes: ###
        for lane_idx, lane_key in enumerate(GT_lanes):
            current_lane = np.array(GT_lanes[lane_key])
            counter = 1
            l_sum = 0
            ### Compute distance matrix from current lane to current trajectory arrays: ###
            current_distance_matrix = distance_matrix(current_lane, current_trajectory)  #[N_GT_lane_points, N_trajectory_points]

            ### Find closest point to each point in current lane to current trajectory: ###
            min_distance_values = np.min(current_distance_matrix, 0)
            
            ### average min distance values to get average distance between current trajectory and current GT lane: ###
            average_distance = np.mean(min_distance_values)

            ### Check if current lane is closer than closest previous lane: ###
            if average_distance < closest_lane:
                closest_lane = average_distance
                # closest_lane_idx = lane_idx
                closest_lane_idx = int(lanes_cluster[lane_key]) - 1

                # next(i for i, sublist in enumerate(lane_clusters) if lane_idx in sublist)

        list_of_lane_counters[closest_lane_idx].id_list_closest_to_lane.append(id_list[trajectory_idx])
        list_of_lane_counters[closest_lane_idx].counter_dict[cls_dict[cls_list[trajectory_idx]]] += 1

    toc("stats")

    for lane_iter, _ in enumerate(list_of_lane_counters):
        print(list_of_lane_counters[lane_iter].counter_dict)
        with open(f"{base_dir}/lanes/{lane_iter+1}_statistics.txt", "w") as f:
            # Iterate over the items in the dictionary and write them to the file
            for key, value in list_of_lane_counters[lane_iter].counter_dict.items():
                f.write(f"{key}\t{value}\n")

    # for lane_name, lane in GT_lanes.items():
    #     print(lane_name, lane)

if __name__ == "__main__":
    root_dir = r"/home/mafat/Desktop/eylon/outputs/output_herzel_930_945"
    get_stats(root_dir)
