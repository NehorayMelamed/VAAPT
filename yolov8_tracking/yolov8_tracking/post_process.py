import scipy.stats

from RDND_proper.RapidBase.import_all import *
import numpy as np
from collections import defaultdict
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

import matplotlib.pyplot as plt
cls_dict = {0: 'car', 1: 'jeep', 2: 'taxi', 3: 'bus', 4: 'truck', 5: 'minibus', 6: 'motorcycle', 7: 'bicycle',
            8: 'minivan', 9: 'person', 10: 'tender'}


def find_duplicates(lst):
    seen = {}
    duplicates = []
    indexes = []

    for i, item in enumerate(lst):
        if item in seen:
            if seen[item] not in duplicates:
                duplicates.append(seen[item])
                indexes.append(seen[item])
            duplicates.append(item)
            indexes.append(i)
        else:
            seen[item] = i

    return len(duplicates) > 0, duplicates, indexes

def get_largest_file(directory):
    largest_file_path = None
    largest_file_size = -1

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > largest_file_size:
                largest_file_size = file_size
                largest_file_path = file_path

    return largest_file_path


def find_common_elements(id_list_1, id_list_2):
    set2 = set(id_list_2)
    common_elements = [element for element in id_list_1 if element in set2]
    return common_elements


def find_unique_elements(id_list_1, id_list_2):
    set2 = set(id_list_2)
    unique_elements = [element for element in id_list_1 if element not in set2]
    return unique_elements


def get_rid_of_redlight_frames(data, threshold=5):
    # Calculate the differences between consecutive points
    diffs = np.diff(data, axis=0)
    # Calculate the distances using Euclidean distance formula
    distances = np.sqrt(np.sum(diffs[:, 1:] ** 2, axis=1))
    # Find the indices where the distance exceeds the threshold
    indices = np.where(distances >= threshold)[0]
    # Add the first point as it will always be kept
    indices += 1
    indices = np.insert(indices, 0, 0)
    # Return the filtered array
    filtered_data = data[indices]
    for i in range(len(filtered_data)):
        filtered_data[i, 0] = i

    return filtered_data


def interpolate_missing_timestamps(data_to_interpolate):

    # Sort the arrays based on frame indexes (x)
    frame_index  = data_to_interpolate[:, 0]
    x  = data_to_interpolate[:, 1]
    y  = data_to_interpolate[:, 2]

    sorted_indexes = np.argsort(frame_index)
    sorted_frame_indexes = frame_index[sorted_indexes]
    sorted_x = x[sorted_indexes]
    sorted_y = y[sorted_indexes]

    # Find the minimum and maximum values in the sorted array
    min_index = sorted_frame_indexes[0]
    max_index = sorted_frame_indexes[-1]

    # Create a new array with consecutive frame indexes
    filled_frame_indexes = numpy_unsqueeze(np.arange(min_index, max_index + 1), dim=1)

    # Interpolate the y values to fill the gaps
    filled_y = np.interp(filled_frame_indexes, sorted_frame_indexes, sorted_y)
    filled_x = np.interp(filled_frame_indexes, sorted_frame_indexes, sorted_x)

    return np.concatenate((filled_frame_indexes, filled_x, filled_y), axis=1)


def interpolate_to_size(data_to_interpolate, size=500):

    # Sort the arrays based on frame indexes (x)
    frame_index = data_to_interpolate[:, 0]
    x  = data_to_interpolate[:, 1]
    y  = data_to_interpolate[:, 2]

    sorted_indexes = np.argsort(frame_index)
    sorted_frame_indexes = frame_index[sorted_indexes]
    sorted_x = x[sorted_indexes]
    sorted_y = y[sorted_indexes]

    min_index = frame_index[0]

    # Create a new array with consecutive frame indexes
    filled_frame_indexes = numpy_unsqueeze(np.arange(min_index, min_index + size), dim=1)

    # Interpolate the y values to fill the gaps
    filled_y = np.interp(filled_frame_indexes, sorted_frame_indexes, sorted_y)
    filled_x = np.interp(filled_frame_indexes, sorted_frame_indexes, sorted_x)

    return np.concatenate((filled_frame_indexes, filled_x, filled_y), axis=1)


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


from RDND_proper.RapidBase.Utils.ML_and_Metrics.Clustering import *
def get_statistics(base_dir):
    trajectories_list = []
    txt_dir = f"{base_dir}/txt_files"
    lane_dir = f"{base_dir}/lanes"
    txt_files_list = os.listdir(txt_dir)
    lane_files_list = os.listdir(lane_dir)
    longest_trajectory = np.loadtxt(get_largest_file(txt_dir)).shape[0]
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
            time_stamps = data[:, 0]
            if not most_common_number == 9:
                cls_list.append(most_common_number)
                id_list.append(int(file.split("_")[0]))
                data = data[:, 1:3]
                if any(data):
                    # data = get_rid_of_redlight_frames(data)
                    trajectories_list.append(data)

    # ### Plot lengths histogram: ###
    # lengths_list = []
    # for i in np.arange(len(trajectories_list)):
    #     current_trajectory = trajectories_list[i]
    #     lengths_list.append(len(current_trajectory))
    # lengths_array = np.array(lengths_list)
    # lengths_histogram, bin_edges = np.histogram(lengths_array, density=False)
    # bin_centers = (bin_edges[1:] + bin_edges[0:-1])/2
    # ones_vec1 = np.ones_like(lengths_array) * np.quantile(lengths_histogram, 0.7)
    # ones_vec2 = np.ones_like(lengths_array) * np.quantile(lengths_histogram, 0.75)
    # plot(bin_centers, lengths_histogram)
    # plot(bin_centers)
    # ### Plot lengths_array: ###
    # plot(lengths_array)
    # plot(ones_vec1)
    # plot(ones_vec2)
    # plt.show()
    def extract_features_polynom(data):
        x = np.zeros((len(data), 6))

        x[:, 0] = np.array([np.polyfit(np.arange(data[i].shape[0]), data[i][:, 0], deg=2)[0] for i in np.arange(len(data))])
        x[:, 1] = np.array([np.polyfit(np.arange(data[i].shape[0]), data[i][:, 0], deg=2)[1] for i in np.arange(len(data))])
        x[:, 2] = np.array([np.polyfit(np.arange(data[i].shape[0]), data[i][:, 0], deg=2)[2] for i in np.arange(len(data))])
        x[:, 3] = np.array([np.polyfit(np.arange(data[i].shape[0]), data[i][:, 1], deg=2)[0] for i in np.arange(len(data))])
        x[:, 4] = np.array([np.polyfit(np.arange(data[i].shape[0]), data[i][:, 1], deg=2)[1] for i in np.arange(len(data))])
        x[:, 5] = np.array([np.polyfit(np.arange(data[i].shape[0]), data[i][:, 1], deg=2)[2] for i in np.arange(len(data))])

        return x


    def extract_features(data):
        x = np.zeros((len(data), 17))
        x[:, 0] = [np.mean(traj[:, 0]) for traj in data]
        x[:, 1] = [np.mean(traj[:, 1]) for traj in data]
        x[:, 2] = [np.std(traj[:, 0]) for traj in data]
        x[:, 3] = [np.std(traj[:, 1]) for traj in data]

        x[:, 4] = [np.mean(traj[1:, 0] - traj[0:-1, 0]) for traj in data]
        x[:, 5] = [np.std(traj[1:, 0] - traj[0:-1, 0]) for traj in data]
        x[:, 6] = [np.mean(traj[1:, 1] - traj[0:-1, 1]) for traj in data]
        x[:, 7] = [np.std(traj[1:, 1] - traj[0:-1, 1]) for traj in data]
        #
        # length as feature
        x[:, 8] = [len(traj) for traj in data]

        x[:, 9] = [scipy.stats.skew(traj[:, 0]) for traj in data]
        x[:, 10] = [scipy.stats.skew(traj[:, 1]) for traj in data]
        x[:, 11] = [scipy.stats.skew(traj[1:, 0] - traj[0:-1, 0]) for traj in data]
        x[:, 12] = [scipy.stats.skew(traj[1:, 1] - traj[0:-1, 1]) for traj in data]

        x[:, 13] = [scipy.stats.kurtosis(traj[:, 0]) for traj in data]
        x[:, 14] = [scipy.stats.kurtosis(traj[:, 1]) for traj in data]
        x[:, 15] = [scipy.stats.kurtosis(traj[1:, 0] - traj[0:-1, 0]) for traj in data]
        x[:, 16] = [scipy.stats.kurtosis(traj[1:, 1] - traj[0:-1, 1]) for traj in data]
        return x

    trajectories_list = [f for f in trajectories_list if f.shape[0] > 1]
    x = extract_features_polynom(trajectories_list)
    y = extract_features(trajectories_list)
    x = np.concatenate((x, y), axis=1)
    number_of_clusters = 9
    index_0, index_1 = np.where(np.isnan(x))




    model = KMeans(n_clusters=number_of_clusters, max_iter=100, tol=1e-40)
    st = time.time()
    res = model.fit_transform(x)
    labels = model.labels_

    trajectories_time_series_numpy = to_time_series_dataset(trajectories_list)



    number_of_clusters_vec = [9]
    number_of_iterations = 50
    for number_of_clusters in number_of_clusters_vec:
        ### Get proper shape: ###
        n, max_size, dim = trajectories_time_series_numpy.shape

        ### Perform TimeSeriesKMeans Fit: ###
        # print("starting label fit")
        kmeans = TimeSeriesKMeans(n_clusters=number_of_clusters, max_iter=number_of_iterations, metric='dtw')
        # tic()
        labels = kmeans.fit_predict(trajectories_time_series_numpy)
        cluster_centers = kmeans.cluster_centers_

    for cluster in range(number_of_clusters):
        colors = plt.cm.get_cmap('Paired', number_of_clusters)
        fig, ax = plt.subplots()
        img = plt.imread(f"/home/mafat/Desktop/eylon/outputs/output_herzel_test_2/first_frame.jpg")
        h,w,c = img.shape
        ax.imshow(img, extent=[0, w, h, 0])

        for trajectory_id in range(len(labels)):
            if labels[trajectory_id] == cluster and np.random.randint(0, 10) == 0:
                plt.scatter(trajectories_list[trajectory_id][:, 0], trajectories_list[trajectory_id][:, 1], color=colors(labels[trajectory_id]))

    ### Loop over cluster centers and append to cluster_centers list Line2D objects to match legend with original color: ###
    cluster_centers = []
    for id_cluster in range(number_of_clusters):
        cluster_centers.append(Line2D([0], [0], marker='o', color=colors(id_cluster), label=id_cluster))
    ax.legend(handles=cluster_centers, loc=4)

    ### Plot trajectory centers (i.e: nominal trajectories): ###
    fig, ax = plt.subplots()
    img = plt.imread(f"/home/mafat/Desktop/eylon/outputs/output_herzel_test_2/first_frame.jpg")
    h,w,c = img.shape
    ax.imshow(img, extent=[0, w, h, 0])
    cluster_centers = []
    for id_cluster in range(len(kmeans.cluster_centers_)):
        plt.scatter(kmeans.cluster_centers_[id_cluster, :, 0], kmeans.cluster_centers_[id_cluster, :, 1], color=colors(id_cluster))
        cluster_centers.append(Line2D([0], [0], marker='o', color=colors(id_cluster), label=id_cluster))
    ax.legend(handles=cluster_centers, loc=4)
    plt.show()








    debug_list = []
    for file in lane_files_list:
        if not file.split(".")[0].endswith("statistics"):
            counter_dict = defaultdict(int)
            passed_ids_in_lane = np.loadtxt(os.path.join(lane_dir, file))
            passed_ids_in_lane = np.array(passed_ids_in_lane, dtype=np.uint16)
            for id_idx, id in enumerate(id_list):
                if id in passed_ids_in_lane:
                    counter_dict[cls_dict[cls_list[id_idx]]] += 1
                    debug_list.append(id)
                else:
                    bla = 0
            with open(f"{base_dir}/lanes/{file.split('.')[0]}_statistics.txt", "w") as f:
                # Iterate over the items in the dictionary and write them to the file
                for key, value in counter_dict.items():
                    f.write(f"{key}\t{value}\n")



get_statistics(r"/home/mafat/Desktop/eylon/outputs/output_herzel_930_945")