




def Update_QS_DataFrame(QS_stats_pd, params, flag_drone_trajectory_inside_BB_list):
    # TODO: at the end it will need to loop on all actual drones present...so i guess on all flashlight trajectories found!
    # TODO: perhaps instead of "flag_drone_in_BB" which only check one large bounding box, i should first squeeze the bounding box or check direction vec???
    # TODO: i shuold really understand what to do about drone swarms
    if len(flag_drone_trajectory_inside_BB_list) > 0:
        #(*). Trajectories were found, fill stuff accordingly:
        for trajectory_index in np.arange(len(flag_drone_trajectory_inside_BB_list)):
            current_pd = {"distance": [np.float(params.distance)],
                          "drone_type": [params.drone_type],
                          "drone_movement": [params.drone_movement],
                          "background_scene": [params.background_scene],
                          "flag_drone_in_BB": [flag_drone_trajectory_inside_BB_list[trajectory_index]],
                          'flag_was_there_drone': (params.was_there_drone == 'True'),
                          'flag_was_drone_detected': True,
                          'experiment_name': os.path.split(params.experiment_folder)[-1],
                          'full_filename_sequence': params.results_folder_seq
                          }
            current_pd = pd.DataFrame.from_dict(current_pd)
            if len(QS_stats_pd) == 0:
                QS_stats_pd = pd.DataFrame.from_dict(current_pd)
            else:
                QS_stats_pd = QS_stats_pd.append(current_pd, ignore_index=True)
    else:
        #(*). No Trajectories were found, fill stuff accordingly:
        current_pd = {"distance": [params.distance],
                      "drone_type": [params.drone_type],
                      "drone_movement": [params.drone_movement],
                      "background_scene": [params.background_scene],
                      "flag_drone_in_BB": False,
                      'flag_was_there_drone': (params.was_there_drone == 'True'),
                      'flag_was_drone_detected': False,
                      'experiment_name': os.path.split(params.experiment_folder)[-1],
                      'full_filename_sequence': params.results_folder_seq
                      }
        current_pd = pd.DataFrame.from_dict(current_pd)
        if len(QS_stats_pd) == 0:
            QS_stats_pd = pd.DataFrame.from_dict(current_pd)
        else:
            QS_stats_pd = QS_stats_pd.append(current_pd, ignore_index=True)

    return QS_stats_pd


def play_with_DataFrame(QS_stats_pd):
    QS_stats_pd.head()
    QS_stats_pd.tail()
    QS_stats_pd.index  # display indices
    QS_stats_pd.columns
    QS_stats_pd.to_numpy()
    QS_stats_pd.describe()
    QS_stats_pd.sort_index(axis=1, ascending=False)
    QS_stats_pd.sort_values(by='distance')
    QS_stats_pd['distance']
    QS_stats_pd[0:1]  # select
    QS_stats_pd.loc[:, ['distance', 'drone_type']]  # select by label
    QS_stats_pd.iloc[1]  # row selection
    QS_stats_pd.iloc[[0], [0, 1]]  # row and column selection
    QS_stats_pd[QS_stats_pd['distance'] == '1500']  # select by condition
    QS_stats_pd.mean()
    QS_stats_pd.columns = [x.lower() for x in QS_stats_pd.columns]  # change column names
    QS_stats_pd['distance'] = QS_stats_pd['distance'].astype(np.float)  # transform to float
    # (*). get only those from distance 1500:
    QS_stats_pd[QS_stats_pd['distance'] == 1500]
    # (*). get only those with urban background:
    QS_stats_pd[QS_stats_pd['background_scene'] == 'urban']
    # (*). get only those with urban background and distance from 1500-2000:
    QS_stats_pd[QS_stats_pd['background_scene'] == 'urban'][QS_stats_pd['distance'] == 1500]
    # (*). get only those where no drone was present (for 24/7 FAR statistics):
    QS_stats_pd[QS_stats_pd['flag_was_there_drone'] == 'True']  # TODO: turn this into boolean value for fuck sake
    # (*). get only those where no drone was detected but drone was present:
    QS_stats_pd[QS_stats_pd['flag_was_there_drone'] == 'True'][QS_stats_pd['flag_was_drone_detected'] == True]  # TODO: why is it boolean in one place and string on the other
    # (*). for those with distance 1500 get confusion matrix
    QS_stats_pd == 'True'


def save_DataFrame_to_csv(input_df, full_filename):
    input_df.to_csv(full_filename)


def load_DataFrame_from_csv(full_filename):
    return pd.read_csv(full_filename).iloc[:, 1:]


def save_DataFrame_DifferentMethods():
    QS_stats_pd.to_csv(os.path.join(params.ExperimentsFold, 'QS_stats_pd.csv'))
    pd.read_csv(os.path.join(params.ExperimentsFold, 'QS_stats_pd.csv')).iloc[:, 1:]

    ### Save/Load From Disk Using Jason: ###
    QS_stats_pd.to_pickle(os.path.join(params.ExperimentsFold, 'QS_stats_pd.json'))
    QS_stats_pd = pd.read_pickle(os.path.join(params.ExperimentsFold, 'QS_stats_pd.json'))

    ### Save/Load From Disk Using HDF: ###
    QS_stats_pd.to_hdf(os.path.join(params.ExperimentsFold, 'QS_stats_pd.h5'), 'table', append=True)
    pd.HDFStore(os.path.join(params.ExperimentsFold, 'QS_stats_pd.h5')).append('Table', QS_stats_pd)

    ### Save Updated DataFrame To Disk Using numpy: ###
    np.save(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy'), QS_stats_pd, allow_pickle=True)
    np.save(os.path.join(params.ExperimentsFold, 'QS_stats_pd_columns.npy'), QS_stats_pd.columns, allow_pickle=True)
    QS_stats_pd = np.load(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy'), allow_pickle=True)
    QS_stats_pd_columns = np.load(os.path.join(params.ExperimentsFold, 'QS_stats_pd_columns.npy'), allow_pickle=True)
    QS_stats_pd = pd.DataFrame(QS_stats_pd)
    QS_stats_pd.columns = QS_stats_pd_columns










