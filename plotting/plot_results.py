import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
from pyquaternion import Quaternion

parser = argparse.ArgumentParser(description='Plot the trajectory of the drone')
parser.add_argument('result_path', type=str, help='The path to the result file')
parser.add_argument('truth_path', type=str, help='The path to the ground truth file')
args = parser.parse_args()

datas = []

# result trajectory
result_data = pd.read_csv(args.result_path, sep=' ', header=None, skiprows=[0],
                          names=['timestamp', 'px', 'py', 'pz', 'rx', 'ry', 'rz', 'rw'],
                          usecols=[0, 1, 2, 3, 4, 5, 6, 7])

truth_data = pd.read_csv(args.truth_path, sep=',', header=None, skiprows=[0,1],
                         names=['timestamp', 'px', 'py', 'pz', 'rw', 'rx', 'ry', 'rz'],
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7])

# use timestamp as time series, but change to seconds first (is nanoseconds in file)
result_data['timestamp'] = result_data['timestamp'].astype('datetime64[ns]')
result_data = result_data.set_index('timestamp')
truth_data['timestamp'] = truth_data['timestamp'].astype('datetime64[ns]')
truth_data = truth_data.set_index('timestamp')

def plot_axis_over_time(axis, label):
    fig, ax = plt.subplots()
    ax.plot(result_data.index, result_data[axis])
    ax.plot(truth_data.index, truth_data[axis])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(label)
    ax.legend(['Estimated', 'Ground Truth'])
    plt.show()


print(result_data.columns)

# skip start
start_time = max(result_data.index[0], truth_data.index[0])
start_time = start_time + pd.Timedelta('2s')
truth_data = truth_data.loc[start_time:]
result_data = result_data.loc[start_time:]

print(result_data.columns)

# The estimated trajectory starts at 0 so it needs to be shifted to the first ground truth pose
def get_first_pose(data):
    first_rotation = Quaternion(data['rw'].iloc[0], data['rx'].iloc[0], data['ry'].iloc[0], data['rz'].iloc[0])
    first_position = np.array([data['px'].iloc[0], data['py'].iloc[0], data['pz'].iloc[0]])
    return first_position, first_rotation

first_truth_position, first_truth_rotation = get_first_pose(truth_data)
first_result_position, first_result_rotation = get_first_pose(result_data)
first_result_rotation_inv = first_result_rotation.inverse

print("First truth position:", first_truth_position, "rotation:", first_truth_rotation, "at time", truth_data.index[0])
print("First result position:", first_result_position, "rotation:", first_result_rotation, "at time", result_data.index[0])
#print("Relative transform: ", relative_transform)

def transform_trajectory(row):
    pos = np.array([row['px'], row['py'], row['pz']])
    rot = Quaternion(row['rw'], row['rx'], row['ry'], row['rz'])
    
    pos = first_result_rotation_inv.rotate(pos - first_result_position)
    rot = first_result_rotation_inv * rot
    
    pos = first_truth_rotation.rotate(pos) + first_truth_position
    rot = first_truth_rotation * rot
    
    return pd.Series([
            row.name,
            pos[0], pos[1], pos[2],
            rot.x, rot.y, rot.z, rot.w
        ], index=['timestamp', 'px', 'py', 'pz', 'rx', 'ry', 'rz', 'rw'])

# print columns first as debug
print(result_data.columns)
result_data = result_data.apply(lambda row: transform_trajectory(row), axis=1)

first_result2_position, first_result2_rotation = get_first_pose(result_data)
print("First result position after transform:", first_result2_position, "rotation:", first_result2_rotation, "at time", result_data.index[0])

def trajectory_error(result, truth):
    downsampled_truth = truth.resample('0.1s').mean()
    downsampled_result = result.resample('0.1s').mean()
    # align the two datasets
    aligned_truth = downsampled_truth.loc[result.index[0]:]
    aligned_result = downsampled_result.loc[result.index[0]:]
    
    # calculate the error
    offsets = aligned_truth[['px', 'py', 'pz']] - aligned_result[['px', 'py', 'pz']]
    distances = np.sqrt(np.sum(np.square(offsets), axis=1))
    
    rmse = np.sqrt(np.mean(np.square(distances)))
    print("mean error: ", np.mean(distances))
    print("RMSE: ", rmse)
    print("worst error: ", np.max(distances))
    print("worst 20% error: ", np.percentile(distances, 80))
    return rmse

trajectory_error(result_data, truth_data)

# plot result data and truth in 2D using timestamps as x-axis
plot_axis_over_time('px', 'X')
plot_axis_over_time('py', 'Y')
plot_axis_over_time('pz', 'Z')

# Initialize a 3D plot with all series
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for data in [result_data, truth_data]:
    ax.plot(data['px'], data['py'], data['pz'])
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend(['Estimated', 'Ground Truth'])

plt.show()
