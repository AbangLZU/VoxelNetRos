import numpy as np
import rospy

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import os


def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

#  publishing function for DEBUG
def publish_pc(np_array, frame_id):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id

    x = np_array[:, 0].reshape(-1)
    y = np_array[:, 1].reshape(-1)
    z = np_array[:, 2].reshape(-1)

    # if intensity field exists
    if np_array.shape[1] == 4:
        i = np_array[:, 3].reshape(-1)
    else:
        i = np.zeros((np_array.shape[0], 1)).reshape(-1)

    cloud = np.stack((x, y, z, i))

    # point cloud segments
    # 4 PointFields as channel description
    msg_segment = pc2.create_cloud(header=header,
                                   fields=_make_point_field(4),
                                   points=cloud.T)

    #  publish to /velodyne_points_modified
    pub_velo.publish(msg_segment)  # DEBUG


if __name__ == '__main__':
    test_root = '../data/lidar_2d'
    # test_root = '/home/adam/data/voxel_net/KITTI/testing/velodyne'
    file_list = []
    tmp_list = os.listdir(test_root)
    tmp_list.sort()
    for f in tmp_list:
        cur_file = os.path.join(test_root, f)
        file_list.append(cur_file)


    rospy.init_node('pub_kitti_point_cloud')
    pub_velo = rospy.Publisher("velodyne_points", PointCloud2, queue_size=1)
    rate = rospy.Rate(2)

    pc_num_counter = 0
    while not rospy.is_shutdown():
        rospy.loginfo(file_list[pc_num_counter])

        # pc_data = np.fromfile(file_list[pc_num_counter], dtype=np.float32).reshape(-1, 4)

        pc_data = np.load(file_list[pc_num_counter])
        pc_data = pc_data[:, :, :4]
        pc_data = pc_data.reshape(-1, 4)

        publish_pc(pc_data, 'velodyne')

        pc_num_counter = pc_num_counter + 1
        if pc_num_counter >= len(file_list):
            pc_num_counter = 0
        rate.sleep()

