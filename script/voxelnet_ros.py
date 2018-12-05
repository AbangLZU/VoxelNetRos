#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import sys
import glob
import argparse
import os
import time
import tensorflow as tf

sys.path.append("../voxelnet/")
from model import RPN3D
from config import cfg
from utils import *
from utils.preprocess import process_pointcloud
from utils.kitti_loader import build_input

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]


# code from /opt/ros/kinetic/lib/python2.7/dist-packages/tf/transformations.py


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    # print("ak : {}".format(type(ak)))
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4,), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion


class Processor_ROS:
    def __init__(self, np_p_ranged):
        self.np_p_ranged = np_p_ranged

    def run(self):
        raw_lidar = self.np_p_ranged
        # print(raw_lidar.shape) #  DEBUG
        voxel = process_pointcloud(raw_lidar)
        return raw_lidar, voxel


def dataset_generator(np_p_ranged, batch_size=1, multi_gpu_sum=1):
    proc = Processor_ROS(np_p_ranged)
    raw_lidar, voxel = proc.run()

    # print("feature_buffer: {}".format(voxel['feature_buffer'].shape)) #  DEBUG [----, 35, 7]
    # print("coordinate_buffer: {}".format(voxel['coordinate_buffer'].shape)) #  DEBUG [----, 3]
    # print("number_buffer: {}".format(voxel['number_buffer'].shape)) #  DEBUG [----]

    # only for voxel -> [gpu, k_single_batch, ...]
    vox_feature, vox_number, vox_coordinate = [], [], []
    single_batch_size = int(batch_size / multi_gpu_sum)

    _, per_vox_feature, per_vox_number, per_vox_coordinate = build_input_ros(
        voxel)
    vox_feature.append(per_vox_feature)
    vox_number.append(per_vox_number)
    vox_coordinate.append(per_vox_coordinate)

    ret = (
        np.array(vox_feature),
        np.array(vox_number),
        np.array(vox_coordinate),
        np.array(raw_lidar)
    )

    return ret


#  point cloud topic's subscriber callback function


def velo_callback(msg):
    global sess, model

    arr_bbox = BoundingBoxArray()

    pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=(
        "x", "y", "z", "intensity","ring"))
    np_p = np.array(list(pcl_msg), dtype=np.float32)
    
    # np_p = np.delete(np_p, -1, 1)  # delete "ring" field
    print(np_p.shape)
    
    dataset = dataset_generator(np_p, batch_size=1, multi_gpu_sum=1)
    # print("{} {} {} {}".format(dataset[0],dataset[1],dataset[2],dataset[3])) #  DEBUG
    results = model.predict_step_ros(sess, dataset)
    #  publish to /velodyne_poitns_modified
    publish_test(np_p, msg.header.frame_id)
    # results: (N, N') (class, x, y, z, h, w, l, rz, score)
    if len(results[0]) != 0:
        # print("len(results[0]) : {} ".format(len(results[0])))
        for result in results[0]:
            # print("[+] result: {}".format(result)) #  DEBUG
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            # bbox.header.stamp = rospy.Time.now()

            # print("result[7] : {} ".format(result[7]))
            q = quaternion_from_euler(0, 0, float(result[7]))

            bbox.pose.orientation.x = q[0]
            bbox.pose.orientation.y = q[1]
            bbox.pose.orientation.z = q[2]
            bbox.pose.orientation.w = q[3]
            bbox.pose.position.x = float(result[1])
            bbox.pose.position.y = float(result[2])
            bbox.pose.position.z = float(result[3])
            bbox.dimensions.x = float(result[6])
            bbox.dimensions.y = float(result[5])
            bbox.dimensions.z = float(result[4])

            arr_bbox.boxes.append(bbox)

    arr_bbox.header.frame_id = msg.header.frame_id
    # arr_bbox.header.stamp = rospy.Time.now()
    # print("arr_bbox.boxes.size() : {} ".format(len(arr_bbox.boxes)))
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes.clear()


#  publishing function for DEBUG
def publish_test(np_p_ranged, frame_id):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id

    x = np_p_ranged[:, 0].reshape(-1)
    y = np_p_ranged[:, 1].reshape(-1)
    z = np_p_ranged[:, 2].reshape(-1)

    # if intensity field exists
    if np_p_ranged.shape[1] == 4:
        i = np_p_ranged[:, 3].reshape(-1)
    else:
        i = np.zeros((np_p_ranged.shape[0], 1)).reshape(-1)

    cloud = np.stack((x, y, z, i))

    # point cloud segments
    # 4 PointFields as channel description
    msg_segment = pc2.create_cloud(header=header,
                                   fields=_make_point_field(4),
                                   points=cloud.T)

    #  publish to /velodyne_points_modified
    pub_velo.publish(msg_segment)  # DEBUG


#  voxelnet initializer
def voxelnet_init():
    global sess, model

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                visible_device_list=cfg.GPU_AVAILABLE,
                                allow_growth=True)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={
            "GPU": cfg.GPU_USE_COUNT,
        },
        allow_soft_placement=True,
    )

    sess = tf.Session(config=config)

    model = RPN3D(
        cls=cfg.DETECT_OBJ,
        single_batch_size=args.single_batch_size,
        avail_gpus=cfg.GPU_AVAILABLE.split(',')
    )

    if tf.train.get_checkpoint_state(save_model_dir):
        print("Reading model parameters from %s" % save_model_dir)
        model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))


#  code from SqueezeSeg (inspired from Durant35)


def hv_in_range(x, y, z, fov, fov_type='h'):
    """
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit

    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`

    Return:
    `cond`:condition of points within fov or not

    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' "
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='pre_trained_car',
                        help='set log tag')
    parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=1,  # def: 2
                        help='set batch size for each gpu')
    args = parser.parse_args()

    save_model_dir = os.path.join('../voxelnet/save_model', args.tag)

    #  initializing voxelnet
    voxelnet_init()

    #  code added for using ROS
    rospy.init_node('voxelnet_ros_node')
    print('hahah')

    sub_ = rospy.Subscriber("velodyne_points", PointCloud2,
                            velo_callback, queue_size=1)

    pub_velo = rospy.Publisher(
        "velodyne_points_modified", PointCloud2, queue_size=1)
    pub_arr_bbox = rospy.Publisher(
        "voxelnet_arr_bbox", BoundingBoxArray, queue_size=10)
    pub_bbox = rospy.Publisher("voxelnet_bbox", BoundingBox, queue_size=1)
    print("[+] voxelnet_ros_node has started!")
    rospy.spin()
