#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

TEST_SCENE = 2


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Statistical Outlier Filtering function
def statistical_outlier_filter(cloud):

    # creating a filter object
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)

    # Set threshold scale factor
    x = 0.1

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    return cloud_filtered

def voxel_grid_filter(cloud):
    # Create a VoxelGrid filter object for the point cloud
    vox = cloud.make_voxel_grid_filter()
    # Choose a voxel (leaf) size (in meters)
    LEAF_SIZE = 0.002
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
    return cloud_filtered


def pass_through_filter(cloud):
    # Create a PassThrough filter object.
    passthrough = cloud.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6097
    axis_max = 0.9
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough.set_filter_limits(axis_min, axis_max)

    #Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    return cloud_filtered


def ransac_plane_segmentation(cloud):

    # Create the segmentation object
    seg = cloud.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # was 0.02 for test world1
    max_distance = 0.02
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers
    plane = cloud.extract(inliers, negative=False)

    # Extract outliers
    objects = cloud.extract(inliers, negative=True)

    return plane, objects


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # publishing unfiltered received data
    dvi_org_pub.publish(pcl_msg)

    # Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # Statistical Outlier Filtering
    cloud_filtered = statistical_outlier_filter(cloud)

    # Publishing test data
    dvi_out_filt_pub.publish(pcl_to_ros(cloud_filtered)) # filtered (statistical outlier) data

    # Voxel Grid Downsampling
    cloud_filtered = voxel_grid_filter(cloud_filtered)

    #publishing test data
    dvi_VOX_filt_pub.publish(pcl_to_ros(cloud_filtered))  # filtered (VoxFilter) data


    # PassThrough Filter
    cloud_filtered = pass_through_filter(cloud_filtered)

    #publishing test data
    dvi_PT_filt_pub.publish(pcl_to_ros(cloud_filtered))  # filtered (PassThrough) data

    # RANSAC Plane Segmentation
    cloud_table, cloud_objects = ransac_plane_segmentation(cloud_filtered)

    #publishing test data
    dvi_RANSAC_inl_pub.publish(pcl_to_ros(cloud_table))  # filtered (RANSAC inl) data
    dvi_RANSAC_out_pub.publish(pcl_to_ros(cloud_objects))  # filtered (RANSAC out) data

    # Euclidean Clustering
    # PCL's Euclidean Clustering algorithm requires a point cloud with only spatial information
    white_cloud =  XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(500)
    ec.set_MaxClusterSize(50000)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Assign a color corresponding to each segmented object in scene
    print "number of indices " + str(len(cluster_indices))
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages and Publish ROS messages
    dvi_Clusters_pub.publish(pcl_to_ros(cluster_cloud))  # filtered (RANSAC out) data


# Exercise-3 Prediction TODOs:
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
            # Grab the points for the cluster from the extracted outliers (cloud_objects)
            pcl_cluster = cloud_objects.extract(pts_list)

            #convert the cluster from pcl to ROS using helper function
            ros_cluster = pcl_to_ros(pcl_cluster)

            # Extract histogram features
            chists = compute_color_histograms(ros_cluster, using_hsv=True)
            normals = get_normals(ros_cluster)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))

            # Make the prediction, retrieve the label for the result
            # and add it to detected_objects_labels list
            prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
            label = encoder.inverse_transform(prediction)[0]
            detected_objects_labels.append(label)

            # Publish a label into RViz
            label_pos = list(white_cloud[pts_list[0]])
            label_pos[2] += .4
            object_markers_pub.publish(make_label(label,label_pos, index))

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cluster
            detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    test_scene_num = Int32()  # message type std_msgs/Int32
    pick_pose = Pose()  # message type geometry_msgs/Pose tuple of float64 .x, .y, .z
    place_pose = Pose()

    object_name = String()
    arm_name = String()  # "right" or "left"
    test_scene_num.data = TEST_SCENE
    dict_list = []  # the list of dictionaries will be used to create yaml files

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')

    # Parse parameters into dictionaries
    object_group_dict = {}
    for d in object_list_param:
        object_group_dict[d['name']] = d['group']

    group_position_dict = {}
    for d in dropbox_param:
        group_position_dict[d['group']] = d['position']


    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the detected object list
    for object in object_list:
        # checking if object is in the object param list
        if object.label in object_group_dict:
            object_name.data = str(object.label)
            # Get the PointCloud for a given object and obtain it's centroid
            points_arr = ros_to_pcl(object.cloud).to_array()
            centr = np.mean(points_arr, axis=0)[:3]
            # Create pick pose in ROS format
            #pose = [np.asscalar(centr[0]), np.asscalar(centr[1]), np.asscalar(centr[2])]
            pick_pose.position.x = np.asscalar(centr[0])
            pick_pose.position.y = np.asscalar(centr[1])
            pick_pose.position.z = np.asscalar(centr[2])

            # Create 'place_pose' for the object
            pose = np.array(group_position_dict[object_group_dict[object.label]])
            place_pose.position.x = np.asscalar(pose[0])
            place_pose.position.y = np.asscalar(pose[1])
            place_pose.position.z = np.asscalar(pose[2])

            # Assign the arm to be used for pick_place
            if object_group_dict[object.label] == 'green':
                arm_name.data = 'right'
            else:
                arm_name.data = 'left'

            # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            dict_list.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))


    # Output your request parameters into output yaml file
    send_to_yaml('output_%s.yaml' %TEST_SCENE, dict_list)

    # Wait for 'pick_place_routine' service to come up
    rospy.wait_for_service('pick_place_routine')

    try:
        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        # Insert your message variables to be sent as a service request
        resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

        print ("Response: ",resp.success)

    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    dvi_org_pub = rospy.Publisher("/dvi_org", PointCloud2, queue_size=1)
    dvi_out_filt_pub = rospy.Publisher("/dvi_out_filt", PointCloud2, queue_size=1)
    dvi_VOX_filt_pub = rospy.Publisher("/dvi_VOX_filt", PointCloud2, queue_size=1)
    dvi_PT_filt_pub = rospy.Publisher("/dvi_PT_filt", PointCloud2, queue_size=1)
    dvi_RANSAC_inl_pub = rospy.Publisher("/dvi_RANSAC_inl", PointCloud2, queue_size=1)
    dvi_RANSAC_out_pub = rospy.Publisher("/dvi_RANSAC_out", PointCloud2, queue_size=1)
    dvi_Clusters_pub = rospy.Publisher("/dvi_Clusters", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model_%s.sav' %TEST_SCENE, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
