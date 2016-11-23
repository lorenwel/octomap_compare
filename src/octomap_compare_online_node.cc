#include <chrono>

#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pointmatcher_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include "octomap_compare/octomap_compare.h"
#include "octomap_compare/octomap_compare_utils.h"

class Online {

  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher changes_pub_;
  ros::Publisher color_pub_;
  ros::Publisher marker_pub_;
  tf::TransformListener tf_listener_;

  OctomapCompare octomap_compare_;
  OctomapCompare::CompareParams params_;

  void cloudCallback(const sensor_msgs::PointCloud2& cloud) {
    if (cloud.width * cloud.height == 0) {
      ROS_WARN("Received empty cloud");
      return;
    }
    tf::StampedTransform T_map_robot;
    try {
      tf_listener_.lookupTransform("map",
                                   cloud.header.frame_id,
                                   cloud.header.stamp,
                                   T_map_robot);
      Eigen::Affine3d T_initial;
      tf::transformTFToEigen(T_map_robot, T_initial);
      PM::DataPoints intermediate_points =
          PointMatcher_ros::rosMsgToPointMatcherCloud<double>(cloud);
      Eigen::MatrixXd points = pointMatcherToMatrix3dEigen(intermediate_points);

      auto start = std::chrono::high_resolution_clock::now();

      visualization_msgs::MarkerArray array;
      PointCloudContainer comp_octree(points, params_.spherical_transform, params_.std_dev);
      OctomapCompare::CompareResult result = octomap_compare_.compare(comp_octree,
                                                                      T_initial.matrix(),
                                                                      &array);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      Eigen::Matrix<double, 3, Eigen::Dynamic> changes_appear, changes_disappear;
      Eigen::VectorXi cluster_appear, cluster_disappear;
      octomap_compare_.getChanges(result, &changes_appear, &changes_disappear,
                                          &cluster_appear, &cluster_disappear);

      pcl::PointCloud<pcl::PointXYZRGB> changes_point_cloud;
      changesToPointCloud(changes_appear, changes_disappear,
                          cluster_appear, cluster_disappear,
                          params_.color_changes,
                          &changes_point_cloud);

      pcl::PointCloud<pcl::PointXYZRGB> distance_point_cloud;
      octomap_compare_.compareResultToPointCloud(result, &distance_point_cloud);

      color_pub_.publish(distance_point_cloud);
      changes_pub_.publish(changes_point_cloud);
      marker_pub_.publish(array);
      std::cout << "Comparing took " << duration.count() << " seconds\n";
    }
    catch (tf::TransformException& e) {
      ROS_ERROR("Could not transform point cloud: %s", e.what());
      ROS_WARN_ONCE("This is expected for the first call back");
    }
  }

public:

  Online(const ros::NodeHandle& nh,
         const std::string& base_file,
         const std::string& cloud_topic,
         const OctomapCompare::CompareParams& params) :
         nh_(nh), octomap_compare_(base_file, params), params_(params) {
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &Online::cloudCallback, this);
    changes_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("changes", 1, true);
    color_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("heat_map", 1, true);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("ellipses", 1, true);

  }

};

int main(int argc, char** argv) {
  ros::init(argc, argv, "octomap_compare");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("~");

  // Get parameters.
  OctomapCompare::CompareParams params;
  nh.param("max_vis_dist", params.max_vis_dist, params.max_vis_dist);
  nh.param("distance_threshold", params.distance_threshold, params.distance_threshold);
  nh.param("eps", params.eps, params.eps);
  nh.param("min_pts", params.min_pts, params.min_pts);
  nh.param("k_nearest_neighbor", params.k_nearest_neighbor, params.k_nearest_neighbor);
  nh.param("show_unobserved_voxels", params.show_unobserved_voxels, params.show_unobserved_voxels);
  nh.param("distance_computation", params.distance_computation, params.distance_computation);
  nh.param("color_changes", params.color_changes, params.color_changes);
  nh.param("perform_icp", params.perform_icp, params.perform_icp);
  nh.param("clustering_algorithm", params.clustering_algorithm, params.clustering_algorithm);
  std::vector<double> temp_transform({1, 0, 0, 0, 1, 0, 0, 0, 1});
  nh.param("spherical_transform", temp_transform, temp_transform);
  params.spherical_transform = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(temp_transform.data());
  std::string base_file;
  if (!nh.getParam("base_file", base_file)) {
    ROS_ERROR("Did not find base file parameter");
    return EXIT_FAILURE;
  }
  std::string cloud_topic;
  if (!nh.getParam("cloud_topic", cloud_topic)) {
    ROS_WARN("Did not find cloud_topic parameter. Set to \"/dynamic_point_cloud\"");
  }
  std::vector<double> std_dev_vec;
  if (!nh.getParam("std_dev", std_dev_vec)) {
    ROS_ERROR("No standard deviation specified");
    return EXIT_FAILURE;
  }
  else {
    params.std_dev = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(std_dev_vec.data());
  }

  Online online(nh, base_file, cloud_topic, params);

  ros::spin();
}
