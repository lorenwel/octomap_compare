#include <chrono>

#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pointmatcher_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

#include "octomap_compare/octomap_compare.h"
#include "octomap_compare/octomap_compare_utils.h"

class Online {

  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher changes_pub_;
  ros::Publisher color_pub_;
  tf::TransformListener tf_listener_;

  OctomapCompare octomap_compare_;
  OctomapCompare::CompareParams params_;

  void cloudCallback(const sensor_msgs::PointCloud2& cloud) {
    sensor_msgs::PointCloud2 map_cloud;
    if (pcl_ros::transformPointCloud("map", cloud, map_cloud, tf_listener_)) {
      PM::DataPoints intermediate_points =
          PointMatcher_ros::rosMsgToPointMatcherCloud<double>(map_cloud);
      Eigen::MatrixXd points = pointMatcherToMatrix3dEigen(intermediate_points);

      auto start = std::chrono::high_resolution_clock::now();

      PointCloudContainer comp_octree(points);
      OctomapCompare::CompareResult result = octomap_compare_.compare(comp_octree);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      Eigen::Matrix<double, 3, Eigen::Dynamic> changes_appear, changes_disappear;
      Eigen::VectorXi cluster_appear, cluster_disappear;
      octomap_compare_.getChanges(result, &changes_appear, &changes_disappear, &cluster_appear, &cluster_disappear);

      pcl::PointCloud<pcl::PointXYZRGB> changes_point_cloud;
      changesToPointCloud(changes_appear, changes_disappear,
                          cluster_appear, cluster_disappear,
                          params_.color_changes,
                          &changes_point_cloud);

      pcl::PointCloud<pcl::PointXYZRGB> distance_point_cloud;
      octomap_compare_.compareResultToPointCloud(result, &distance_point_cloud);

      color_pub_.publish(distance_point_cloud);
      changes_pub_.publish(changes_point_cloud);
      std::cout << "Comparing took " << duration.count() << " seconds\n";
    }
    else {
      ROS_WARN("Could not transform point cloud");
    }
  }

public:

  Online(const ros::NodeHandle& nh,
         const std::string& base_file,
         const OctomapCompare::CompareParams& params) :
         nh_(nh), octomap_compare_(base_file, params), params_(params) {
    cloud_sub_ = nh_.subscribe("/dynamic_point_cloud", 1, &Online::cloudCallback, this);
    changes_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("changes", 1, true);
    color_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("heat_map", 1, true);

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
  nh.param("initial_transform", params.initial_transform, params.initial_transform);
  nh.param("perform_icp", params.perform_icp, params.perform_icp);
  std::string base_file, comp_file;
  if (!nh.getParam("base_file", base_file)) {
    ROS_ERROR("Did not find base file parameter");
    return EXIT_FAILURE;
  }

  Online online(nh, base_file, params);

  ros::spin();
}
