#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include "octomap_compare/octomap_compare.h"
#include "octomap_compare/octomap_compare_utils.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "octomap_compare");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("~");

  ros::WallTime start = ros::WallTime::now();

  // Get parameters.
  OctomapCompare::CompareParams params;
  nh.param("max_vis_dist", params.max_vis_dist, params.max_vis_dist);
  nh.param("distance_threshold", params.distance_threshold, params.distance_threshold);
  nh.param("eps", params.eps, params.eps);
  nh.param("min_pts", params.min_pts, params.min_pts);
  nh.param("k_nearest_neighbor", params.k_nearest_neighbor, params.k_nearest_neighbor);
  nh.param("show_unobserved_voxels", params.show_unobserved_voxels, params.show_unobserved_voxels);
  nh.param("distance_metric", params.distance_metric, params.distance_metric);
  std::string base_file, comp_file;
  if (!nh.getParam("base_file", base_file)) {
    ROS_ERROR("Did not find base file parameter");
    return EXIT_FAILURE;
  }
  if (!nh.getParam("comp_file", comp_file)) {
    ROS_ERROR("Did not find comp file parameter");
    return EXIT_FAILURE;
  }

  OctomapCompare compare(base_file, comp_file, params);

  OctomapCompare::CompareResult result = compare.compare();

  Eigen::Matrix<double, 3, Eigen::Dynamic> changes;
  Eigen::VectorXi cluster;
  compare.getChanges(result, &changes, &cluster);

  pcl::PointCloud<pcl::PointXYZRGB> changes_point_cloud;
  changesToPointCloud(changes, cluster, &changes_point_cloud);

  pcl::PointCloud<pcl::PointXYZRGB> distance_point_cloud;
  compare.compareResultToPointCloud(result, &distance_point_cloud);

  ros::Publisher color_pub, changes_pub;
  color_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("distance_cloud", 1, true);
  changes_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("changes_cloud", 1, true);
  color_pub.publish(distance_point_cloud);
  changes_pub.publish(changes_point_cloud);
  std::cout << "Published changes point cloud\n";
  std::cout << "Published color point cloud\n";

  ros::WallDuration dif = ros::WallTime::now() - start;
  std::cout << "This all took " << dif.toSec() << " seconds\n";

  // Keep topic advertised.
  while (ros::ok()) {
    ros::Duration(1.0).sleep();
  }
}
