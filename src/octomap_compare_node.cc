#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include "octomap_compare/octomap_compare.h"

void changesToPointCloud(const Eigen::Matrix<double, 3, Eigen::Dynamic> matrix,
                        const Eigen::VectorXi& cluster,
                        pcl::PointCloud<pcl::PointXYZRGB>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  std::srand(0);
  std::unordered_map<unsigned int, pcl::RGB> color_map;
  std::unordered_set<int> cluster_indices;
  for (unsigned int i = 0; i < cluster.rows(); ++i) {
    cluster_indices.insert(cluster(i));
    pcl::RGB color;
    color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
    color_map[cluster(i)] = color;
  }
  std::cout << "Found " << cluster_indices.size() - 1 << " clusters\n";
  for (unsigned int i = 0; i < matrix.cols(); ++i) {
    if (cluster(i) != 0) {
      pcl::RGB color = color_map[cluster(i)];
      pcl::PointXYZRGB point(color.r, color.g, color.b);
      point.x = matrix(0,i);
      point.y = matrix(1,i);
      point.z = matrix(2,i);
      cloud->push_back(point);
    }
  }
  std::cout << cloud->size() << " points left after clustering\n";
}

void matrixToPointCloud(const Eigen::Matrix<double, 3, Eigen::Dynamic> matrix,
                        pcl::PointCloud<pcl::PointXYZ>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  for (unsigned int i = 0; i < matrix.cols(); ++i) {
    pcl::PointXYZ point(matrix(0,i), matrix(1,i), matrix(2,i));
    cloud->push_back(point);
  }
}

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
  // Keep topic advertised.
  while (ros::ok()) {
    ros::Duration(1.0).sleep();
  }
}
