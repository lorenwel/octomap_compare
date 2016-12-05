#ifndef FILE_WRITER_H_
#define FILE_WRITER_H_

#include <fstream>

#include <Eigen/Dense>
#include <geometry_msgs/PointStamped.h>
#include <rosgraph_msgs/Clock.h>
#include <ros/ros.h>

#include "octomap_compare/octomap_compare_utils.h"

class FileWriter {
  ros::NodeHandle& nh_;
  ros::Publisher centroid_pub_;
  ros::Subscriber point_sub_;
  ros::Subscriber clock_sub_;  // Use to stop writing to file.

  ClusterCentroidVector cluster_centroids_;
  std::ofstream out_file_;
  ros::Time stamp_;

  ros::Time last_clock_;

  bool shut_down_;
  bool time_stopped_;

  void pointCallback(const geometry_msgs::PointStamped& point_msg);

  void clockCallback(const rosgraph_msgs::Clock& clock);

  int findClosestCentroid(const Eigen::Vector3d& point);

  void publishCentroids();

  void removeCentroids();

public:
  FileWriter(ros::NodeHandle& nh,
             ClusterCentroidVector& cluster_centroids,
             const std::string& file_name, const ros::Time& stamp);

  ~FileWriter() {
    out_file_.close();
  }
};

#endif /* FILE_WRITER_H_ */
