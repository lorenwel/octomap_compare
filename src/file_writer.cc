#include "octomap_compare/file_writer.h"

#include <limits>

#include <glog/logging.h>
#include <ros/callback_queue.h>
#include <visualization_msgs/MarkerArray.h>

FileWriter::FileWriter(ros::NodeHandle &nh,
                       ClusterCentroidVector& cluster_centroids,
                       const std::string &file_name,
                       const ros::Time& stamp) :
                       nh_(nh), stamp_(stamp),
                       cluster_centroids_(cluster_centroids),
                       shut_down_(false), time_stopped_(false) {
  out_file_.open(file_name, std::ios::app);
  CHECK(out_file_.is_open()) << "Could not open file \"" << file_name << "\".";
  out_file_ << "\n";

  // Set separate callback queue so this can be called from within callback.
  ros::CallbackQueueInterface* backup = nh_.getCallbackQueue();
  ros::CallbackQueue queue;
  nh_.setCallbackQueue(&queue);

  centroid_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("centroids", 2, true);
  point_sub_ = nh_.subscribe("/clicked_point", 10, &FileWriter::pointCallback, this);
  clock_sub_ = nh_.subscribe("/clock", 1, &FileWriter::clockCallback, this);

  publishCentroids();

  ros::WallDuration dur(0.1);
  while (!shut_down_ && ros::ok()) {
    queue.callAvailable();
    dur.sleep();
  }

  removeCentroids();
  point_sub_.shutdown();
  clock_sub_.shutdown();
  centroid_pub_.shutdown();
  // Reset callback queue.
  nh_.setCallbackQueue(backup);
}

void FileWriter::removeCentroids() {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker marker;
  const std::string base_text("Cluster: ");
  marker.header.frame_id = "map";
  marker.header.stamp = stamp_;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::DELETE;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.5;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;

  for (const auto& centroid : cluster_centroids_) {
    if (centroid.first != 0) {
      marker.id = centroid.first;
      marker.pose.position.x = centroid.second.x();
      marker.pose.position.y = centroid.second.y();
      marker.pose.position.z = centroid.second.z();
      marker.text = base_text + std::to_string(centroid.first);

      array.markers.push_back(marker);
    }
  }
  centroid_pub_.publish(array);
}

void FileWriter::publishCentroids() {
  visualization_msgs::MarkerArray array;
  visualization_msgs::Marker marker;
  const std::string base_text("Cluster: ");
  marker.header.frame_id = "map";
  marker.header.stamp = stamp_;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.5;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;

  for (const auto& centroid : cluster_centroids_) {
    if (centroid.first != 0) {
      marker.id = centroid.first;
      marker.pose.position.x = centroid.second.x();
      marker.pose.position.y = centroid.second.y();
      marker.pose.position.z = centroid.second.z();
      marker.text = base_text + std::to_string(centroid.first);

      array.markers.push_back(marker);
    }
  }
  centroid_pub_.publish(array);
}

int FileWriter::findClosestCentroid(const Eigen::Vector3d& point) {
  int index = 0;
  float min_dist = std::numeric_limits<float>::max();
  for (const auto& cur : cluster_centroids_) {
    const float cur_dist = (point - cur.second).squaredNorm();
    if (cur_dist < min_dist) {
      min_dist = cur_dist;
      index = cur.first;
    }
  }
  if (index == 0) LOG(WARNING) << "Got outlier cluster as closest centroid.";
  return index;
}

void FileWriter::pointCallback(const geometry_msgs::PointStamped& point_msg) {
  Eigen::Vector3d point(point_msg.point.x, point_msg.point.y, point_msg.point.z);
  const int dynamic_cluster = findClosestCentroid(point);
  std::cout << "Set cluster " << dynamic_cluster << " dynamic.\n";
  std::cout.flush();
  out_file_ << dynamic_cluster << ", ";
}

void FileWriter::clockCallback(const rosgraph_msgs::Clock& clock) {
  if (clock.clock == last_clock_) time_stopped_ = true;
  else if (time_stopped_) {
    shut_down_ = true;
    std::cout << "Received FileWriter shutdown notice.\n";
    std::cout.flush();
  }
  last_clock_ = clock.clock;
}
