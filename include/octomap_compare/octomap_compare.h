#ifndef LASER_SLAM_OCTOMAP_COMPARE_H_
#define LASER_SLAM_OCTOMAP_COMPARE_H_

#include <memory>

#include <glog/logging.h>
#include <nabo/nabo.h>
#include <octomap/OcTree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "octomap_compare/octomap_container.h"

class OctomapCompare {

public:
  struct CompareResult {
    Eigen::Matrix<double, 3, Eigen::Dynamic> observed_points;
    Eigen::Matrix<double, 3, Eigen::Dynamic> unobserved_points;
    Eigen::MatrixXd distances;
    double max_dist;
  };

  struct CompareParams {
    CompareParams() :
      max_vis_dist(8),
      distance_threshold(0.1),
      eps(0.3),
      min_pts(10) {}

    double max_vis_dist;
    double distance_threshold;
    double eps;
    double min_pts;
  };

private:

  // Octomap considered to be the base map for comparison.
  OctomapContainer base_octree_;

  // Octomap which will be compared to the base octomap.
  OctomapContainer comp_octree_;

  // Parameters.
  CompareParams params_;

  /// \brief Gets point color based on distance for use in visualization.
  pcl::RGB getColorFromDistance(const double& dist, const double& max_dist = 1);

public:
  OctomapCompare(const std::string& base_file, const std::string& comp_file,
                 const CompareParams& params);

  /// \brief Compare both loaded point clouds and return colored point cloud.
  CompareResult compare();

  void getChanges(const CompareResult& result,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>* output,
                  Eigen::VectorXi* cluster);

  void compareResultToPointCloud(const CompareResult& result,
                                 pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud);
};

#endif // LASER_SLAM_OCTOMAP_COMPARE_H_
