#ifndef LASER_SLAM_OCTOMAP_COMPARE_H_
#define LASER_SLAM_OCTOMAP_COMPARE_H_

#include <memory>
#include <unordered_map>

#include <glog/logging.h>
#include <nabo/nabo.h>
#include <octomap/OcTree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "octomap_compare/octomap_container.h"

class OctomapCompare {

public:
  typedef std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash> KeyToDistMap;

  struct CompareResult {
    Eigen::Matrix<double, 3, Eigen::Dynamic> base_observed_points;
    Eigen::Matrix<double, 3, Eigen::Dynamic> comp_observed_points;
    Eigen::MatrixXd base_distances;
    Eigen::MatrixXd comp_distances;
    Eigen::Matrix<double, 3, Eigen::Dynamic> unobserved_points;
    double max_dist;
  };

  struct CompareParams {
    CompareParams() :
      max_vis_dist(8),
      distance_threshold(0.1),
      eps(0.3),
      min_pts(10),
      k_nearest_neighbor(1),
      show_unobserved_voxels(true){}

    // Distance value used as maximum for voxel coloring in visualization.
    double max_vis_dist;
    // Distances threshold for change detection.
    double distance_threshold;
    // DBSCAN: Radius around point to be considered a neighbor.
    double eps;
    // DBSCAN: Minimum number of points in neighborhood to be considered a base point.
    double min_pts;
    // Number of neighbors to query for. Currently only closest neighbor is used.
    // Setting this to 1 enbables some optimizations.
    int k_nearest_neighbor;
    // Show unobserved voxels in visualization of distance between octomaps.
    bool show_unobserved_voxels;
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

  /// \brief Get changes with distances from comp voxel to closest base voxel.
  double compareForward(std::list<Eigen::Vector3d>* observed_points,
                        std::list<Eigen::VectorXd>* distances,
                        std::list<Eigen::Vector3d>* unobserved_points,
                        std::list<octomap::OcTreeKey>* keys);

  /// \brief Get changes with distances from base voxel to closest comp voxel.
  double compareBackward(const KeyToDistMap& key_to_dist,
                         double max_dist,
                         std::list<Eigen::Vector3d>* observed_points,
                         std::list<Eigen::VectorXd>* distances,
                         std::list<Eigen::Vector3d>* unobserved_points);

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
