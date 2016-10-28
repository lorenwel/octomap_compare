#ifndef OCTOMAP_COMPARE_H_
#define OCTOMAP_COMPARE_H_

#include <memory>
#include <unordered_map>

#include <glog/logging.h>
#include <nabo/nabo.h>
#include <octomap/OcTree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "octomap_compare/octomap_compare_utils.h"
#include "octomap_compare/octomap_container.h"

class OctomapCompare {

public:
  typedef std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash> KeyToDistMap;

  struct CompareResult {
    Eigen::Matrix<double, 3, Eigen::Dynamic> base_observed_points;
    Eigen::Matrix<double, 3, Eigen::Dynamic> comp_observed_points;
    Eigen::VectorXd base_distances;
    Eigen::VectorXd comp_distances;
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
      show_unobserved_voxels(true), 
      distance_computation("max"), 
      color_changes(true) {}

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
    // Distance metric to use when computing distance from knn search.
    std::string distance_computation;
    // Color changes point cloud based on appearing/disappearing.
    bool color_changes;
  };

private:

  // Octomap considered to be the base map for comparison.
  std::shared_ptr<OctomapContainer> base_octree_;

  // Octomap which will be compared to the base octomap.
  std::shared_ptr<OctomapContainer> comp_octree_;

  // Transformation from comp_octree_ frame to base_octree_ frame.
  Eigen::Affine3d T_base_comp_;
  Eigen::Affine3d T_comp_base_;

  // Parameters.
  CompareParams params_;

  /// \brief Gets point color based on distance for use in visualization.
  pcl::RGB getColorFromDistance(const double& dist, const double& max_dist = 1);

  /// \brief Get changes with distances from comp voxel to closest base voxel.
  double compareForward(std::list<Eigen::Vector3d>* observed_points,
                        std::list<double>* distances,
                        std::list<Eigen::Vector3d>* unobserved_points,
                        std::list<octomap::OcTreeKey>* keys);

  /// \brief Get changes with distances from base voxel to closest comp voxel.
  double compareBackward(const KeyToDistMap& key_to_dist,
                         double max_dist,
                         std::list<Eigen::Vector3d>* observed_points,
                         std::list<double>* distances,
                         std::list<Eigen::Vector3d>* unobserved_points);

  /// \brief Get transform to align octomaps.
  void getTransformFromICP();

public:
  /// \brief Constructor which reads octomaps from specified files.
  OctomapCompare(const std::string& base_file, const std::string& comp_file,
                 const CompareParams& params = CompareParams());

  OctomapCompare(const std::shared_ptr<octomap::OcTree>& base_tree,
                 const std::shared_ptr<octomap::OcTree>& comp_tree,
                 const CompareParams& params = CompareParams());

  /// \brief Compare both loaded point clouds and return colored point cloud.
  CompareResult compare();

  /// \brief Get changes between octomaps using the result from a comparison.
  void getChanges(const CompareResult& result,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>* output_appear,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>* output_disappear,
                  Eigen::VectorXi* cluster_appear, 
                  Eigen::VectorXi* cluster_disappear);

  /// \brief Get a point cloud visualizing the comparison result.
  void compareResultToPointCloud(const CompareResult& result,
                                 pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud);
};

#endif // OCTOMAP_COMPARE_H_
