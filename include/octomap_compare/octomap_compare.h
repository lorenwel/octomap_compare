#ifndef OCTOMAP_COMPARE_H_
#define OCTOMAP_COMPARE_H_

#include <memory>
#include <unordered_map>

#include <glog/logging.h>
#include <nabo/nabo.h>
#include <octomap/OcTree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/MarkerArray.h>

#include "octomap_compare/octomap_compare_utils.h"
#include "octomap_compare/container/octomap_container.h"
#include "octomap_compare/container/point_cloud_container.h"

class OctomapCompare {

public:
  typedef std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash> KeyToDistMap;

  struct CompareParams {
    CompareParams() :
      max_vis_dist(8),
      distance_threshold(0.1),
      eps(0.3),
      min_pts(10),
      k_nearest_neighbor(1),
      show_unobserved_voxels(true), 
      show_outliers(true),
      distance_computation("max"), 
      color_changes(true), 
      perform_icp(true) {}

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
    // Which clustering algorithm to use.
    std::string clustering_algorithm;
    // Color changes point cloud based on appearing/disappearing.
    bool color_changes;
    // Show outlier after clustering.
    bool show_outliers;
    // Perform ICP or not.
    bool perform_icp;
    // Standard deviation of laser.
    Eigen::Matrix3d std_dev;
    // Transform from sensor frame to spherical coordinate frame.
    Eigen::Matrix3d spherical_transform;
    // File for ICP configuration.
    std::string icp_configuration_file;
    // File for ICP input filters.
    std::string icp_input_filters_file;
    std::string icp_base_filters_file;
  };

private:

  // ICP object.
  PM::ICP icp_;
  PM::DataPointsFilters input_filters_;
  PM::DataPointsFilters base_filters_;

  // Octomap considered to be the base map for comparison.
  OctomapContainer base_octree_;

  // Transformation from comp_octree_ frame to base_octree_ frame.
  Eigen::Affine3d T_base_comp_;
  Eigen::Affine3d T_comp_base_;

  std::list<std::pair<size_t, double>> base_index_to_distances_;
  std::list<std::pair<size_t, int>> base_index_to_cluster_;
  std::list<size_t> base_unobserved_points_;

  std::list<std::pair<size_t, double>> comp_index_to_distances_;
  std::list<std::pair<size_t, int>> comp_index_to_cluster_;
  std::list<size_t> comp_unobserved_points_;

  double base_max_dist_;
  double comp_max_dist_;

  const PointCloudContainer* compare_container_;

  // Parameters.
  CompareParams params_;

  /// \brief Get changes with distances from comp voxel to closest base voxel.
  double compareForward(PointCloudContainer& compare_container);

  /// \brief Get changes with distances from base voxel to closest comp voxel.
//  template<typename ContainerType>
  double compareBackward(const PointCloudContainer& compare_container);

  /// \brief Get transform to align octomaps.
  void getTransformFromICP(const ContainerBase& compare_container,
                           const Eigen::Matrix<double, 4, 4>& T_initial);

  /// \brief Cluster the points.
  void cluster(const Eigen::Matrix<double, Eigen::Dynamic, 3> points, Eigen::VectorXi* indices);

  /// \brief Reset all member variables.
  void Reset();

  /// \brief Get changes between octomaps using the result from a comparison.
  void computeChanges(const PointCloudContainer &compare_container);

  /// \brief Load ICP config from file.
  void loadICPConfig();

  /// \brief Returns number of angular std deviations an octomap voxel occupies at certain distance.
  ///        Is only a sensible correction if std_dev of theta == std_dev of phi.
  inline double getDistanceCorrection(const SphericalPoint& point) {
    return base_octree_->getResolution() / point.r / params_.std_dev(0,0) / 2;
  }

public:
  /// \brief Constructor which reads octomaps from specified files.
  OctomapCompare(const std::string& base_file,
                 const CompareParams& params = CompareParams());

  OctomapCompare(const std::shared_ptr<octomap::OcTree>& base_tree,
                 const CompareParams& params = CompareParams());

  /// \brief Compare both loaded point clouds and return colored point cloud.
  void compare(PointCloudContainer& compare_container,
               const Eigen::Matrix<double, 4, 4>& T_initial =
                     Eigen::MatrixXd::Identity(4, 4));

  /// \brief Save result of clustering to file for evaluation in MATLAB.
  void saveClusterResultToFile(const std::string& filename, ClusterCentroidVector* cluster_centroids);

  /// \brief Get the changes as point cloud. Color indicates cluster.
  void getChanges(pcl::PointCloud<pcl::PointXYZRGB>* cloud);

  /// \brief Get a point cloud visualizing the comparison result.
  void getDistanceHeatMap(pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud);
};

#endif // OCTOMAP_COMPARE_H_
