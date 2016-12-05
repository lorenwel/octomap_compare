#include "octomap_compare/octomap_compare.h"

#include <fstream>
#include <list>
#include <limits>
#include <thread>

#include <dbscan/dbscan.h>
#include <hdbscan.h>
#include <pcl/conversions.h>

OctomapCompare::OctomapCompare(const std::string& base_file,
                               const CompareParams& params) :
    base_octree_(base_file), params_(params) {}

OctomapCompare::OctomapCompare(const std::shared_ptr<octomap::OcTree>& base_tree,
                               const CompareParams &params) :
    base_octree_(base_tree), params_(params) {}

void OctomapCompare::Reset() {
  base_index_to_distances_.clear();
  base_index_to_cluster_.clear();
  base_unobserved_points_.clear();

  comp_index_to_distances_.clear();
  comp_index_to_cluster_.clear();
  comp_unobserved_points_.clear();

  base_max_dist_ = 0;
  comp_max_dist_ = 0;
}

void OctomapCompare::compare(PointCloudContainer &compare_container,
                             const Eigen::Matrix<double, 4, 4>& T_initial) {
  compare_container_ = &compare_container;
  std::cout << "Comparing...\n";
  Reset();

  // Do ICP.
  getTransformFromICP(compare_container, T_initial);  

  // Compare backwards first so we fill its spherical point matrix.
  std::cout << "Comparing backward...\n";
  base_max_dist_ = compareBackward(compare_container);

  std::cout << "Comparing forward...\n";
  comp_max_dist_ = compareForward(compare_container);

  // Apply threshold and cluster.
  computeChanges(compare_container);

  std::cout << "Compare cloud has " << comp_index_to_distances_.size() << " observed voxels\n";
  std::cout << "Base cloud has " << base_index_to_distances_.size() << " observed voxels\n";
  std::cout << "There are a combined "
            << base_unobserved_points_.size() + comp_unobserved_points_.size()
            <<" unobserved voxels.\n";
  std::cout << "base max dist: " << sqrt(base_max_dist_) <<
               " comp max dist: " << sqrt(comp_max_dist_) << "\n";
}

double OctomapCompare::compareForward(PointCloudContainer& compare_container) {
  double max_dist = 0;
  octomap::OcTreeNode* base_node;  // Keep node to check occupancy.
  // Do the comparison.
  const unsigned int n_points = compare_container.Points().cols();
  for (unsigned int i = 0; i < n_points; ++i) {
    const Eigen::Vector3d point(compare_container.Points().col(i));
    const Eigen::Vector3d query_point = T_base_comp_ * point;
    compare_container.TransformedPoints().col(i) = query_point;
    if (point.squaredNorm() > 2.25 && // TODO: Make nifti laser model work and take this out.
        base_octree_.isObserved(query_point, &base_node)) {
      // If base tree node is occupied distance is very low -> set 0.
      if (params_.k_nearest_neighbor == 1 &&
          base_node &&
          base_octree_->isNodeOccupied(base_node)) {
        comp_index_to_distances_.push_back(std::make_pair(i, 0));
      }
      else {
        const SphericalVector spherical = compare_container.SphericalPoints().col(i);
        OctomapContainer::KNNResult knn_result =
            base_octree_.findKNN(spherical, params_.k_nearest_neighbor);
        const double cur_dist = getCompareDist(knn_result.distances, params_.distance_computation);
        comp_index_to_distances_.push_back(std::make_pair(i, cur_dist));
        if (cur_dist > max_dist) max_dist = cur_dist;
      }
    }
    else {
      comp_unobserved_points_.push_back(i);
    }
  }
  return max_dist;
}

double OctomapCompare::compareBackward(
    const PointCloudContainer& compare_container) {
  double max_dist = 0;
  std::list<SphericalVector> base_spherical;
  SphericalVector spherical_point;
  // Compare base octree to comp octree.
  const unsigned int n_points = base_octree_.Points().cols();
  for (unsigned int i = 0; i < n_points; ++i) {
    const Eigen::Vector3d map_point(base_octree_.Points().col(i));
    const Eigen::Vector3d query_point(T_comp_base_ * map_point);
    if (compare_container.isApproxObserved(query_point, &spherical_point)) {
      base_spherical.push_back(spherical_point);
      if (compare_container.isObserved(spherical_point)) {
        OctomapContainer::KNNResult knn_result =
            compare_container.findKNN(spherical_point, params_.k_nearest_neighbor);
        const double cur_dist = getCompareDist(knn_result.distances, params_.distance_computation);
        base_index_to_distances_.push_back(std::make_pair(i, cur_dist));
        if (cur_dist > max_dist) max_dist = cur_dist;
      }
      else {
        base_index_to_distances_.push_back(std::make_pair(i, 0));
      }
    }
    else {
      base_unobserved_points_.push_back(i);
    }
  }
  base_octree_.setSpherical(base_spherical);

  return max_dist;
}

void OctomapCompare::getTransformFromICP(const ContainerBase& compare_container,
                                         const Eigen::Matrix<double, 4, 4>& T_initial) {
  // Get initial transform.
  std::cout << "Initial transform is\n" << T_initial << "\n";
  // Initialize ICP.
  PM::ICP icp;
  icp.setDefault();
  PM::DataPoints base_points = matrix3dEigenToPointMatcher(base_octree_.Points());
  PM::DataPoints comp_points = matrix3dEigenToPointMatcher(compare_container.Points());
  // Do ICP.
  PM::TransformationParameters T(T_initial);
  if (params_.perform_icp) T = icp(comp_points, base_points, T);
  // Get transform.
  T_base_comp_.matrix() = T;
  if (params_.perform_icp) std::cout << "Done ICPeeing, transform is\n"
                                     << T_base_comp_.matrix() << "\n";
  T_comp_base_ = T_base_comp_.inverse();
}

void OctomapCompare::computeChanges(const PointCloudContainer &compare_container) {
  // Apply threshold.
  const double thres = params_.distance_threshold * params_.distance_threshold;

  std::list<size_t> thres_base_indices;
  for (const auto& ind_dist : base_index_to_distances_) {
    if (ind_dist.second > thres) thres_base_indices.push_back(ind_dist.first);
  }

  std::list<size_t> thres_comp_indices;
  for (const auto& ind_dist : comp_index_to_distances_) {
    if (ind_dist.second > thres) thres_comp_indices.push_back(ind_dist.first);
  }

  Eigen::Matrix<double, Eigen::Dynamic, 3> appear_transpose(thres_comp_indices.size(), 3);
  Eigen::Matrix<double, Eigen::Dynamic, 3> disappear_transpose(thres_base_indices.size(), 3);
  size_t counter = 0;
  for (auto&& index : thres_comp_indices) {
    appear_transpose.row(counter++) = compare_container.Points().col(index).transpose();
  }
  std::cout << counter << " appearing voxels left after distance thresholding\n";
  counter = 0;
  for (auto&& index : thres_base_indices) {
    disappear_transpose.row(counter++) = base_octree_.Points().col(index).transpose();
  }
  std::cout << counter << " disappearing voxels left after distance thresholding\n";
  // Save thresholded indices.

  // Cluster.
  Eigen::VectorXi appear_cluster;
  Eigen::VectorXi disappear_cluster;
  cluster(appear_transpose, &appear_cluster);
  cluster(disappear_transpose, &disappear_cluster);

  counter = 0;
  for (const auto& index : thres_comp_indices) {
    comp_index_to_cluster_.push_back(std::make_pair(index, appear_cluster(counter++)));
  }
  counter = 0;
  for (const auto& index : thres_base_indices) {
    base_index_to_cluster_.push_back(std::make_pair(index, disappear_cluster(counter++)));
  }
}

void OctomapCompare::cluster(const Eigen::Matrix<double, Eigen::Dynamic, 3> points,
                             Eigen::VectorXi* indices) {
  if (params_.clustering_algorithm == "dbscan") {
    Dbscan dbscan_appear(points, params_.eps, params_.min_pts);
    dbscan_appear.cluster(indices);
  }
  else if (params_.clustering_algorithm == "hdbscan") {
    Hdbscan hdbscan(params_.min_pts);
    if (points.rows() > 0) hdbscan.cluster(points, indices);
  }
  else {
    std::cerr << "Unknown clustering algorithm: \"" << params_.clustering_algorithm << "\"\n";
  }
}

double findDist(const size_t& index, std::list<std::pair<size_t, double>>& ind_to_dist) {
  for (auto&& ind_dist : ind_to_dist) {
    if (ind_dist.first == index) return ind_dist.second;
  }
  LOG(FATAL) << "Did not find distance for index " << index << ".";
}

// This function is terribly inefficient DO NOT USE EXCEPT FOR DEBUGGING.
void OctomapCompare::saveClusterResultToFile(const std::string& filename,
                                             ClusterCentroidVector* cluster_centroids) {
  CHECK_NOTNULL(cluster_centroids)->clear();
  std::ofstream out_file;
  out_file.open(filename, std::ios::trunc);
  if (out_file.is_open()) {
    // Print labels.
    out_file << "x, y, z, phi, theta, r, distance, cluster\n";

    std::unordered_map<int, Eigen::Vector3d> cluster_to_centroid;
    std::unordered_map<int, unsigned int> cluster_to_n_points;

    for (const auto& ind_cluster : comp_index_to_cluster_) {
      const Eigen::Vector3d cartesian(compare_container_->Points().col(ind_cluster.first));
      const Eigen::Vector3d cartesian_base(T_base_comp_ * cartesian);
      const SphericalVector spherical(compare_container_->cartesianToSpherical(cartesian));
      const double dist = findDist(ind_cluster.first, comp_index_to_distances_);
      out_file << cartesian(0) << ", "
               << cartesian(1) << ", "
               << cartesian(2) << ", "
               << spherical(0) << ", "
               << spherical(1) << ", "
               << spherical(2) << ", "
               << dist << ", "
               << ind_cluster.second << "\n";

      const auto res = cluster_to_centroid.insert(
            std::make_pair(ind_cluster.second, cartesian_base));
      if (res.second) {
        // First point of that cluster.
        cluster_to_n_points[ind_cluster.second] = 1;
      }
      else {
        // Already points.
        const unsigned int n = cluster_to_n_points[ind_cluster.second]++;
        res.first->second = res.first->second * n/(n+1.0) + cartesian_base * 1.0/(n+1);
      }
    }
    for (const auto& ind_cluster : base_index_to_cluster_) {
      const Eigen::Vector3d cartesian_base(base_octree_.Points().col(ind_cluster.first));
      const Eigen::Vector3d cartesian(T_comp_base_ * cartesian_base);
      const SphericalVector spherical(compare_container_->cartesianToSpherical(cartesian));
      const double dist = findDist(ind_cluster.first, base_index_to_distances_);
      out_file << cartesian(0) << ", "
               << cartesian(1) << ", "
               << cartesian(2) << ", "
               << spherical(0) << ", "
               << spherical(1) << ", "
               << spherical(2) << ", "
               << dist << ", "
               << -ind_cluster.second << "\n";

      const auto res = cluster_to_centroid.insert(
            std::make_pair(ind_cluster.second, cartesian_base));
      if (res.second) {
        // First point of that cluster.
        cluster_to_n_points[-ind_cluster.second] = 1;
      }
      else {
        // Already points.
        const unsigned int n = cluster_to_n_points[-ind_cluster.second]++;
        res.first->second = res.first->second * n/(n+1.0) + cartesian_base * 1.0/(n+1);
      }
    }

    LOG(INFO) << "Wrote to file \"" << filename << "\".";
    cluster_centroids->insert(cluster_centroids->begin(),
                              cluster_to_centroid.begin(),
                              cluster_to_centroid.end());
  }
  else {
    LOG(ERROR) << "Could not open file \"" << filename << "\".";
  }
  out_file.close();
}

void OctomapCompare::getChanges(pcl::PointCloud<pcl::PointXYZRGB>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  std::srand(0);
  std::unordered_map<unsigned int, pcl::RGB> color_map_appear;
  std::unordered_map<unsigned int, pcl::RGB> color_map_disappear;
  std::unordered_set<int> cluster_indices;
  // Generate cluster to color map.
  pcl::RGB color;
  if (!params_.color_changes) {
    for (const auto& ind_cluster : comp_index_to_cluster_) {
      const auto success = cluster_indices.insert(ind_cluster.second);
      // Generate color if it is new cluster.
      if (success.second) {
        color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
        color_map_appear[ind_cluster.second] = color;
      }
    }
    cluster_indices.clear();
    for (const auto& ind_cluster : base_index_to_cluster_) {
      const auto success = cluster_indices.insert(ind_cluster.second);
      // Generate color if it is new cluster.
      if (success.second) {
        color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
        color_map_disappear[ind_cluster.second] = color;
      }
    }
  }
  // Add outlier color.
  if (params_.show_outliers) {
    color.r = 180; color.g = 180; color.b = 180;
    if (params_.color_changes) {
      color.g = 255;
      color_map_appear[0] = color;
      color.r = 255; color.g = 180;
      color_map_disappear[0] = color;
    }
    else {
      color_map_appear[0] = color;
      color_map_disappear[0] = color;
    }
  }
  // Build point cloud.
  pcl::PointXYZRGB point;
  Eigen::Vector3d eigen_point;
  for (const auto& ind_cluster : comp_index_to_cluster_) {
    if (params_.show_outliers || ind_cluster.second != 0) {
      if (params_.color_changes) {
        color.r = 0; color.g = 255; color.b = 0;
      }
      else {
        color = color_map_appear[ind_cluster.second];
      }
      eigen_point = T_base_comp_ * Eigen::Vector3d(
                                       compare_container_->Points().col(ind_cluster.first));
      applyColorToPoint(color, point);
      setXYZFromEigen(eigen_point, point);
      cloud->push_back(point);
    }
  }
  for (const auto& ind_cluster : base_index_to_cluster_) {
    if (params_.show_outliers || ind_cluster.second != 0) {
      if (params_.color_changes) {
        color.r = 255; color.g = 0; color.b = 0;
      }
      else {
        color = color_map_disappear[ind_cluster.second];
      }
      eigen_point = base_octree_.Points().col(ind_cluster.first);
      setXYZFromEigen(eigen_point, point);
      cloud->push_back(point);
    }
  }
  std::cout << cloud->size() << " points left after clustering\n";
}

void OctomapCompare::getDistanceHeatMap(pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud) {
  CHECK_NOTNULL(distance_point_cloud)->clear();
  distance_point_cloud->header.frame_id = "map";
  double max_dist;
  if (base_max_dist_ > comp_max_dist_) max_dist = sqrt(base_max_dist_);
  else max_dist = sqrt(comp_max_dist_);
  if (params_.max_vis_dist > 0) max_dist = params_.max_vis_dist;
  pcl::RGB color;
  pcl::PointXYZRGB pointrgb;
  for (const auto& ind_dist : base_index_to_distances_) {
    color = getColorFromDistance(sqrt(ind_dist.second), max_dist);
    applyColorToPoint(color, pointrgb);
    setXYZFromEigen(base_octree_.Points().col(ind_dist.first), pointrgb);
    distance_point_cloud->push_back(pointrgb);
  }
  for (const auto& ind_dist : comp_index_to_distances_) {
    color = getColorFromDistance(sqrt(ind_dist.second), max_dist);
    applyColorToPoint(color, pointrgb);
    setXYZFromEigen(compare_container_->TransformedPoints().col(ind_dist.first), pointrgb);
    distance_point_cloud->push_back(pointrgb);
  }
  if (params_.show_unobserved_voxels) {
    for (const auto& index : base_unobserved_points_) {
      color = getColorFromDistance(-1.0);
      applyColorToPoint(color, pointrgb);
      setXYZFromEigen(base_octree_.Points().col(index), pointrgb);
      distance_point_cloud->push_back(pointrgb);
    }
    for (const auto& index : comp_unobserved_points_) {
      color = getColorFromDistance(-1.0);
      applyColorToPoint(color, pointrgb);
      setXYZFromEigen(compare_container_->TransformedPoints().col(index), pointrgb);
      distance_point_cloud->push_back(pointrgb);
    }
  }
}


