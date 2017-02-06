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
    base_octree_(base_file), params_(params) {
  loadICPConfig();
}

OctomapCompare::OctomapCompare(const std::shared_ptr<octomap::OcTree>& base_tree,
                               const CompareParams &params) :
    base_octree_(base_tree), params_(params) {
  loadICPConfig();
}

void OctomapCompare::loadICPConfig() {
  // Load the ICP configurations.
  std::ifstream ifs_icp_configurations(params_.icp_configuration_file.c_str());
  if (ifs_icp_configurations.good()) {
    LOG(INFO) << "Loading ICP configurations from: " << params_.icp_configuration_file;
    icp_.loadFromYaml(ifs_icp_configurations);
  } else {
    LOG(WARNING) << "Could not open ICP configuration from: \""
                 << params_.icp_configuration_file
                 << "\" file. Using default configuration.";
    icp_.setDefault();
  }

  // Load the ICP input filters configurations.
  std::ifstream ifs_input_filters(params_.icp_input_filters_file.c_str());
  if (ifs_input_filters.good()) {
    LOG(INFO) << "Loading ICP input filters from: " << params_.icp_input_filters_file;
    input_filters_ = PM::DataPointsFilters(ifs_input_filters);
  } else {
    LOG(WARNING) << "Could not open ICP input filters configuration file.";
  }

  // Load the ICP input filters configurations.
  std::ifstream ifs_base_filters(params_.icp_base_filters_file.c_str());
  if (ifs_base_filters.good()) {
    LOG(INFO) << "Loading ICP base filters from: " << params_.icp_base_filters_file;
    base_filters_ = PM::DataPointsFilters(ifs_base_filters);
  } else {
    LOG(WARNING) << "Could not open ICP base filters configuration file.";
  }
}

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

// This is different from the algorithm in the master thesis because we already have mahalanobis
// transformed points. We can just mahalanobis transform p_corre and will get the same result.
double OctomapCompare::computeCorrectedDist(const Eigen::Vector3d& point,
                                            const Eigen::Vector3d& neighbor) {
  const Eigen::Vector3d p_diff = point - neighbor;
  const double d_octo = base_octree_->getResolution()/2;
  const Eigen::Vector3d p_corr =
      params_.mahalanobis_transform * Eigen::Vector3d(d_octo/point(2), d_octo/point(2), d_octo);
  const double p_diff_norm2 = p_diff.squaredNorm();
  const double scale = p_diff.cwiseAbs().cwiseProduct(p_corr).sum() / p_diff_norm2;
  const Eigen::Vector3d p_dist = p_diff*(1 - scale);
  return p_dist.norm();
}

double OctomapCompare::getCompareDist(const Eigen::Vector3d point,
                                      const Eigen::VectorXi &neighbor_indices,
                                      const Matrix3xDynamic& neighbor_source,
                                      const std::string &dist_metric) {
  const size_t n_neighbors = neighbor_indices.size();
  if (dist_metric == "max") {
    return computeCorrectedDist(point, neighbor_source.col(neighbor_indices(n_neighbors - 1)));
  }
  else if (dist_metric == "mean") {
    double dist = 0;
    for (size_t i = 0; i < n_neighbors; ++i) {
      dist += computeCorrectedDist(point, neighbor_source.col(neighbor_indices[i]));
    }
    dist /= n_neighbors;
    return dist;
  }
  else if (dist_metric == "min") {
    return computeCorrectedDist(point, neighbor_source.col(neighbor_indices(0)));
  }
  else {
    LOG(FATAL) << "Invalid distances metric \"" << dist_metric << "\".\n";
    return -1;
  }
}

void OctomapCompare::compare(PointCloudContainer &compare_container,
                             Eigen::Matrix<double, 4, 4>* T_initial) {
  compare_container_ = &compare_container;
  std::cout << "Comparing...\n";
  Reset();

  Timer icp_timer("compare/icp");
  // Do ICP.
  getTransformFromICP(compare_container, T_initial);  
  icp_timer.stop();

  // Compare backwards first so we fill its spherical point matrix.
  std::cout << "Comparing backward...\n";
  Timer back_timer("compare/backwards");
  base_max_dist_ = compareBackward(compare_container);
  back_timer.stop();

  std::cout << "Comparing forward...\n";
  Timer forward_timer("compare/forward");
  comp_max_dist_ = compareForward(compare_container);
  forward_timer.stop();

  // Apply threshold and cluster.
  Timer change_timer("compare/changes");
  computeChanges();
  change_timer.stop();

  std::cout << "Compare cloud has " << comp_index_to_distances_.size() << " observed voxels\n";
  std::cout << "Base cloud has " << base_index_to_distances_.size() << " observed voxels\n";
  std::cout << "There are a combined "
            << base_unobserved_points_.size() + comp_unobserved_points_.size()
            << " unobserved voxels.\n";
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
        comp_index_to_distances_[i] = 0;
      }
      else {
        const SphericalVector spherical_scaled = compare_container.SphericalPointsScaled().col(i);
        OctomapContainer::KNNResult knn_result =
            base_octree_.findKNN(spherical_scaled, params_.k_nearest_neighbor);
        const double cur_dist = getCompareDist(spherical_scaled,
                                               knn_result.indices,
                                               base_octree_.SphericalPointsScaled(),
                                               params_.distance_computation);
        comp_index_to_distances_[i] = cur_dist;
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
  std::list<Eigen::Vector3d> comp_points;
  std::list<SphericalPoint> base_spherical;
  std::list<SphericalPoint> base_spherical_scaled;
  SphericalPoint spherical_point;
  SphericalPoint spherical_point_scaled;
  // Compare base octree to comp octree.
  const unsigned int n_points = base_octree_.Points().cols();
  const double octomap_resolution = base_octree_->getResolution()/2;
  size_t n_obs_points = 0;
  for (unsigned int i = 0; i < n_points; ++i) {
    const Eigen::Vector3d map_point(base_octree_.Points().col(i));
    const Eigen::Vector3d query_point(T_comp_base_ * map_point);
    if (compare_container.isApproxObserved(query_point,
                                           &spherical_point,
                                           &spherical_point_scaled)) {
      comp_points.push_back(query_point);
      base_spherical_scaled.push_back(spherical_point_scaled);
      base_spherical.push_back(spherical_point);
      base_ind_conv_[i] = n_obs_points++;
      if (compare_container.isObserved(spherical_point, octomap_resolution)) {
        OctomapContainer::KNNResult knn_result =
            compare_container.findKNN(spherical_point_scaled, params_.k_nearest_neighbor);
        const double cur_dist = getCompareDist(spherical_point_scaled,
                                               knn_result.indices,
                                               compare_container.SphericalPointsScaled(),
                                               params_.distance_computation);
        base_index_to_distances_[i] = cur_dist;
        if (cur_dist > max_dist) max_dist = cur_dist;
      }
      else {
        base_index_to_distances_[i] = 0;
      }
    }
    else {
      base_unobserved_points_.push_back(i);
    }
  }
  base_octree_.setTransformedPoints(comp_points);
  base_octree_.setSphericalScaled(base_spherical_scaled);
  base_octree_.setSpherical(base_spherical);

  return max_dist;
}

void cylindricalFilter(const Matrix3xDynamic& points,
                       const Eigen::Matrix<double, 4, 4>& T_initial,
                       const double& d,
                       Matrix3xDynamic* filtered) {
  CHECK_NOTNULL(filtered);
  const double d2 = d*d;
  const Eigen::Vector3d translation = T_initial.block<3,1>(0,3);
  std::list<size_t> indices;
  const size_t n_points = points.cols();
  for (size_t i = 0; i < n_points; ++i) {
    if (pow(points.col(i)(0) - translation(0), 2) +
        pow(points.col(i)(1) - translation(1), 2) +
        pow(points.col(i)(2) - translation(2), 2) < d2) {
      indices.push_back(i);
    }
  }
  const size_t n_filtered = indices.size();
  std::cout << "Cylindrical filter reduced from " << n_points << " to "
                                                  << n_filtered << " points\n";
  filtered->resize(3, n_filtered);
  size_t counter = 0;
  for (const auto& ind: indices) {
    filtered->col(counter++) = points.col(ind);
  }
}

void OctomapCompare::getTransformFromICP(const ContainerBase& compare_container,
                                         Eigen::Matrix<double, 4, 4>* T_initial) {
  // Get initial transform.
  std::cout << "Initial transform is\n" << *T_initial << "\n";
  Matrix3xDynamic filtered;
  cylindricalFilter(base_octree_.Points(), *T_initial, 25, &filtered);
  PM::DataPoints base_points = matrix3dEigenToPointMatcher(filtered);
//  PM::DataPoints base_points = matrix3dEigenToPointMatcher(base_octree_.Points());
  PM::DataPoints comp_points = matrix3dEigenToPointMatcher(compare_container.Points());
//  base_filters_.apply(base_points);
//  input_filters_.apply(comp_points);
  // Do ICP.
  PM::TransformationParameters T(*T_initial);
  if (params_.perform_icp) T = icp_(comp_points, base_points, T);
  // Get transform.
  T_base_comp_.matrix() = T;
  *T_initial = T;
  if (params_.perform_icp) std::cout << "Done ICPeeing, transform is\n"
                                     << T_base_comp_.matrix() << "\n";
  T_comp_base_ = T_base_comp_.inverse();
}

void OctomapCompare::getClusterMatrices(
    Eigen::Matrix<double, Eigen::Dynamic, 3>* appear_transpose,
    Eigen::Matrix<double, Eigen::Dynamic, 3>* disappear_transpose,
    std::list<size_t>* comp_indices,
    std::list<size_t>* base_indices) {
  CHECK_NOTNULL(comp_indices)->clear();
  CHECK_NOTNULL(base_indices)->clear();
  // Do comp points.
  std::list<size_t> thres_comp_indices;
  for (const auto& ind_dist : comp_index_to_distances_) {
    if (ind_dist.second > params_.distance_threshold) thres_comp_indices.push_back(ind_dist.first);
  }
  *comp_indices = thres_comp_indices;
  appear_transpose->resize(thres_comp_indices.size(), 3);
  size_t counter = 0u;
  if (params_.clustering_space == "cartesian") {
    for (auto&& index : thres_comp_indices) {
      appear_transpose->row(counter++) = compare_container_->Points().col(index).transpose();
    }
  }
  else if (params_.clustering_space == "spherical") {
    for (auto&& index : thres_comp_indices) {
      appear_transpose->row(counter++) =
          compare_container_->SphericalPoints().col(index).transpose();
    }
  }
  else if (params_.clustering_space == "spherical_scaled") {
    for (auto&& index : thres_comp_indices) {
      appear_transpose->row(counter++) =
          compare_container_->SphericalPointsScaled().col(index).transpose();
    }
  }
  else {
    LOG(FATAL) << "Unknown clustering space \"" << params_.clustering_space << "\".";
  }
  std::cout << counter << " appearing voxels left after distance thresholding\n";

  // Do base points.
  std::list<size_t> thres_base_indices;
  counter = 0u;
  if (params_.clustering_space == "cartesian") {
    for (const auto& ind_dist: base_index_to_distances_) {
      if (ind_dist.second > params_.distance_threshold) {
        thres_base_indices.push_back(ind_dist.first);
        base_indices->push_back(ind_dist.first);
      }
    }
  }
  else {  // Can only be one of the two spherical
    for (const auto& ind_dist: base_index_to_distances_) {
      if (ind_dist.second > params_.distance_threshold) {
        thres_base_indices.push_back(counter);
        base_indices->push_back(ind_dist.first);
      }
      ++counter;
    }
  }
  disappear_transpose->resize(thres_base_indices.size(), 3);
  counter = 0u;
  if (params_.clustering_space == "cartesian") {
    for (auto&& index : thres_base_indices) {
      disappear_transpose->row(counter++) = base_octree_.Points().col(index).transpose();
    }
  }
  else if (params_.clustering_space == "spherical") {
    for (auto&& index : thres_base_indices) {
      disappear_transpose->row(counter++) =
          base_octree_.SphericalPoints().col(base_ind_conv_[index]).transpose();
    }
  }
  else if (params_.clustering_space == "spherical_scaled") {
    for (auto&& index : thres_base_indices) {
      disappear_transpose->row(counter++) =
          base_octree_.SphericalPointsScaled().col(base_ind_conv_[index]).transpose();
    }
  }
  std::cout << counter << " disappearing voxels left after distance thresholding\n";
}

inline bool IndClusterCompare(const std::pair<size_t, int>& cmp1,
                              const std::pair<size_t, int>& cmp2) {
  return abs(cmp1.second) > abs(cmp2.second);
}

void OctomapCompare::computeChanges() {
  Eigen::Matrix<double, Eigen::Dynamic, 3> appear_transpose;
  Eigen::Matrix<double, Eigen::Dynamic, 3> disappear_transpose;
  std::list<size_t> comp_indices;
  std::list<size_t> base_indices;

  getClusterMatrices(&appear_transpose, &disappear_transpose, &comp_indices, &base_indices);

  // Cluster.
  Eigen::VectorXi appear_cluster;
  Eigen::VectorXi disappear_cluster;
  cluster(appear_transpose, &appear_cluster);
  cluster(disappear_transpose, &disappear_cluster);
  disappear_cluster *= -1;

  // Save cluster result.
  size_t counter = 0u;
  for (const auto& index : comp_indices) {
    comp_index_to_cluster_.push_back(std::make_pair(index, appear_cluster(counter++)));
  }
  counter = 0u;
  for (const auto& index : base_indices) {
    base_index_to_cluster_.push_back(std::make_pair(index, disappear_cluster(counter++)));
  }
  // Sort result.
  comp_index_to_cluster_.sort(IndClusterCompare);
  base_index_to_cluster_.sort(IndClusterCompare);
}

void OctomapCompare::cluster(const Eigen::Matrix<double, Eigen::Dynamic, 3>& points,
                             Eigen::VectorXi* indices) {
  if (params_.clustering_algorithm == "dbscan") {
    Dbscan::Cluster(points, params_.eps, params_.min_pts, indices);
  }
  else if (params_.clustering_algorithm == "hdbscan") {
    Hdbscan hdbscan(params_.min_pts);
    if (points.rows() > 0u) hdbscan.cluster(points, indices);
  }
  else {
    std::cerr << "Unknown clustering algorithm: \"" << params_.clustering_algorithm << "\"\n";
  }
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
      const double dist = comp_index_to_distances_[ind_cluster.first];
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
      const double dist = base_index_to_distances_[ind_cluster.first];
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

void OctomapCompare::getChangeCandidates(pcl::PointCloud<pcl::PointXYZRGB>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  std::srand(0);
  std::unordered_map<int, pcl::RGB> color_map_appear;
  std::unordered_map<int, pcl::RGB> color_map_disappear;
  std::unordered_set<int> cluster_indices;
  // Generate cluster to color map.
  pcl::RGB color;
  color.r = 0; color.g = 255; color.b = 0;
  for (const auto& ind_cluster : comp_index_to_cluster_) {
    const auto success = cluster_indices.insert(ind_cluster.second);
    // Generate color if it is new cluster.
    if (success.second) {
      if (!params_.color_changes) {
        color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
      }
      color_map_appear[ind_cluster.second] = color;
    }
  }
  cluster_indices.clear();
  color.r = 255; color.g = 0; color.b = 0;
  for (const auto& ind_cluster: base_index_to_cluster_) {
    const auto success = cluster_indices.insert(ind_cluster.second);
    // Generate color if it is new cluster.
    if (success.second) {
      if (!params_.color_changes) {
        color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
      }
      color_map_disappear[ind_cluster.second] = color;
    }
  }
  // Add outlier color.
  if (params_.show_outliers) {
    color.r = 127; color.g = 127; color.b = 127;
    if (params_.color_changes) {
      color.g = 180;
      color_map_appear[0] = color;
      color.r = 180; color.g = 127;
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
  for (const auto& ind_cluster: comp_index_to_cluster_) {
    if (params_.show_outliers || ind_cluster.second != 0) {
      color = color_map_appear[ind_cluster.second];
      eigen_point = T_base_comp_ * Eigen::Vector3d(
                                       compare_container_->Points().col(ind_cluster.first));
      applyColorToPoint(color, point);
      setXYZFromEigen(eigen_point, point);
      cloud->push_back(point);
    }
  }
  for (const auto& ind_cluster : base_index_to_cluster_) {
    if (params_.show_outliers || ind_cluster.second != 0) {
      color = color_map_disappear[ind_cluster.second];
      eigen_point = base_octree_.Points().col(ind_cluster.first);
      applyColorToPoint(color, point);
      setXYZFromEigen(eigen_point, point);
      cloud->push_back(point);
    }
  }
  std::cout << cloud->size() << " points left after clustering\n";
}

void OctomapCompare::getDistanceHeatMap(pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud,
                                        pcl::PointCloud<pcl::PointXYZRGB>* threshold_point_cloud) {
  CHECK_NOTNULL(distance_point_cloud)->clear();
  CHECK_NOTNULL(threshold_point_cloud)->clear();
  distance_point_cloud->header.frame_id = "map";
  threshold_point_cloud->header.frame_id = "map";
  double max_dist;
  if (base_max_dist_ > comp_max_dist_) max_dist = base_max_dist_;
  else max_dist = comp_max_dist_;
  if (params_.max_vis_dist > 0) max_dist = params_.max_vis_dist;
  pcl::RGB color;
  pcl::PointXYZRGB pointrgb;
  for (const auto& ind_dist : base_index_to_distances_) {
    color = getColorFromDistance(ind_dist.second, max_dist);
    applyColorToPoint(color, pointrgb);
    setXYZFromEigen(base_octree_.Points().col(ind_dist.first), pointrgb);
    distance_point_cloud->push_back(pointrgb);
    if (ind_dist.second > params_.distance_threshold) threshold_point_cloud->push_back(pointrgb);
  }
  for (const auto& ind_dist : comp_index_to_distances_) {
    color = getColorFromDistance(ind_dist.second, max_dist);
    applyColorToPoint(color, pointrgb);
    setXYZFromEigen(compare_container_->TransformedPoints().col(ind_dist.first), pointrgb);
    distance_point_cloud->push_back(pointrgb);
    if (ind_dist.second > params_.distance_threshold) threshold_point_cloud->push_back(pointrgb);
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

void OctomapCompare::getClusters(std::vector<Cluster>* clusters) {
  CHECK_NOTNULL(clusters)->clear();
  ClusterPoint cluster_point;
  int cur_cluster = std::numeric_limits<int>::max();

  CHECK_NE(base_index_to_cluster_.front().second, -cur_cluster)
      << "More clusters than int::max() can hold";
  CHECK_LE(base_index_to_cluster_.front().second, 0)
      << "Found positive index disappearing cluster";
  // base_index_to_cluster_ is ordered descending by absolute of cluster index.
  for (const auto& ind_cluster: base_index_to_cluster_) {
    // Push back new cluster if we encounter new one
    if (ind_cluster.second != cur_cluster) {
      cur_cluster = ind_cluster.second;
      if (cur_cluster == 0) break; // Finish if we arrived at outlier cluster.
      clusters->push_back(Cluster(cur_cluster));  // Push back new cluster.
    }
    // Add point.
    cluster_point.cartesian =
        base_octree_.TransformedPoints().col(base_ind_conv_[ind_cluster.first]);
    cluster_point.spherical =
        base_octree_.SphericalPoints().col(base_ind_conv_[ind_cluster.first]);
    cluster_point.distance = base_index_to_distances_[ind_cluster.first];
    clusters->back().points.push_back(cluster_point);
  }

  cur_cluster = std::numeric_limits<int>::max();
  CHECK_NE(comp_index_to_cluster_.front().second, cur_cluster)
      << "More clusters than int::max() can hold";
  CHECK_GE(comp_index_to_cluster_.front().second, 0) << "Found negative index appearing cluster";
  // comp_index_to_cluster_ is ordered descending by cluster index.
  for (const auto& ind_cluster: comp_index_to_cluster_) {
    // Push back new cluster if we encounter new one
    if (ind_cluster.second != cur_cluster) {
      cur_cluster = ind_cluster.second;
      if (cur_cluster == 0) break; // Finish if we arrived at outlier cluster.
      clusters->push_back(Cluster(cur_cluster));  // Push back new cluster.
    }
    // Add point.
    cluster_point.cartesian = compare_container_->Points().col(ind_cluster.first);
    cluster_point.spherical = compare_container_->SphericalPoints().col(ind_cluster.first);
    cluster_point.distance = comp_index_to_distances_[ind_cluster.first];
    clusters->back().points.push_back(cluster_point);
  }
}
