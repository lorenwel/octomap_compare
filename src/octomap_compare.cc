#include "octomap_compare/octomap_compare.h"

#include <list>
#include <limits>
#include <thread>

#include <dbscan/dbscan.h>
#include <pcl/conversions.h>

OctomapCompare::OctomapCompare(const std::string& base_file,
                               const CompareParams& params) :
    base_octree_(base_file), params_(params) {}

OctomapCompare::OctomapCompare(const std::shared_ptr<octomap::OcTree>& base_tree,
                               const CompareParams &params) :
    base_octree_(base_tree), params_(params) {}

template<typename ContainerType>
OctomapCompare::CompareResult OctomapCompare::compare(const ContainerType &compare_container) {
  CompareResult result;
  std::cout << "Comparing...\n";

  // Do ICP.
  getTransformFromICP(compare_container);

  std::list<Eigen::Vector3d> comp_observed_points;
  std::list<double> comp_distances;
  std::list<Eigen::Vector3d> unobserved_points;
  double max_dist = compareForward(compare_container,
                                   &comp_observed_points,
                                   &comp_distances,
                                   &unobserved_points);

  std::list<Eigen::Vector3d> base_observed_points;
  std::list<double> base_distances;
  max_dist = compareBackward<ContainerType>(compare_container,
                                            max_dist,
                                            &base_observed_points,
                                            &base_distances,
                                            &unobserved_points);

  // Fill result matrices.
  const size_t base_size = base_observed_points.size();
  const size_t comp_size = comp_observed_points.size();
  const size_t unobserved_size = unobserved_points.size();
  result.base_observed_points.resize(3, base_size);
  result.comp_observed_points.resize(3, comp_size);
  result.unobserved_points.resize(3, unobserved_size);
  result.base_distances.resize(base_size);
  result.comp_distances.resize(comp_size);
  result.max_dist = max_dist;
  for (size_t i = 0; i < base_size; ++i) {
    result.base_observed_points.col(i) = base_observed_points.front();
    base_observed_points.pop_front();
    result.base_distances(i) = base_distances.front();
    base_distances.pop_front();
  }
  for (size_t i = 0; i < comp_size; ++i) {
    result.comp_observed_points.col(i) = comp_observed_points.front();
    comp_observed_points.pop_front();
    result.comp_distances(i) = comp_distances.front();
    comp_distances.pop_front();
  }
  for (size_t i = 0; i < unobserved_size; ++i) {
    result.unobserved_points.col(i) = unobserved_points.front();
    unobserved_points.pop_front();
  }
  std::cout << "Compare cloud has " << comp_size << " observed voxels\n";
  std::cout << "Base cloud has " << base_size << " observed voxels\n";
  std::cout << "There are a combined " << unobserved_size << " unobserved voxels.\n";
  std::cout << "max dist: " << sqrt(max_dist) << "\n";

  return result;
}

template OctomapCompare::CompareResult OctomapCompare::compare(const OctomapContainer& compare_container);
template OctomapCompare::CompareResult OctomapCompare::compare(const PointCloudContainer& compare_container);

double OctomapCompare::compareForward(const ContainerBase& compare_container,
                                      std::list<Eigen::Vector3d>* observed_points,
                                      std::list<double>* distances,
                                      std::list<Eigen::Vector3d>* unobserved_points) {
  CHECK_NOTNULL(observed_points);
  CHECK_NOTNULL(distances);
  CHECK_NOTNULL(unobserved_points);
  double max_dist = 0;
  // Do the comparison.
  const unsigned int n_points = compare_container.Points().cols();
  for (unsigned int i = 0; i < n_points; ++i) {
    Eigen::Vector3d query_point(compare_container.Points().col(i));
    query_point = T_base_comp_ * query_point;
    octomap::OcTreeNode* base_node;  // Keep node to check occupancy.
    if (base_octree_.isObserved(query_point, &base_node)) {
      observed_points->push_back(query_point);
      // If base tree node is occupied distance is very low -> set 0.
      if (params_.k_nearest_neighbor == 1 &&
          base_node &&
          base_octree_->isNodeOccupied(base_node)) {
        distances->push_back(0);
      }
      else {
        OctomapContainer::KNNResult knn_result =
            base_octree_.findKNN(query_point, params_.k_nearest_neighbor);
        distances->push_back(getCompareDist(knn_result.distances, params_.distance_computation));
        if (distances->back() > max_dist) max_dist = distances->back();
      }
    }
    else {
      unobserved_points->push_back(query_point);
    }
  }
  return max_dist;
}

template<>
double OctomapCompare::compareBackward(
    const OctomapContainer& compare_container,
    double max_dist,
    std::list<Eigen::Vector3d>* observed_points,
    std::list<double>* distances,
    std::list<Eigen::Vector3d>* unobserved_points) {
  CHECK_NOTNULL(observed_points);
  CHECK_NOTNULL(distances);
  CHECK_NOTNULL(unobserved_points);
  // Compare base octree to comp octree.
  const unsigned int n_points = base_octree_.Points().cols();
  for (unsigned int i = 0; i < n_points; ++i) {
    Eigen::Vector3d query_point(base_octree_.Points().col(i));
    Eigen::Vector3d query_point_comp = T_comp_base_ * query_point;
    octomap::OcTreeNode* comp_node;
    if (compare_container.isObserved(query_point_comp, &comp_node)) {
      observed_points->push_back(query_point);
      // Check if comp node is also occupied.
      if (params_.k_nearest_neighbor == 1 &&
          comp_node &&
          compare_container->isNodeOccupied(comp_node)) {
        distances->push_back(0);
      }
      else {
        OctomapContainer::KNNResult knn_result =
            compare_container.findKNN(query_point_comp, params_.k_nearest_neighbor);
        distances->push_back(getCompareDist(knn_result.distances, params_.distance_computation));
        if (distances->back() > max_dist) max_dist = knn_result.distances(0);
      }
    }
    else {
      unobserved_points->push_back(query_point);
    }
  }
  return max_dist;
}

template<>
double OctomapCompare::compareBackward(
    const PointCloudContainer& compare_container,
    double max_dist,
    std::list<Eigen::Vector3d>* observed_points,
    std::list<double>* distances,
    std::list<Eigen::Vector3d>* unobserved_points) {
  CHECK_NOTNULL(observed_points);
  CHECK_NOTNULL(distances);
  CHECK_NOTNULL(unobserved_points);
  // Compare base octree to comp octree.
  const unsigned int n_points = base_octree_.Points().cols();
  for (unsigned int i = 0; i < n_points; ++i) {
    Eigen::Vector3d query_point(base_octree_.Points().col(i));
    Eigen::Vector3d query_point_comp = T_comp_base_ * query_point;

    OctomapContainer::KNNResult knn_result =
        compare_container.findKNN(query_point_comp, params_.k_nearest_neighbor);
    distances->push_back(getCompareDist(knn_result.distances, params_.distance_computation));
    if (distances->back() > max_dist) max_dist = knn_result.distances(0);
  }

  return max_dist;
}

void OctomapCompare::getTransformFromICP(const ContainerBase& compare_container) {
  // Get initial transform.
  Eigen::Matrix<double, 4, 4> T_initial_transpose(params_.initial_transform.data());
  Eigen::Matrix<double, 4, 4> T_initial;
  T_initial = T_initial_transpose.transpose();
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
  Eigen::Matrix<double, 4, 4> T_eigen;
  T_eigen = T;
  T_base_comp_.matrix() = T_eigen*T_initial;
  if (params_.perform_icp) std::cout << "Done ICPeeing, transform is\n"
                                     << T_base_comp_.matrix() << "\n";
  T_comp_base_ = T_base_comp_.inverse();
}

void OctomapCompare::getChanges(const CompareResult& result,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>* output_appear,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>* output_disappear,
                  Eigen::VectorXi* cluster_appear, 
                  Eigen::VectorXi* cluster_disappear) {
  CHECK_NOTNULL(output_appear);
  CHECK_NOTNULL(output_disappear);
  CHECK_NOTNULL(cluster_appear);
  CHECK_NOTNULL(cluster_disappear);
  // Apply threshold.
  std::list<unsigned int> comp_indices;
  const double thres = params_.distance_threshold * params_.distance_threshold;
  for (unsigned int i = 0; i < result.comp_distances.size(); ++i) {
    if (result.comp_distances(i) > thres) comp_indices.push_back(i);
  }
  std::list<unsigned int> base_indices;
  for (unsigned int i = 0; i < result.base_distances.size(); ++i) {
    if (result.base_distances(i) > thres) base_indices.push_back(i);
  }
  output_appear->resize(3, comp_indices.size());
  output_disappear->resize(3, base_indices.size());
  size_t counter = 0;
  for (auto&& index : comp_indices) {
    output_appear->col(counter++) = result.comp_observed_points.col(index);
  }
  counter = 0;
  for (auto&& index : base_indices) {
    output_disappear->col(counter++) = result.base_observed_points.col(index);
  }
  std::cout << output_appear->cols() << " appearing voxels left after distance thresholding\n";
  std::cout << output_disappear->cols() << " disappearing voxels left after distance thresholding\n";
  // Copy of transpose is necessary because dbscan works with a reference and has issues when
  // directly passing output->transpose().
  Eigen::MatrixXd transpose_appear(output_appear->transpose());
  Eigen::MatrixXd transpose_disappear(output_disappear->transpose());
  // DBSCAN filtering.
  Dbscan dbscan_appear(transpose_appear, params_.eps, params_.min_pts);
  Dbscan dbscan_disappear(transpose_disappear, params_.eps, params_.min_pts);
  dbscan_appear.cluster(cluster_appear);
  dbscan_disappear.cluster(cluster_disappear);

}

void OctomapCompare::compareResultToPointCloud(const CompareResult& result,
                                               pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud) {
  CHECK_NOTNULL(distance_point_cloud)->clear();
  distance_point_cloud->header.frame_id = "map";
  const double max_dist = sqrt(result.max_dist);
  for (unsigned int i = 0; i < result.base_observed_points.cols(); ++i) {
    pcl::RGB color = getColorFromDistance(sqrt(result.base_distances(i)), max_dist);
    pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
    pointrgb.x = result.base_observed_points(0,i);
    pointrgb.y = result.base_observed_points(1,i);
    pointrgb.z = result.base_observed_points(2,i);
    distance_point_cloud->push_back(pointrgb);
  }
  for (unsigned int i = 0; i < result.comp_observed_points.cols(); ++i) {
    pcl::RGB color = getColorFromDistance(sqrt(result.comp_distances(i)), max_dist);
    pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
    pointrgb.x = result.comp_observed_points(0,i);
    pointrgb.y = result.comp_observed_points(1,i);
    pointrgb.z = result.comp_observed_points(2,i);
    distance_point_cloud->push_back(pointrgb);
  }
  if (params_.show_unobserved_voxels) {
    for (unsigned int i = 0; i < result.unobserved_points.cols(); ++i) {
      pcl::RGB color = getColorFromDistance(-1.0);
      pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
      pointrgb.x = result.unobserved_points(0,i);
      pointrgb.y = result.unobserved_points(1,i);
      pointrgb.z = result.unobserved_points(2,i);
      distance_point_cloud->push_back(pointrgb);
    }
  }
}

pcl::RGB OctomapCompare::getColorFromDistance(const double& dist, const double& max_dist) {
  const double first = max_dist / 4;
  const double second = max_dist / 2;
  const double third = first + second;
  pcl::RGB color;
  if (dist > max_dist) {
    color.r = 255;
    color.g = 0;
    color.b = 0;
  }
  else if (dist > third) {
    color.r = 255;
    color.g = (max_dist - dist)/first*255;
    color.b = 0;
  }
  else if (dist > second) {
    color.r = (dist - second)/first*255;
    color.g = 255;
    color.b = 0;
  }
  else if (dist > first) {
    color.r = 0;
    color.g = 255;
    color.b = (second - dist)/first*255;
  }
  else if (dist >= 0) {
    color.r = 0;
    color.g = dist/first * 255;
    color.b = 255;
  }
  else {
    // Negative distance. Used to mark previously unobserved points.
    color.r = 180;
    color.g = 180;
    color.b = 180;
  }
  return color;
}
