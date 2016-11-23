#include "octomap_compare/octomap_compare.h"

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

OctomapCompare::CompareResult OctomapCompare::compare(const PointCloudContainer &compare_container,
                                                      const Eigen::Matrix<double, 4, 4>& T_initial,
                                                      visualization_msgs::MarkerArray* array) {
  CompareResult result;
  std::cout << "Comparing...\n";

  // Do ICP.
  getTransformFromICP(compare_container, T_initial);

  std::list<Eigen::Vector3d> base_observed_points;
  std::list<Eigen::Vector3d> unobserved_points;

  // Get Octomap spherical points.
  const size_t n_base_points = base_octree_.Points().cols();
  std::vector<SphericalVector> base_spherical;
  for (unsigned int i = 0; i < n_base_points; ++i) {
    const Eigen::Vector3d map_point(base_octree_.Points().col(i));
    const Eigen::Vector3d query_point(T_comp_base_ * map_point);
    SphericalVector spherical_point;
    if (compare_container.isObserved(query_point, &spherical_point)) {
      base_spherical.push_back(spherical_point);
      base_observed_points.push_back(map_point);
    }
    else {
      unobserved_points.push_back(map_point);
    }
  }
  Eigen::MatrixXd base_spherical_eigen(3, base_spherical.size());
  const size_t n_spherical = base_spherical.size();
  for (unsigned int i = 0; i < n_spherical; ++i) {
    base_spherical_eigen.col(i) = base_spherical[i];
  }
  base_octree_.setSpherical(base_spherical_eigen);

  std::list<Eigen::Vector3d> comp_observed_points;
  std::list<double> comp_distances;
  std::cout << "Comparing forward...\n";
  double max_dist = compareForward(compare_container,
                                   &comp_observed_points,
                                   &comp_distances,
                                   &unobserved_points);

  std::list<double> base_distances;
  std::cout << "Comparing backward...\n";
  max_dist = compareBackward(compare_container,
                                            max_dist,
                                            &base_observed_points,
                                            &base_distances,
                                            &unobserved_points);
  std::cout << "Filling matrices\n";
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

  static const unsigned int n_ellipses = 50;
  if (array != NULL) {
    std::srand(std::time(0));
    for (size_t i = 0; i < n_ellipses; ++i) {
      const unsigned int ind = std::rand()%result.comp_observed_points.cols();
      if ((T_comp_base_ * result.comp_observed_points.col(ind)).squaredNorm() < 4) {
        --i;
      }
      else {
        array->markers.push_back(getEllipsis(result.comp_observed_points.col(ind), i));
        array->markers.push_back(getText(result.comp_observed_points.col(ind), i + n_ellipses));
      }
    }
  }

  return result;
}

visualization_msgs::Marker OctomapCompare::getText(const Eigen::Vector3d& point_base, const unsigned int& id) {
  Eigen::Vector3d point (params_.spherical_transform * T_comp_base_ * point_base);
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.id = id;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.color.a = 0.6; // Don't forget to set the alpha!
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;
  marker.scale.z = 0.1;
  const double d = point.norm();
  const double phi = atan2(point(1), point(0))/M_PI*180;
  const double theta = atan2(sqrt(point(0) * point(0) + point(1) * point(1)), point(2))/M_PI*180;
  marker.text = "Phi: " + std::to_string(phi) + " Theta: " + std::to_string(theta) + " d: " + std::to_string(d) + "\n";
  marker.pose.position.x = point_base.x();
  marker.pose.position.y = point_base.y();
  marker.pose.position.z = point_base.z();
  return marker;
}

visualization_msgs::Marker OctomapCompare::getEllipsis(const Eigen::Vector3d& point_base, const unsigned int& id) {
  Eigen::Vector3d point(params_.spherical_transform * T_comp_base_ * point_base);

  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.id = id;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.color.a = 0.6; // Don't forget to set the alpha!
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;

  const double d = point.norm();
  const double phi = atan2(point(1), point(0));
  const double theta = atan2(sqrt(point(0) * point(0) + point(1) * point(1)), point(2));

  static const double dev_d = 0.03;
  static const double dev_phi = 0.035;
  static const double dev_theta = 0.004;
  Eigen::AngleAxisd first(-phi, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd second(-theta, Eigen::Vector3d::UnitY());

  Eigen::Vector3d scales(
      d * dev_theta + 0.075/2,
      d * sin(theta) * dev_phi + 0.075/2,
      dev_d + 0.075/2);

  scales = second * first * scales;

  scales = T_base_comp_.rotation() * scales;

  marker.scale.x = std::fabs(scales.x());
  marker.scale.y = std::fabs(scales.y());
  marker.scale.z = std::fabs(scales.z());

  marker.pose.position.x = point_base.x();
  marker.pose.position.y = point_base.y();
  marker.pose.position.z = point_base.z();
  return marker;
}

//template OctomapCompare::CompareResult OctomapCompare::compare(
//    const OctomapContainer& compare_container,
//    const Eigen::Matrix<double, 4, 4>& T_initial,
//visualization_msgs::MarkerArray* array);
//template OctomapCompare::CompareResult OctomapCompare::compare(
//    const PointCloudContainer& compare_container,
//    const Eigen::Matrix<double, 4, 4>& T_initial,
//visualization_msgs::MarkerArray* array);

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
    const Eigen::Vector3d point(compare_container.Points().col(i));
    const Eigen::Vector3d query_point = T_base_comp_ * point;
    octomap::OcTreeNode* base_node;  // Keep node to check occupancy.
    if (point.squaredNorm() > 2.25 &&
        base_octree_.isObserved(query_point, &base_node)) {
      observed_points->push_back(query_point);
      // If base tree node is occupied distance is very low -> set 0.
      if (params_.k_nearest_neighbor == 1 &&
          base_node &&
          base_octree_->isNodeOccupied(base_node)) {
        distances->push_back(0);
      }
      else {
        SphericalVector spherical = compare_container.SphericalPoints().col(i);
        OctomapContainer::KNNResult knn_result =
            base_octree_.findKNN(spherical, params_.k_nearest_neighbor);
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

//template<>
//double OctomapCompare::compareBackward(
//    const OctomapContainer& compare_container,
//    double max_dist,
//    std::list<Eigen::Vector3d>* observed_points,
//    std::list<double>* distances,
//    std::list<Eigen::Vector3d>* unobserved_points) {
//  CHECK_NOTNULL(observed_points);
//  CHECK_NOTNULL(distances);
//  CHECK_NOTNULL(unobserved_points);
//  // Compare base octree to comp octree.
//  const unsigned int n_points = base_octree_.Points().cols();
//  for (unsigned int i = 0; i < n_points; ++i) {
//    Eigen::Vector3d query_point(base_octree_.Points().col(i));
//    Eigen::Vector3d query_point_comp = T_comp_base_ * query_point;
//    octomap::OcTreeNode* comp_node;
//    if (compare_container.isObserved(query_point_comp, &comp_node)) {
//      observed_points->push_back(query_point);
//      // Check if comp node is also occupied.
//      if (params_.k_nearest_neighbor == 1 &&
//          comp_node &&
//          compare_container->isNodeOccupied(comp_node)) {
//        distances->push_back(0);
//      }
//      else {
//        OctomapContainer::KNNResult knn_result =
//            compare_container.findKNN(query_point_comp, params_.k_nearest_neighbor);
//        distances->push_back(getCompareDist(knn_result.distances, params_.distance_computation));
//        if (distances->back() > max_dist) max_dist = knn_result.distances(0);
//      }
//    }
//    else {
//      unobserved_points->push_back(query_point);
//    }
//  }
//  return max_dist;
//}

//template<>
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
  const unsigned int n_points = base_octree_.SphericalPoints().cols();
  for (unsigned int i = 0; i < n_points; ++i) {
    Eigen::Vector3d query_point(base_octree_.SphericalPoints().col(i));

    OctomapContainer::KNNResult knn_result =
        compare_container.findKNN(query_point, params_.k_nearest_neighbor);
    distances->push_back(0);
//    distances->push_back(getCompareDist(knn_result.distances, params_.distance_computation));
//    if (distances->back() > max_dist) max_dist = knn_result.distances(0);
  }

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
  if (params_.clustering_algorithm == "dbscan") {
    Dbscan dbscan_appear(transpose_appear, params_.eps, params_.min_pts);
    Dbscan dbscan_disappear(transpose_disappear, params_.eps, params_.min_pts);
    dbscan_appear.cluster(cluster_appear);
    dbscan_disappear.cluster(cluster_disappear);
  }
  else if (params_.clustering_algorithm == "hdbscan") {
    Hdbscan hdbscan(params_.min_pts);
    if (transpose_appear.rows() > 0) hdbscan.cluster(transpose_appear, cluster_appear);
    if (transpose_disappear.rows() > 0) hdbscan.cluster(transpose_disappear, cluster_disappear);
  }
  else {
    std::cerr << "Unknown clustering algorithm: \"" << params_.clustering_algorithm << "\"\n";
  }
}

void OctomapCompare::compareResultToPointCloud(const CompareResult& result,
                                               pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud) {
  CHECK_NOTNULL(distance_point_cloud)->clear();
  distance_point_cloud->header.frame_id = "map";
  double max_dist = sqrt(result.max_dist);
  if (params_.max_vis_dist > 0) max_dist = params_.max_vis_dist;
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


