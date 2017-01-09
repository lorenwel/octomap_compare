#include "octomap_compare/random_forest_classifier.h"

#include <algorithm>

template <typename T>
size_t writeVectorToOpenCvMatrix(const std::vector<T>& vec,
                                 cv::Mat* mat,
                                 const size_t& mat_start_ind = 0) {
  for (size_t i = 0; i < vec.size(); ++i) {
    mat->at<T>(0, i + mat_start_ind) = vec[i];
  }
  return mat_start_ind + vec.size();
}

size_t writeEigenVectorToOpenCvMatrix(const Eigen::VectorXd& vec,
                                      cv::Mat* mat,
                                      const size_t& mat_start_ind = 0) {
  for (size_t i = 0; i < vec.size(); ++i) {
    mat->at<double>(0, i + mat_start_ind) = vec(i);
  }
  return mat_start_ind + vec.size();
}

void FeatureExtractor::getClusterFeatures(const Cluster& cluster, cv::Mat* features) {
  CHECK_NOTNULL(features);
  const size_t n_points = cluster.points.size();
  CHECK_GT(n_points, 1) << "Cluster too small.";

  Eigen::MatrixXd cartesian_mat(3, n_points);
  Eigen::MatrixXd spherical_mat(3, n_points);
  std::vector<double> distances(n_points,0);
  double dist_mean = 0;

  // Fill matrices
  for (size_t i = 0; i < n_points; ++i) {
    cartesian_mat.col(i) = cluster.points[i].cartesian;
    spherical_mat.col(i) = cluster.points[i].spherical;
    distances[i] = cluster.points[i].distance;
    dist_mean += cluster.points[i].distance;
  }
  dist_mean /= n_points;

  // Get means.
  const Eigen::Vector3d cartesian_mean = cartesian_mat.rowwise().mean();
  const Eigen::Vector3d spherical_mean = spherical_mat.rowwise().mean();

  // Make matrices zero-mean for variance computation.
  cartesian_mat -= cartesian_mean;
  spherical_mat -= spherical_mean;

  // Sort, so that we can acces min, max, median.
  std::sort(distances.begin(), distances.end());

  // Tensors for eigen value computation.
  const Eigen::MatrixXd cartesian_tensor =  cartesian_mat * cartesian_mat.transpose();
  const Eigen::MatrixXd spherical_tensor =  spherical_mat * spherical_mat.transpose();

  // Compute eigen values. Cartesian eigen vectors needed for surface normal.
  const Eigen::EigenSolver<Eigen::MatrixXd> cartesian_solver(cartesian_tensor, true);
  const Eigen::EigenSolver<Eigen::MatrixXd> spherical_solver(spherical_tensor, false);

  const Eigen::Vector3d cartesian_eigenvalues = cartesian_solver.eigenvalues().real();
  const Eigen::Vector3d spherical_eigenvalues = spherical_solver.eigenvalues().real();
  const Eigen::Matrix3d cartesian_eigen_vectors = cartesian_solver.eigenvectors().real();

  // Compute sample variance.
  const Eigen::Vector3d cartesian_variance = cartesian_tensor.diagonal() / (n_points - 1);
  const Eigen::Vector3d spherical_variance = spherical_tensor.diagonal() / (n_points - 1);
  double dist_variance = 0;
  for (const auto& dist: distances) {
    const double dist_dif = dist - dist_mean;
    dist_variance += dist_dif * dist_dif;
  }
  dist_variance /= n_points - 1;

  // Get surface normal.
  size_t min_eig_ind, dummy;
  cartesian_eigenvalues.minCoeff(&min_eig_ind, &dummy);
  const Eigen::Vector3d surface_normal = cartesian_eigen_vectors.col(min_eig_ind);

  // Compute surface normal features.
  const double incident_angle =
      fabs(acos(surface_normal.transpose() * cartesian_mean.normalized()));
  const double surface_angle = atan2(sqrt(1 - surface_normal(2)), surface_normal(2));

  // Get Eigenvalue features.
  std::vector<double> eigenvalue_features;
  getEigenvalueFeatures(cartesian_eigenvalues, &eigenvalue_features);

  // Get histogram.
  std::vector<size_t> histogram;
  getHistogram(cluster, &histogram);

  // Get densities.
  const std::pair<double, double> densities = getDensities(cartesian_mat, surface_normal);
  const std::vector<double> density_vec =
      {densities.first/spherical_mean(2),
       densities.second/spherical_mean(2)}; // Normalize density with range.

  // Write features to matrix.
  features->at<double>(0,0) = n_points;
  features->at<double>(0,1) = distances.back() - distances.front(); // max dist diff
  features->at<double>(0,2) = dist_mean;
  features->at<double>(0,3) = dist_variance;
  features->at<double>(0,4) = distances.front(); // min_distance
  features->at<double>(0,5) = distances.back(); // max_distance
  features->at<double>(0,6) = distances[distances.size()/2];  // median_distance
  features->at<double>(0,7) = incident_angle;
  features->at<double>(0,8) = surface_angle;
  size_t cur_mat_index = 9;
  cur_mat_index = writeEigenVectorToOpenCvMatrix(cartesian_eigenvalues, features, cur_mat_index);
  cur_mat_index = writeEigenVectorToOpenCvMatrix(spherical_eigenvalues, features, cur_mat_index);
  cur_mat_index = writeEigenVectorToOpenCvMatrix(cartesian_mean, features, cur_mat_index);
  cur_mat_index = writeEigenVectorToOpenCvMatrix(spherical_mean, features, cur_mat_index);
  cur_mat_index = writeEigenVectorToOpenCvMatrix(cartesian_variance, features, cur_mat_index);
  cur_mat_index = writeEigenVectorToOpenCvMatrix(spherical_variance, features, cur_mat_index);
  cur_mat_index = writeVectorToOpenCvMatrix(eigenvalue_features, features, cur_mat_index);
  cur_mat_index = writeVectorToOpenCvMatrix(histogram, features, cur_mat_index);
  cur_mat_index = writeVectorToOpenCvMatrix(density_vec, features, cur_mat_index);
}

std::pair<double, double> FeatureExtractor::getDensities(Eigen::MatrixXd points,
                                                         const Eigen::Vector3d& normal) {
  const size_t n_points = points.cols();
  const Eigen::Vector3d mean = points.rowwise().mean();
  points -= mean.replicate(1, n_points);
  const Eigen::VectorXd dist_from_mean = points.colwise().norm();
  const double r3 = dist_from_mean.maxCoeff();
  const double density_vol = n_points / (4/3*M_PI*r3*r3*r3);

  const Eigen::RowVectorXd scalar_products = normal.transpose() * points;
  const Eigen::MatrixXd surface_offset =
      scalar_products.replicate(3,1).cwiseProduct(normal.replicate(1, n_points));
  const Eigen::MatrixXd projected = points - surface_offset;
  const Eigen::VectorXd dist_from_center = projected.colwise().norm();
  const double r2 = dist_from_center.maxCoeff();
  const double density_surf = n_points / (M_PI*r2*r2);

  return std::make_pair(density_vol, density_surf);
}

void FeatureExtractor::getHistogram(const Cluster& cluster,
                                          std::vector<size_t>* histogram) {
  CHECK_NOTNULL(histogram);
  histogram->assign(params_.n_bins, 0);

  size_t ind;
  for (auto&& point: cluster.points) {
    if (point.distance >= params_.min_val) {
      ind = (point.distance - params_.min_val) / params_.bin_size;
      if (ind > params_.n_bins) ind = params_.n_bins;
      ++(histogram->at(ind));
    }
  }
}

void FeatureExtractor::getEigenvalueFeatures(Eigen::Vector3d eig,
                                             std::vector<double>* eigenvalue_features) {
  CHECK_NOTNULL(eigenvalue_features)->clear();
  std::sort(eig.data(), eig.data() + 3, std::greater<double>());  // Sort descending.
  eig /= eig.sum(); // Normalize.
  eigenvalue_features->push_back((eig(1) - eig(2)) / eig(1)); // linearity
//  eigenvalue_features->push_back((eig(2) - eig(3)) / eig(1)); // planarity
//  eigenvalue_features->push_back(eig(3) / eig(1));  // scattering
  eigenvalue_features->push_back(pow(eig.prod(), 1/3)); // omnivariance
//  eigenvalue_features->push_back((eig(1) - eig(3)) / eig(1)); // anisotrophy
//  eigenvalue_features->push_back(-(eig.array() * eig.array().log()).sum()); // eigenentropy
  eigenvalue_features->push_back(eig(3)); // change_of_curvature

  // Commented features have shown to be counterproductive in MATLAB.
  // They're still here in case that changes with OpenCV.
}
