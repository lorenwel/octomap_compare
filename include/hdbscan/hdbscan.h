#ifndef HDBSCAN_H_
#define HDBSCAN_H_

#include <functional>

#include <Eigen/Dense>

class Hdbscan {

  std::function<void (const Eigen::Matrix<double, Eigen::Dynamic, 3>&,
                      Eigen::VectorXi*)>
                        cluster_func_;

  const unsigned int min_pts_;

public:

  Hdbscan(const unsigned int& min_pts = 15);

  void cluster(const Eigen::Matrix<double, Eigen::Dynamic, 3>& matrix,
               Eigen::VectorXi* indices);

};

#endif /* HDBSCAN_H_ */
