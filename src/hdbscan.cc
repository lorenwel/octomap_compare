#include "hdbscan/hdbscan.h"

#include <chrono>
#include <iostream>

#include <boost/python.hpp>

#include <numpy_eigen/NumpyEigenConverter.hpp>

  Hdbscan::Hdbscan(const unsigned int& min_pts) : min_pts_(min_pts) {
    try {
      Py_Initialize();
      import_array();
      // Retrieve the main module.
      boost::python::object main = boost::python::import("__main__");
      // Retrieve the main module's namespace
      boost::python::object global = main.attr("__dict__");

      // Create python HDBSCAN object.
      std::string code =

R"(import hdbscan
def getHdbscan():
  return hdbscan.HDBSCAN(min_cluster_size=)" +
                         std::to_string(min_pts) +
                         ", gen_min_span_tree=False, metric='euclidean')";

      boost::python::exec(boost::python::str(code), global, global);
      boost::python::object temp = global["getHdbscan"];
      boost::python::object clusterer = temp();

      // Register conversions.
      NumpyEigenConverter<Eigen::Matrix<double, Eigen::Dynamic, 3>>::register_converter();
      NumpyEigenConverter<Eigen::VectorXi>::register_converter();

      // Pack all this into a function. We do this to make a copy of boost::python::object clusterer.
      // Otherwise we would need to have it as a class member variable which makes an include of
      // Python.h in the file that uses this clustering necessary.
      cluster_func_ = [&, clusterer](const Eigen::Matrix<double, Eigen::Dynamic, 3>& matrix,
                                     Eigen::VectorXi* indices) {
        try {
          auto start = std::chrono::high_resolution_clock::now();
          indices->resize(matrix.rows(), 1);
          const boost::python::object numpy_matrix(matrix);
          const boost::python::object result = clusterer.attr("fit_predict")(numpy_matrix);
          auto end = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> duration = end - start;
          std::cout << "HDBSCAN took " << duration.count() << " seconds\n";
          const size_t length = boost::python::len(result);
          for (size_t i = 0; i < length; ++i) {
            (*indices)(i) = boost::python::extract<int>(result[i]) + 1;
          }
    //      *indices = boost::python::extract<Eigen::VectorXi>(result);
        }
        catch (boost::python::error_already_set const&) {
          PyErr_Print();
        }
      };
    }
    catch (boost::python::error_already_set const&) {
      PyErr_Print();
    }

  }

  void Hdbscan::cluster(const Eigen::Matrix<double, Eigen::Dynamic, 3>& matrix,
                        Eigen::VectorXi* indices) {
    const size_t n_points = matrix.rows();
    if (n_points >= min_pts_) {
      cluster_func_(matrix, indices);
    }
    else {
      *indices = Eigen::VectorXi::Zero(n_points);
    }
  }

// For reference: this is the python code packed into std::string code
//
//  import hdbscan
//
//  def getHdbscan():
//    return hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=False, metric='euclidean')
