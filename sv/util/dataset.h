#pragma once

#include <absl/container/flat_hash_map.h>

#include <array>
#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include "sv/util/poly.h"

namespace sv {

/// @brief Convert mat to SE3d, assume rotation is orthogonal
Sophus::SE3d SE3dFromMat(const cv::Mat& mat);
void MatToSE3d(const cv::Mat& mat, Sophus::SE3d& tf);

/// @brief Get all file in a directory, optionally filter by extension
/// @return all files
std::vector<std::string> GetFiles(const std::string& dir,
                                  std::string_view ext = "",
                                  bool sort = false);

/// @brief Read color/grayscale image
/// @return cv::Mat 8UC3 BGR8 or 8UC1 MONO8
cv::Mat CvReadImage(const std::string& file);

/// @brief CvReadDepth
/// @param file
/// @param div_factor divide raw depth by this to get metric depth
/// @return cv::Mat 32FC1 float
cv::Mat CvReadDepth(const std::string& file, double div_factor = 1.0);

/// @brief Threshold depth by min_depth and max_depth
cv::Mat ThresholdDepth(const cv::Mat& depth,
                       double min_depth = -1.0,
                       double max_depth = -1.0);

// Maybe use an enum?
struct DataType {
  inline static const std::string kImage = "image";
  inline static const std::string kDepth = "depth";
  inline static const std::string kIntrin = "intrin";
  inline static const std::string kPose = "pose";
};

// Data retrieving function, takes index and returns cv::Mat
using CvMatDict = absl::flat_hash_map<std::string, cv::Mat>;
using DtypeFilesDict =
    absl::flat_hash_map<std::string, std::vector<std::string>>;

// Base class of dataset
class DatasetBase {
 public:
  inline static const std::string kDatasetDir = "/home/chao/Workspace/dataset";

  DatasetBase() = default;
  DatasetBase(const std::string& name,
              const std::string& data_dir,
              const std::vector<std::string>& dtypes);
  virtual ~DatasetBase() noexcept = default;

  cv::Mat Get(std::string_view dtype, int i, int cam) const;
  virtual cv::Mat GetImpl(std::string_view dtype, int i, int cam) const = 0;

  virtual std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DatasetBase& rhs) {
    return os << rhs.Repr();
  }

  int size() const noexcept { return size_; }
  std::string name() const noexcept { return name_; }
  const auto& dtypes() const noexcept { return dtypes_; }
  const auto& data_dir() const noexcept { return data_dir_; }
  bool is_stereo() const;

 protected:
  std::string name_;                 // name of dataset
  std::string data_dir_;             // dir of dataset
  std::vector<std::string> dtypes_;  // types of data
  int size_{};                       // size of dataset

  // map from dtype to files
  DtypeFilesDict files_;
  // map from dtype to data of each camera
  CvMatDict data_;
};

// Type-erased Dataset
class Dataset : public Poly<DatasetBase> {
 public:
  using Poly::Poly;

  bool Ok() const noexcept { return self_ != nullptr; }
  int size() const { return self_->size(); }
  std::string name() const { return self_->name(); }
  std::string Repr() const { return self_->Repr(); }
  const auto& dtypes() const noexcept { return self_->dtypes(); }
  bool is_stereo() const { return self_->is_stereo(); }

  friend std::ostream& operator<<(std::ostream& os, const Dataset& rhs) {
    return os << rhs.Repr();
  }

  cv::Mat Get(std::string_view dtype, int i, int cam = 0) const {
    return self_->Get(dtype, i, cam);
  }
};

/// IclNuim dataset
/// https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
class IclNuim final : public DatasetBase {
 public:
  // Raw depth should be divided by this to get metric depth in meter
  static constexpr double kDepthDivFactor = 5000.0;
  // Default data types for this dataset
  inline static const std::vector<std::string> kDtypes = {
      DataType::kImage, DataType::kDepth, DataType::kIntrin, DataType::kPose};

  explicit IclNuim(const std::string& data_dir);

 private:
  // Get data of dtype at index i, camera cam
  cv::Mat GetImpl(std::string_view dtype, int i, int cam) const override;
  cv::Mat ReadPoses(const std::string& file) const;
  cv::Mat ConvertPoses(const cv::Mat& poses) const;
};

/// Vkitti2 dataset
/// https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
class Vkitti2 final : public DatasetBase {
 public:
  // Raw depth should be divided by this to get metric depth in meter
  static constexpr double kDepthDivFactor = 100.0;
  // Baseline for stereo cameras [m]
  static constexpr double kBaseline = 0.532725;
  // Number of cameras
  static constexpr int kNumCams = 2;
  // Default data types for this dataset
  inline static const std::vector<std::string> kDtypes = {
      DataType::kImage, DataType::kDepth, DataType::kIntrin, DataType::kPose};

  explicit Vkitti2(const std::string& data_dir);

  static Vkitti2 Create(const std::string& base_dir,
                        int seq,
                        const std::string& var);

 private:
  int ToDataInd(int i, int cam = 0) const { return i * kNumCams + cam; }
  int ToFileInd(int i, int cam = 0) const { return i + cam * size_; }

  // Get data of dtype at index i, camera cam
  cv::Mat GetImpl(std::string_view dtype, int i, int cam) const override;
  cv::Mat ReadExtrinsics(const std::string& file) const;
  cv::Mat ReadIntrinsics(const std::string& file) const;
  cv::Mat ConvertExtrins(const cv::Mat& extrins) const;
};

/// @brief TartanAir dataset
/// https://github.com/castacks/tartanair_tools
class TartanAir final : public DatasetBase {
 public:
  // Default data types for this dataset
  inline static const std::vector<std::string> kDtypes = {
      DataType::kImage, DataType::kDepth, DataType::kIntrin, DataType::kPose};

  explicit TartanAir(const std::string& data_dir);
  static TartanAir Create(const std::string& base_dir,
                          const std::string& scene,
                          const std::string& mode,
                          int seq);

 private:
  int ToInd(int i, int cam = 0) const { return i + cam * size_; }

  // Get data of dtype at index i, camera cam
  cv::Mat GetImpl(std::string_view dtype, int i, int cam) const override;
  cv::Mat ReadPoses(const std::string& file) const;
  cv::Mat ConvertPoses(const cv::Mat& poses) const;
};

class KittiOdom final : public DatasetBase {
 public:
  // Number of cameras
  static constexpr int kNumCams = 2;
  // Default data types for this dataset
  inline static const std::vector<std::string> kDtypes = {
      DataType::kImage, DataType::kIntrin, DataType::kPose};

  explicit KittiOdom(const std::string& data_dir);
  static KittiOdom Create(const std::string& base_dir, int seq);

 private:
  int ToInd(int i, int cam = 0) const { return i + cam * size_; }

  // Get data of dtype at index i, camera cam
  cv::Mat GetImpl(std::string_view dtype, int i, int cam) const override;
  cv::Mat ReadIntrinsics(const std::string& file);
  cv::Mat ReadPoses(const std::string& file) const;
  cv::Mat ConvertPoses(const cv::Mat& poses) const;

  double baseline_{};
};

class StereoFolder final : public DatasetBase {
 public:
  // Default data types for this dataset
  inline static const std::vector<std::string> kDtypes = {DataType::kImage,
                                                          DataType::kIntrin};

  explicit StereoFolder(const std::string& name,
                        const std::string& left_dir,
                        const std::string& right_dir,
                        const std::string& calib_file);
  static StereoFolder Create(const std::string& data_name,
                             const std::string& data_dir,
                             const std::string& left_name,
                             const std::string& right_name,
                             const std::string& calib_name);

 private:
  int ToInd(int i, int cam = 0) const { return i + cam * size_; }

  // Get data of dtype at index i, camera cam
  cv::Mat GetImpl(std::string_view dtype, int i, int cam) const override;
  cv::Mat ReadIntrinsics(const std::string& file);

  double baseline_{};
};

/// @brief Create dataset by extracting dataset name from base_dir, or use name
/// if given
Dataset CreateDataset(const std::string& base_dir, std::string name = "");

}  // namespace sv
