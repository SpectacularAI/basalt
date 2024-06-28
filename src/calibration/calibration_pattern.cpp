/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
Copyright (c) 2022, Otto Seiskari, Spectacular AI Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cassert>
#include <fstream>
#include <sstream>

#include <basalt/calibration/calibration_pattern.h>
#include <basalt/calibration/calibration_helper.h>
#include <basalt/utils/apriltag.h>

#include <cereal/archives/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace basalt {
namespace {
struct AbstractCalibrationPattern {
  virtual ~AbstractCalibrationPattern() = default;

  virtual void detectCorners(const ImageData &img,
    CalibCornerData &good,
    CalibCornerData &bad,
    const int64_t timestamp_ns,
    int camIdx) const = 0;

  virtual std::vector<std::vector<int>> getFocalLengthTestLines() const = 0;
};

struct AprilGrid : AbstractCalibrationPattern {
  int tagCols;        // number of apriltags
  int tagRows;        // number of apriltags
  double tagSize;     // size of apriltag, edge to edge [m]
  double tagSpacing;  // ratio of space between tags to tagSize

  AprilGrid(std::istream &is, CalibrationPattern &pattern) {
    cereal::JSONInputArchive ar(is);
    ar(cereal::make_nvp("tagCols", tagCols));
    ar(cereal::make_nvp("tagRows", tagRows));
    ar(cereal::make_nvp("tagSize", tagSize));
    ar(cereal::make_nvp("tagSpacing", tagSpacing));

    double x_corner_offsets[4] = {0, tagSize, tagSize, 0};
    double y_corner_offsets[4] = {0, 0, tagSize, tagSize};

    pattern.corner_pos_3d.resize(tagCols * tagRows * 4);

    for (int y = 0; y < tagRows; y++) {
      for (int x = 0; x < tagCols; x++) {
        int tag_id = tagCols * y + x;
        double x_offset = x * tagSize * (1 + tagSpacing);
        double y_offset = y * tagSize * (1 + tagSpacing);

        for (int i = 0; i < 4; i++) {
          int corner_id = (tag_id << 2) + i;

          Eigen::Vector4d &pos_3d = pattern.corner_pos_3d[corner_id];

          pos_3d[0] = x_offset + x_corner_offsets[i];
          pos_3d[1] = y_offset + y_corner_offsets[i];
          pos_3d[2] = 0;
          pos_3d[3] = 1;
        }
      }
    }

    int num_vign_points = 5;
    int num_blocks = tagCols * tagRows * 2;

    pattern.vignette_pos_3d.resize((num_blocks + tagCols + tagRows) *
                                    num_vign_points);

    for (int k = 0; k < num_vign_points; k++) {
      for (int i = 0; i < tagCols * tagRows; i++) {
        // const Eigen::Vector3d p0 = corner_pos_3d[4 * i + 0];
        const Eigen::Vector4d p1 = pattern.corner_pos_3d[4 * i + 1];
        const Eigen::Vector4d p2 = pattern.corner_pos_3d[4 * i + 2];
        const Eigen::Vector4d p3 = pattern.corner_pos_3d[4 * i + 3];

        double coeff = double(k + 1) / double(num_vign_points + 1);

        pattern.vignette_pos_3d[k * num_blocks + 2 * i + 0] =
            (p1 + coeff * (p2 - p1));
        pattern.vignette_pos_3d[k * num_blocks + 2 * i + 1] =
            (p2 + coeff * (p3 - p2));

        pattern.vignette_pos_3d[k * num_blocks + 2 * i + 0][0] +=
            tagSize * tagSpacing / 2;
        pattern.vignette_pos_3d[k * num_blocks + 2 * i + 1][1] +=
            tagSize * tagSpacing / 2;
      }
    }

    size_t curr_idx = num_blocks * num_vign_points;

    for (int k = 0; k < num_vign_points; k++) {
      for (int i = 0; i < tagCols; i++) {
        const Eigen::Vector4d p0 = pattern.corner_pos_3d[4 * i + 0];
        const Eigen::Vector4d p1 = pattern.corner_pos_3d[4 * i + 1];

        double coeff = double(k + 1) / double(num_vign_points + 1);

        pattern.vignette_pos_3d[curr_idx + k * tagCols + i] =
            (p0 + coeff * (p1 - p0));

        pattern.vignette_pos_3d[curr_idx + k * tagCols + i][1] -=
            tagSize * tagSpacing / 2;
      }
    }

    curr_idx += tagCols * num_vign_points;

    for (int k = 0; k < num_vign_points; k++) {
      for (int i = 0; i < tagRows; i++) {
        const Eigen::Vector4d p0 = pattern.corner_pos_3d[4 * i * tagCols + 0];
        const Eigen::Vector4d p3 = pattern.corner_pos_3d[4 * i * tagCols + 3];

        double coeff = double(k + 1) / double(num_vign_points + 1);

        pattern.vignette_pos_3d[curr_idx + k * tagRows + i] =
            (p0 + coeff * (p3 - p0));

        pattern.vignette_pos_3d[curr_idx + k * tagRows + i][0] -=
            tagSize * tagSpacing / 2;
      }
    }
  }

  void detectCorners(const ImageData &img,
    CalibCornerData &ccd_good,
    CalibCornerData &ccd_bad,
    const int64_t timestamp_ns,
    int camIdx) const final
  {
    (void)timestamp_ns;
    (void)camIdx;
    ApriltagDetector ad(tagCols*tagRows);
    ad.detectTags(*img.img, ccd_good.corners,
                  ccd_good.corner_ids, ccd_good.radii,
                  ccd_bad.corners, ccd_bad.corner_ids, ccd_bad.radii);
  }

  std::vector<std::vector<int>> getFocalLengthTestLines() const final {
    std::vector<std::vector<int>> result;
    for (int tag_corner_offset = 0; tag_corner_offset < 2; tag_corner_offset++)
      for (int r = 0; r < tagRows; ++r) {
        result.push_back({});
        std::vector<int> &line = result.back();


        for (int c = 0; c < tagCols; ++c) {
          int tag_offset = (r * tagCols + c) << 2;

          for (int i = 0; i < 2; i++) {
            int corner_id = tag_offset + i + tag_corner_offset * 2;

            // std::cerr << corner_id << " ";
            line.push_back(corner_id);
          }
        }

        // std::cerr << std::endl;
      }
    return result;
  }
};

struct Checkerboard : AbstractCalibrationPattern {
  // Checkerboard fields as named & defined in Kalibr
  int targetCols;        // number of internal chessboard corners
  int targetRows;        // number of internal chessboard corners
  double rowSpacingMeters;  // size of one chessboard square [m]
  double colSpacingMeters;  // size of one chessboard square [m]

  Checkerboard(std::istream &is, CalibrationPattern &pattern) {
    cereal::JSONInputArchive ar(is);
    ar(cereal::make_nvp("targetCols", targetCols));
    ar(cereal::make_nvp("targetRows", targetRows));
    ar(cereal::make_nvp("rowSpacingMeters", rowSpacingMeters));
    ar(cereal::make_nvp("colSpacingMeters", colSpacingMeters));
    pattern.corner_pos_3d.resize(targetCols * targetRows);

    Eigen::Vector4d vignette_offset(colSpacingMeters * 0.5, rowSpacingMeters * 0.5, 0, 0);

    for (int y = -1; y < targetRows; y++) {
      for (int x = -1; x < targetCols; x++) {
        Eigen::Vector4d pos_3d;
        pos_3d[0] = x * colSpacingMeters;
        pos_3d[1] = y * rowSpacingMeters;
        pos_3d[2] = 0;
        pos_3d[3] = 1;

        if (x >= 0 && y >= 0) {
          int corner_id = targetCols * y + x;
          pattern.corner_pos_3d[corner_id] = pos_3d;
        }

        if ((x + y + (targetCols*targetRows*2)) % 2 == 1) {
          // white corners
          pattern.vignette_pos_3d.push_back(pos_3d + vignette_offset);
        }
      }
    }
  }

  void detectCorners(const ImageData &img,
    CalibCornerData &ccd_good,
    CalibCornerData &ccd_bad,
    const int64_t timestamp_ns,
    int camIdx) const final
  {
    (void)timestamp_ns;
    (void)camIdx;
    ccd_good.corner_ids.clear();
    ccd_good.corners.clear();
    ccd_good.radii.clear();

    ccd_bad.corner_ids.clear();
    ccd_bad.corners.clear();
    ccd_bad.radii.clear();

    const auto &img_raw = *img.img;
    cv::Mat image(img_raw.h, img_raw.w, CV_8U);
    uint8_t* dst = image.ptr();
    const uint16_t* src = img_raw.ptr;
    for (size_t i = 0; i < img_raw.size(); i++) dst[i] = (src[i] >> 8);

    cv::Size pattern_size(targetCols, targetRows); // apparently cols, rows (the docs say otherwise?)
    std::vector<cv::Point2f> corners;
    if (!cv::findChessboardCorners(image, pattern_size, corners)) return;
    assert(int(corners.size()) == targetCols * targetRows);

    // auto-detect subpix pattern size
    constexpr int MAX_SUBPIX_PATTERN_SIZE = 15;
    float minDist2 = MAX_SUBPIX_PATTERN_SIZE*MAX_SUBPIX_PATTERN_SIZE;
    for (int y=0; y<targetRows; ++y) {
      for (int x=1; x<targetCols; ++x) {
        int c_id1 = y * targetCols + x;
        int c_id0 = c_id1 - 1;
        auto diff = corners.at(c_id1) - corners.at(c_id0);
        float dist2 = diff.x*diff.x + diff.y*diff.y;
        minDist2 = std::min(minDist2, dist2);
      }
    }

    const int subpixPatternSize = std::min(MAX_SUBPIX_PATTERN_SIZE, int(std::sqrt(minDist2)*0.5 + 0.5));
    // std::cout << "subpixel pattern size " << subpixPatternSize << std::endl;

    if (subpixPatternSize > 1) {
      // values from in OpenCV docs
      constexpr int SUBPIX_ITR = 30;
      constexpr double SUBPIX_EPS = 0.1;
      cv::cornerSubPix(image, corners,
        cv::Size(subpixPatternSize, subpixPatternSize),
        cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, SUBPIX_ITR, SUBPIX_EPS));
      assert(int(corners.size()) == targetCols * targetRows);
    }

    constexpr double RADIUS = 2.0; // not sure how this is defined

    for (int corner_id = 0; corner_id < int(corners.size()); ++corner_id) {
      ccd_good.corner_ids.push_back(corner_id);
      const auto &c = corners.at(corner_id);
      ccd_good.corners.push_back(Eigen::Vector2d(c.x, c.y));
      ccd_good.radii.push_back(RADIUS);
    }
  }

  std::vector<std::vector<int>> getFocalLengthTestLines() const final {
    std::vector<std::vector<int>> result;
    for (int r = 0; r < targetRows; ++r) {
      result.push_back({});
      std::vector<int> &line = result.back();
      for (int c = 0; c < targetCols; ++c) {
        int corner_id = r * targetCols + c;
        line.push_back(corner_id);
      }
    }
    return result;
  }
};

struct CustomCalibrationTargetPoint2D {
    std::vector<double> pixel;
    int id;

    // Make this struct serializable
    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(pixel), CEREAL_NVP(id));
    }
};

// Define a struct for the images
struct CustomCalibrationTargetFrame {
    int id;
    double time;
    std::vector<CustomCalibrationTargetPoint2D> points2d;

    // Make this struct serializable
    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(id), CEREAL_NVP(time), CEREAL_NVP(points2d));
    }
};

struct FrameDataIndex {
  std::unordered_map<int64_t, int> timeToFrameId;
  std::unordered_map<int, int> numberToFrameId;

  void add(int64_t time, int index) {
    // TODO: use camIdx
    if (timeToFrameId.count(time)) {
      std::cerr << "Duplicate timestamp " << time << " in custom calibration target!" << std::endl;
    }
    timeToFrameId[time] = index;
  }

  int get(int64_t time) const {
    if (timeToFrameId.count(time)) {
      return timeToFrameId.at(time);
    }

    return -1;
  }
};

struct CustomCalibrationTarget : AbstractCalibrationPattern {
private:
  std::vector<CustomCalibrationTargetFrame> frameData;
  FrameDataIndex frameDataIndex;

public:

  CustomCalibrationTarget(std::istream &is, CalibrationPattern &pattern) {
    cereal::JSONInputArchive ar(is);
    std::string type;
    ar(cereal::make_nvp("kind", type));
    assert(type == "custom");

    std::vector<std::vector<double>> pos3d;
    ar(cereal::make_nvp("features3d", pos3d));
    pattern.corner_pos_3d.resize(pos3d.size());
    for (size_t i = 0; i < pos3d.size(); i++) {
      const auto &pos = pos3d[i];
      pattern.corner_pos_3d[i] = Eigen::Vector4d(pos[0], pos[1], pos[2], 1);
    }
    std::cout << "Found " << pattern.corner_pos_3d.size() << " 3d points" << std::endl;

    ar(cereal::make_nvp("images", frameData));
    for (size_t i = 0; i < frameData.size(); i++) {
      int64_t time = frameData[i].time * 1e9;
      frameDataIndex.add(time, i);
    }
    std::cout << "Found " << frameData.size() << " frames" << std::endl;

  }

  void detectCorners(const ImageData &img,
    CalibCornerData &ccd_good,
    CalibCornerData &ccd_bad,
    const int64_t timestamp_ns,
    int camIdx) const final
  {
    (void)img;
    (void)camIdx;

    ccd_good.corner_ids.clear();
    ccd_good.corners.clear();
    ccd_good.radii.clear();

    ccd_bad.corner_ids.clear();
    ccd_bad.corners.clear();
    ccd_bad.radii.clear();

    int idx = frameDataIndex.get(timestamp_ns);
    if (idx == -1) return;
    const auto &fd = frameData[idx];

    constexpr double RADIUS = 2.0; // not sure how this is defined

    for (size_t i = 0; i < fd.points2d.size(); i++) {
      const auto &p = fd.points2d[i];
      ccd_good.corner_ids.push_back(p.id);
      ccd_good.corners.push_back(Eigen::Vector2d(p.pixel[0], p.pixel[1]));
      ccd_good.radii.push_back(RADIUS);
    }
  }

  std::vector<std::vector<int>> getFocalLengthTestLines() const final {
    std::cerr << "getFocalLengthTestLines() not implemented on Custom calibration target!" << std::endl;
    std::vector<std::vector<int>> result;
    return result;
  }
};

struct ExternalCheckerboard : AbstractCalibrationPattern {
private:
  std::vector<CustomCalibrationTargetFrame> frameData;
  FrameDataIndex frameDataIndex;
  int targetCols;        // number of internal chessboard corners
  int targetRows;        // number of internal chessboard corners
  double rowSpacingMeters;  // size of one chessboard square [m]
  double colSpacingMeters;  // size of one chessboard square [m]

public:

  ExternalCheckerboard(std::istream &is, CalibrationPattern &pattern) {
    cereal::JSONInputArchive ar(is);
    std::string type;
    ar(cereal::make_nvp("kind", type));
    assert(type == "ext_checker");

    ar(cereal::make_nvp("targetCols", targetCols));
    ar(cereal::make_nvp("targetRows", targetRows));
    ar(cereal::make_nvp("rowSpacingMeters", rowSpacingMeters));
    ar(cereal::make_nvp("colSpacingMeters", colSpacingMeters));
    pattern.corner_pos_3d.resize(targetCols * targetRows);
    Eigen::Vector4d vignette_offset(colSpacingMeters * 0.5, rowSpacingMeters * 0.5, 0, 0);
    for (int y = -1; y < targetRows; y++) {
      for (int x = -1; x < targetCols; x++) {
        Eigen::Vector4d pos_3d;
        pos_3d[0] = x * colSpacingMeters;
        pos_3d[1] = y * rowSpacingMeters;
        pos_3d[2] = 0;
        pos_3d[3] = 1;
        if (x >= 0 && y >= 0) {
          int corner_id = targetCols * y + x;
          pattern.corner_pos_3d[corner_id] = pos_3d;
        }
        if ((x + y + (targetCols*targetRows*2)) % 2 == 1) {
          // white corners
          pattern.vignette_pos_3d.push_back(pos_3d + vignette_offset);
        }
      }
    }

    ar(cereal::make_nvp("images", frameData));
    for (size_t i = 0; i < frameData.size(); i++) {
      int64_t time = frameData[i].time * 1e9;
      frameDataIndex.add(time, i);
    }
    std::cout << "Found " << frameData.size() << " frames" << std::endl;
  }

  void detectCorners(const ImageData &img,
    CalibCornerData &ccd_good,
    CalibCornerData &ccd_bad,
    const int64_t timestamp_ns,
    int camIdx) const final
  {
    (void)img;
    (void)camIdx;

    ccd_good.corner_ids.clear();
    ccd_good.corners.clear();
    ccd_good.radii.clear();

    ccd_bad.corner_ids.clear();
    ccd_bad.corners.clear();
    ccd_bad.radii.clear();

    int idx = frameDataIndex.get(timestamp_ns);
    if (idx == -1) return;
    const auto &fd = frameData[idx];

    constexpr double RADIUS = 2.0; // not sure how this is defined

    for (size_t i = 0; i < fd.points2d.size(); i++) {
      const auto &p = fd.points2d[i];
      ccd_good.corner_ids.push_back(p.id);
      ccd_good.corners.push_back(Eigen::Vector2d(p.pixel[0], p.pixel[1]));
      ccd_good.radii.push_back(RADIUS);
    }
  }

  std::vector<std::vector<int>> getFocalLengthTestLines() const final {
    std::vector<std::vector<int>> result;
    for (int r = 0; r < targetRows; ++r) {
      result.push_back({});
      std::vector<int> &line = result.back();
      for (int c = 0; c < targetCols; ++c) {
        int corner_id = r * targetCols + c;
        line.push_back(corner_id);
      }
    }
    return result;
  }
};

std::string slurpFile(const std::string &path) {
  std::ifstream is(path);
  if (!is.is_open()) std::abort();
  std::stringstream buf;
  buf << is.rdbuf();
  return buf.str();
}
}

struct CalibrationPattern::Impl {
  std::unique_ptr<AbstractCalibrationPattern> abstractPattern;

  Impl(const std::string &config_path, CalibrationPattern &pattern) {
    std::ifstream is(config_path);
    if (!is.is_open()) {
      std::cerr << "Could not open calibration pattern configuration: " << config_path
                << std::endl;
      std::abort();
    }

    // Deduce pattern type from JSON file (crude since Cereal has poor support for optional stuff)
    const std::string file_content = slurpFile(config_path);
    if (file_content.find("checkerboard")  != std::string::npos)
      abstractPattern = std::make_unique<Checkerboard>(is, pattern);
    else if (file_content.find("custom")  != std::string::npos)
      abstractPattern = std::make_unique<CustomCalibrationTarget>(is, pattern);
    else if (file_content.find("ext_checker")  != std::string::npos)
      abstractPattern = std::make_unique<ExternalCheckerboard>(is, pattern);
    else if (file_content.find("aprilgrid")  != std::string::npos || file_content.find("tagCols") != std::string::npos)
      abstractPattern = std::make_unique<AprilGrid>(is, pattern);
    else {
      std::cerr << "Invalid calibration pattern type" << std::endl;
      std::abort();
    }
  }
};

CalibrationPattern::CalibrationPattern(const std::string &config_path) :
  pImpl(std::make_unique<Impl>(config_path, *this))
{}

CalibrationPattern::~CalibrationPattern() = default;

void CalibrationPattern::detectCorners(const ImageData &img,
  CalibCornerData &goodCorners,
  CalibCornerData &badCorners,
  const int64_t timestamp_ns,
  int camIdx) const {
  pImpl->abstractPattern->detectCorners(img, goodCorners, badCorners, timestamp_ns, camIdx);
}

// Returns a list of lines, each of which consists of a list of corner IDs
std::vector<std::vector<int>> CalibrationPattern::getFocalLengthTestLines() const {
  return pImpl->abstractPattern->getFocalLengthTestLines();
}

}  // namespace basalt
