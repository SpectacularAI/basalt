/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
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

#include <basalt/calibration/cam_calib.h>

#include <CLI/CLI.hpp>

int main(int argc, char **argv) {
  std::string dataset_path;
  std::string dataset_type;
  std::string calib_pattern_path;
  std::string result_path;
  std::vector<std::string> cam_types;
  std::string cache_dataset_name = "calib-cam";
  int skip_images = 1;

  CLI::App app{"Calibrate IMU"};

  app.add_option("--dataset-path", dataset_path, "Path to dataset")->required();
  app.add_option("--result-path", result_path, "Path to result folder")
      ->required();
  app.add_option("--dataset-type", dataset_type, "Dataset type (euroc, bag)")
      ->required();

  app.add_option("--aprilgrid", calib_pattern_path, // TODO: rename / add alias
                 "Path to calibration pattern (e.g., Aprilgrid) config file)")
      ->required();

  app.add_option("--cache-name", cache_dataset_name,
                 "Name to save cached files");

  app.add_option("--skip-images", skip_images, "Number of images to skip");
  app.add_option("--cam-types", cam_types,
                 "Type of cameras (eucm, ds, kb4, pinhole)")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  basalt::CamCalib cv(dataset_path, dataset_type, calib_pattern_path, result_path,
                      cache_dataset_name, skip_images, cam_types);

  cv.renderingLoop();

  return 0;
}
