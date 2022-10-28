/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "trustid.grpc.pb.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"
#include "trustid_image_processing/face_detector.h"
#include "trustid_image_processing/face_verificator.h"
//#include "DeviceEnumerator.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using trustid::grpc::BuildModelRequest;
using trustid::grpc::TRUSTIDClientProcessor;
using trustid::grpc::VerifyFaceRequest;
using trustid::grpc::VerifyFaceResponse;

class TRUSTIDClientImpl {
 public:
  TRUSTIDClientImpl(std::shared_ptr<Channel> channel)
      : stub_(TRUSTIDClientProcessor::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  trustid::image::FaceDetectionResult DetectFaces(const cv::Mat& image) {
    // Data we are sending to the server.

    trustid::grpc::DetectFacesRequest request;

    serialize_to_grpc(image, request.mutable_image(), false);

    // Container for the data we expect from the server.
    trustid::grpc::DetectFacesResponse reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->DetectFaces(&context, request, &reply);

    std::vector<trustid::image::FaceDetectionConfidenceBoundingBox>
        faceEntryVector;
    faceEntryVector.reserve(reply.detectionresults_size());

    for (auto& entry : reply.detectionresults()) {
      trustid::image::FaceDetectionResultEntry faceEntry;
      std::stringstream ss(entry.dlibserializeddata());
      dlib::deserialize(ss) >>
          faceEntry;
      faceEntryVector.push_back(faceEntry.getFaceDetBoundingBox());
    }

    // Act upon its status.
    if (status.ok()) {
      return trustid::image::FaceDetectionResult(image, faceEntryVector);
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      // throw std::runtime_error("RPC failed");
      return trustid::image::FaceDetectionResult(
          image, faceEntryVector,
          trustid::image::FaceDetectionResultValueEnum::NO_RESULTS);
    }
  }

  trustid::image::FaceVerificationResultEnum VerifyFace(
      const trustid::image::FaceDetectionResultEntry& detectionEntry, 
      const trustid::image::impl::DlibFaceVerificatorModelParams& modelData) {
    // Data we are sending to the server.

    trustid::grpc::VerifyFaceRequest request;

    std::stringstream ss;
    dlib::serialize(ss) << detectionEntry;
    request.mutable_previousdetection()->set_dlibserializeddata(ss.str());

    std::stringstream ss1;
    dlib::serialize(ss1) << modelData;
    request.mutable_modeldata()->set_dlibserializeddata(ss1.str());
    
    
    // Container for the data we expect from the server.
    trustid::grpc::VerifyFaceResponse reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->VerifyFace(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      if (reply.result() == "SAME_USER") {
        return trustid::image::FaceVerificationResultEnum::SAME_USER;
      } else if (reply.result() == "DIFFERENT_USER") {
        return trustid::image::FaceVerificationResultEnum::DIFFERENT_USER;
      } else {
        return trustid::image::FaceVerificationResultEnum::UNKNOWN;
      }
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      // throw std::runtime_error("RPC failed");
      return trustid::image::FaceVerificationResultEnum::UNKNOWN;
    }
  }

  trustid::image::impl::DlibFaceVerificatorModelParams BuildModel(
      const std::vector<trustid::image::FaceDetectionResultEntry>&
          detectionEntries) {
    // Data we are sending to the server.

    trustid::grpc::BuildModelRequest request;

    for (auto entry : detectionEntries) {
      std::stringstream ss;
      dlib::serialize(ss) << entry;
      request.add_detectionresults()->set_dlibserializeddata(ss.str());
    }

    // Container for the data we expect from the server.
    trustid::grpc::BuildModelResponse reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The actual RPC.
    Status status = stub_->BuildModel(&context, request, &reply);
    trustid::image::impl::DlibFaceVerificatorModelParams modelData;
    // Act upon its status.
    if (status.ok()) {
      std::stringstream ss(reply.modeldata().dlibserializeddata());
      dlib::deserialize(
          ss) >>
          modelData;
      return modelData;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return modelData;
    }
  }

 private:
  std::unique_ptr<TRUSTIDClientProcessor::Stub> stub_;
};

cv::VideoCapture openCamera(int camera_index){
  cv::VideoCapture cap(camera_index, cv::CAP_ANY);
  // check if we succeeded
  if (!cap.isOpened()) {
    std::cerr << "ERROR: Unable to open default camera\n" << std::endl;
    throw std::runtime_error("Unable to open camera");
  }
  return cap;
}

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint specified by
  // the argument "--target=" which is the only expected argument.
  // We indicate that the channel isn't authenticated (use of
  // InsecureChannelCredentials()).

  std::string camera_name;
  int camera_index = 0;
  std::string target_str = "0.tcp.eu.ngrok.io:17321"; //"localhost:50051";
  std::string arg_str("--cap");

  if (argc > 1) {
    std::string arg_val = argv[1];
    size_t start_pos = arg_val.find(arg_str);
    if (start_pos != std::string::npos) {
      start_pos += arg_str.size();
      if (arg_val[start_pos] == '=') {
        camera_name = arg_val.substr(start_pos + 1);
      } else {
        std::cout << "The only correct argument syntax is --cap="
                  << std::endl;
        return 0;
      }
    } else {
      std::cout << "The only acceptable argument is --cap=" << std::endl;
      return 0;
    }
  } else {
    target_str = "localhost:50051";
  }

  /*DeviceEnumerator enumerator;
  std::cerr << "\nListing all available cameras:" << std::endl;
  for (auto device : enumerator.getVideoDevicesMap()){
    std::cout << device.first << " " << device.second.id << " " << device.second.deviceName << " " << device.second.devicePath << std::endl;
    if (camera_name != "" && device.second.deviceName == camera_name){
      camera_index = device.second.id;
    }
  }
  std::cout<< "\nCamera to use: " << camera_name << std::endl;
  */
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(-1);
  args.SetMaxSendMessageSize(-1);

  TRUSTIDClientImpl greeter(grpc::CreateCustomChannel(
      target_str, grpc::InsecureChannelCredentials(), args));

  //--- INITIALIZE VIDEOCAPTURE
  cv::VideoCapture cap= openCamera(camera_index);

  time_t old_time = time(NULL);

  bool canVerifyFaces = false;
  std::vector<trustid::image::FaceDetectionResultEntry> faceEntryVector;
  trustid::image::impl::DlibFaceVerificatorModelParams modelParams;

  //--- GRAB AND WRITE LOOP
  std::cout << "Start grabbing" << std::endl
            << "Press any key to terminate" << std::endl;
  for (;;) {
    cv::Mat frame;
    // wait for a new frame from camera and store it into 'frame'
    cap.read(frame);

    // check if we succeeded
    if (frame.empty()) {
      std::cerr << "ERROR! blank frame grabbed\n";
      break;
    }

    auto detectedFaces = greeter.DetectFaces(frame);

    // check if there's exactly one face on the image
    if (detectedFaces.getResult() == trustid::image::ONE_RESULT) {
      if (!canVerifyFaces) {
        time_t new_time = time(NULL);
        if (new_time - old_time > 1) {
          if (faceEntryVector.size() < 5) {
            old_time = new_time;
            faceEntryVector.push_back(detectedFaces.getEntry());
            std::cout << "Current count: " << faceEntryVector.size()
                      << std::endl;
          } else {
            modelParams = greeter.BuildModel(faceEntryVector);
            canVerifyFaces = true;
          }
        }
      } else {
        auto verificationResult = greeter.VerifyFace(detectedFaces.getEntry(), modelParams);
        cv::putText(
            frame,
            ((verificationResult ==
              trustid::image::FaceVerificationResultEnum::SAME_USER)
                 ? std::string("Same User")
                 : std::string("Different User")),
            cv::Point(detectedFaces.getEntry().getBoundingBox().tl().x,
                      detectedFaces.getEntry().getBoundingBox().tl().y - 20),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
      }
      cv::rectangle(frame, detectedFaces.getEntry().getBoundingBox(),
                    cv::Scalar(255, 0, 0), 2);
    } else {
      for (auto face : detectedFaces.getBoundingBoxEntries()) {
        cv::rectangle(frame, face.getBoundingBox(), cv::Scalar(255, 0, 0), 2);
      }
    }

    // show live and wait for a key with timeout long enough to show images
    cv::imshow("Live", frame);
    if (cv::waitKey(5) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor

  return 0;
}