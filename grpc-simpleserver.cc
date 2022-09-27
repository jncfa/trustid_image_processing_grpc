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

#include "common.h"
#include "trustid.grpc.pb.h"
#include "trustid_image_processing/client/client_processor.h"
#include "trustid_image_processing/serialize.h"
#include "trustid_image_processing/server/server_processor.h"
#include "trustid_image_processing/utils.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using trustid::grpc::TRUSTIDClientProcessor;

// Logic and data behind the server's behavior.
class TRUSTIDClientProcessorImpl final
    : public trustid::grpc::TRUSTIDClientProcessor::Service {
  trustid::image::ClientImageProcessor clientProcessor;
  trustid::image::ServerImageProcessor serverProcessor;

  Status DetectFaces(::grpc::ServerContext *context,
                     const ::trustid::grpc::DetectFacesRequest *request,
                     ::trustid::grpc::DetectFacesResponse *response) override {
    cv::Mat image = deserialize_from_grpc(request->image());

    auto result = clientProcessor.detectFaces(image);
    
    for (auto entry: result.getBoundingBoxEntries()){
      std::ostringstream ss;
      dlib::serialize(ss) << entry;
      auto detectionResult= response->add_detectionresults();
      detectionResult->set_dlibserializeddata(ss.str());
    }

    switch (result.getResult()) {
      case trustid::image::FaceDetectionResultValueEnum::NO_RESULTS:
        response->set_result("NO_RESULTS");
        break;
      case trustid::image::FaceDetectionResultValueEnum::ONE_RESULT:
        response->set_result("ONE_RESULT");
        break;
      case trustid::image::FaceDetectionResultValueEnum::MULTIPLE_RESULTS:
        response->set_result("MULTIPLE_RESULTS");

        break;
      default:
        throw std::runtime_error("Unknown result");
        break;
    }
    return Status::OK;
  }

  Status VerifyFace(::grpc::ServerContext *context,
                    const ::trustid::grpc::VerifyFaceRequest *request,
                    ::trustid::grpc::VerifyFaceResponse *response) override {
    std::cout << "ReceivedVerifyFace RPC" << std::endl;

    trustid::image::FaceDetectionResultEntry parsedRequest;
    dlib::deserialize(std::istringstream(
        request->previousdetection().dlibserializeddata())) >>
        parsedRequest;
    auto result = clientProcessor.verifyUser(parsedRequest);
    switch (result.getResult()) {
      case trustid::image::FaceVerificationResultEnum::SAME_USER:
        response->set_result("SAME_USER");
        break;
      case trustid::image::FaceVerificationResultEnum::DIFFERENT_USER:
        response->set_result("DIFFERENT_USER");
        break;
      default:
        response->set_result("UNKNOWN");
        throw std::runtime_error("Unknown result");
    }
    return Status::OK;
  }

  Status LoadVerificationData(
      ::grpc::ServerContext *context,
      const ::trustid::grpc::LoadDataRequest *request,
      ::trustid::grpc::LoadDataResponse *response) override {
    std::cout << "Received LoadVerificationData RPC" << std::endl;
   
    trustid::image::impl::DlibFaceVerificatorConfig config;
    dlib::deserialize(
        std::stringstream(request->modeldata().dlibserializeddata())) >>
        config;
    clientProcessor.loadFaceVerificationData(config);
    response->set_result(true);
    return Status::OK;
  }

  Status EstimateHeadPose(
      ::grpc::ServerContext *context,
      const ::trustid::grpc::EstimateHeadPoseRequest *request,
      ::trustid::grpc::EstimateHeadPoseResponse *response) override {
    std::cout << "Received EstimateHeadPose RPC" << std::endl;
    // trustid::image::FaceDetectionResultEntry result;
    // dlib::deserialize(
    //     std::istringstream(request->detectionresult().dlibserializeddata()))
    //     >> result;
    // auto headPoseEstimation = clientProcessor.estimateHeadPose(result);
    response->set_result("UNKNOWN");

    // trustid::grpc::HeadPoseResult* poseResult =
    // response->mutable_headposedata(); auto strbuf = std::istringstream();
    // dlib::serialize(headPoseEstimation, strbuf);
    // poseResult->set_dlibserializeddata(strbuf.str());

    return Status::OK;
  }

  Status CheckImageQuality(
      ::grpc::ServerContext *context,
      const ::trustid::grpc::CheckImageQualityRequest *request,
      ::trustid::grpc::CheckImageQualityResponse *response) override {
    
    auto result = clientProcessor.checkImageQuality(
        deserialize_from_grpc(request->imagetocheck()));

    switch (result) {
      case trustid::image::utils::ImageQualityResultEnum::LOW_CONTRAST:
        response->set_result("LOW_CONTRAST");
        break;
      case trustid::image::utils::ImageQualityResultEnum::HIGH_CONTRAST:
        response->set_result("HIGH_CONTRAST");
        break;

      default:
        throw std::runtime_error("Unknown result");
        break;
    }
    return Status::OK;
  }
  
  Status BuildModel(
      ::grpc::ServerContext *context,
      const ::trustid::grpc::BuildModelRequest *request,
      ::trustid::grpc::BuildModelResponse *response) override {
        
        std::vector<trustid::image::FaceDetectionResultEntry> faceEntryVector;
        faceEntryVector.reserve(request->detectionresults_size());

        for (auto& entry : request->detectionresults()) {
          trustid::image::FaceDetectionResultEntry faceEntry;
          dlib::deserialize(std::stringstream(entry.dlibserializeddata())) >> faceEntry;
          faceEntryVector.push_back(faceEntry); 
        }
        
        // build model, grab config and return it
        auto result = serverProcessor.createVerificationModel(faceEntryVector);

        std::stringstream ss;
        dlib::serialize(ss) << result->getConfig();
        response->mutable_model()->set_dlibserializeddata(ss.str());
        response->set_result("MODEL_BUILT");
        return Status::OK;
      }
 public:
  TRUSTIDClientProcessorImpl() : clientProcessor(), serverProcessor() {}
};

void RunServer() {
  //HWND hWnd = GetConsoleWindow();
  //ShowWindow(hWnd, SW_HIDE);

  std::string server_address("0.0.0.0:50051");
  TRUSTIDClientProcessorImpl service;

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  builder.SetMaxMessageSize(-1);

  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char **argv) {
  RunServer();

  return 0;
}
