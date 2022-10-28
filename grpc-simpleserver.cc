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
#include "trustid_image_processing/serialize.h"
#include "trustid_image_processing/utils.h"
#include "trustid_image_processing/dlib_impl/face_detector.h"
#include "trustid_image_processing/dlib_impl/face_verificator.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using trustid::grpc::TRUSTIDClientProcessor;

// Logic and data behind the server's behavior.
class TRUSTIDClientProcessorImpl final
    : public trustid::grpc::TRUSTIDClientProcessor::Service {
  
  std::shared_ptr<ResNet34> net;
  std::shared_ptr<dlib::shape_predictor> sp;
  std::unique_ptr<trustid::image::IFaceDetector> faceDetector;
  

  Status DetectFaces(::grpc::ServerContext *context,
                     const ::trustid::grpc::DetectFacesRequest *request,
                     ::trustid::grpc::DetectFacesResponse *response) override {

    try { 
      auto start = std::chrono::steady_clock().now();

      //std::cout << "Image size: " << request->image().data().size() << std::endl;
      cv::Mat image = deserialize_from_grpc(request->image());

      auto result = faceDetector->detectFaces(image);
      
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
      auto finish = std::chrono::steady_clock().now();
      auto elapsed = finish - start;
      std::cout << "DetectFaces took " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" << std::endl;
      return Status::OK;
    }
    catch (std::exception &e){
      std::cerr << e.what() << std::endl;
      return Status::CANCELLED;
    }
  }

  Status VerifyFace(::grpc::ServerContext *context,
                    const ::trustid::grpc::VerifyFaceRequest *request,
                    ::trustid::grpc::VerifyFaceResponse *response) override {
    auto start = std::chrono::steady_clock().now();
    trustid::image::FaceDetectionResultEntry parsedRequest;
    std::istringstream ss(
        request->previousdetection().dlibserializeddata());
    dlib::deserialize(ss) >>
        parsedRequest;
    ss.clear();
    
    trustid::image::impl::DlibFaceVerificatorModelParams modelParams;
    ss.str(request->modeldata().dlibserializeddata());
    dlib::deserialize(
        ss) >>
        modelParams;
    
    auto faceVerificator = std::make_unique<trustid::image::impl::DlibFaceVerificator>(net, sp, modelParams);
    
    auto result = faceVerificator->verifyUser(parsedRequest);
    switch (result.getResult()) {
      case trustid::image::FaceVerificationResultEnum::SAME_USER:
        response->set_result("SAME_USER");
        response->set_confidencescore(result.getMatchConfidence());
        break;
      case trustid::image::FaceVerificationResultEnum::DIFFERENT_USER:
        response->set_result("DIFFERENT_USER");
        response->set_confidencescore(result.getMatchConfidence());
        break;
      default:
        response->set_result("UNKNOWN");
        response->set_confidencescore(result.getMatchConfidence());
        throw std::runtime_error("Unknown result");
    }
    auto finish = std::chrono::steady_clock().now();
    auto elapsed = finish - start;
    std::cout << "VerifyFaces took " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" << std::endl;
    return Status::OK;
  }
  
  Status BuildModel(
      ::grpc::ServerContext *context,
      const ::trustid::grpc::BuildModelRequest *request,
      ::trustid::grpc::BuildModelResponse *response) override {
        auto start = std::chrono::steady_clock().now();
        std::vector<trustid::image::FaceDetectionResultEntry> faceEntryVector;
        faceEntryVector.reserve(request->detectionresults_size());

        for (auto& entry : request->detectionresults()) {
          trustid::image::FaceDetectionResultEntry faceEntry;
          std::stringstream ss(entry.dlibserializeddata());
          dlib::deserialize(ss) >> faceEntry;
          faceEntryVector.push_back(faceEntry); 
        }
        
        // build model, grab config and return it
        auto result = std::make_unique<trustid::image::impl::DlibFaceVerificator>(net, sp, faceEntryVector)->getUserParams();

        std::stringstream ss;
        dlib::serialize(ss) << result;
        response->mutable_modeldata()->set_dlibserializeddata(ss.str());
        response->set_result("MODEL_BUILT");
        auto finish = std::chrono::steady_clock().now();
        auto elapsed = finish - start;
        std::cout << "VerifyFaces took " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" << std::endl;
        return Status::OK;
      }
 public:
  TRUSTIDClientProcessorImpl() : 
  net(trustid::image::impl::loadResNet34FromDisk("resources/dlib_face_recognition_resnet_model_v1.dat")), 
  sp(trustid::image::impl::loadShapePredictorFromDisk("resources/ERT68.dat")),
  faceDetector(std::make_unique<trustid::image::impl::DlibFaceDetector>()) {}
};

void RunServer() {
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
