#pragma once

#include "iostream"
#include "trustid.grpc.pb.h"
#include "opencv2/opencv.hpp"

trustid::grpc::OCVImage* serialize_to_grpc(const cv::Mat item, trustid::grpc::OCVImage* image, bool copyData = true) {
    // check if we should copy the data or not
    if (copyData){
      void* data = malloc(item.total() * item.elemSize());
      memcpy(data, item.data, item.total() * item.elemSize());
      image->set_data(data, item.total() * item.elemSize());
    }
    else{
      image->set_data(item.data, item.total() * item.elemSize());
    }
    image->set_type(item.type());
    image->set_width(item.cols);
    image->set_height(item.rows);
    image->set_step(item.step);
    return image;
}

cv::Mat deserialize_from_grpc(const trustid::grpc::OCVImage in, bool copyData = true) {
  // check if we should copy the data or not
  if (copyData){
    // allocate data in the heap and pass the new pointer to cv::Mat (they will take ownership of the data)
    void* data = malloc(in.data().size());
    memcpy(data, in.data().c_str(), in.data().size());
    return cv::Mat(in.height(), in.width(), in.type(), std::move(data), in.step());
  }
  else{
    return cv::Mat(in.height(), in.width(), in.type(), (void*)in.data().c_str(), in.step());
  }
}

cv::Mat deserialize_from_grpc(const trustid::grpc::OCVImage* in, bool copyData = true) {
  // check if we should copy the data or not
  if (copyData){
    void* data = malloc(in->data().size());
    memcpy(data, in->data().c_str(), in->data().size());

    // convert to a placeholder vector to store byte data
    return cv::Mat(in->height(), in->width(), in->type(), data);
  }
  else{
    // convert to a placeholder vector to store byte data
    return cv::Mat(in->height(), in->width(), in->type(), (void*)in->data().c_str());
  }
}