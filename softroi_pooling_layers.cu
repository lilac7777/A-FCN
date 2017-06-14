// --------------------------------------------------------
// R-FCN
// Written by Yi Li, 2016.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softroi_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void SOFTROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const int N,
    const int height, const int width,
    const Dtype* bottom_mask,
    const int output_dim,
    const int K,
    Dtype* top_data,
    Dtype * mask_spat_sum) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (N, output_dim, K)
      int mask_index = index % ( K * height * width );
      int n = index / output_dim / K / height / width ;
      int k = (index / height / width)% K ;
      int m = (index / height / width / K)% output_dim;
      mask_index += n* K * height * width;

      top_data[n*output_dim*K + m*K + k] += bottom_data[index]*bottom_mask[mask_index];
      mask_spat_sum[n*output_dim*K + m*K + k] += bottom_mask[mask_index];
      //printf("%d,%d,%d,%d,%d\n",index,n,m,k,mask_index);


    }
    
  }

  template <typename Dtype>
  void SOFTROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_mask = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* mask_spat_sum = mask_spat_sum_.mutable_gpu_data();
    int count = bottom[0]->count();
    caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
    caffe_gpu_set(N_*K*output_dim_, Dtype(0), mask_spat_sum);
    
    // NOLINT_NEXT_LINE(whitespace/operators)
    SOFTROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, N_ , height_, width_, bottom_mask, output_dim_, K,  top_data, mask_spat_sum);
    CUDA_POST_KERNEL_CHECK;
    
    //printf("%d,%d,%d\n",N_,output_dim_,K);
    caffe_gpu_div(N_*output_dim_*K, top_data,mask_spat_sum,top_data);
    
  }

  template <typename Dtype>
  __global__ void SOFTROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int height, const int width,
    const int K,
    const int output_dim,
    Dtype* bottom_diff_data,
    Dtype* bottom_diff_mask,
    const Dtype* bottom_data,
    const Dtype* bottom_mask,
    const Dtype* mask_spat_sum) {

    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int idx = index % (height * width); 
      int mask_index = index % ( K * height * width );
      int n = index / output_dim / K / height / width ;
      int k = (index / height / width)% K ;
      int m = (index / height / width / K)% output_dim;
      mask_index += n* K * height * width;
      bottom_diff_data[index] = top_diff[n*output_dim*K + m*K + k]*bottom_mask[mask_index]/mask_spat_sum[n*output_dim*K + m*K + k];
      //printf("%f,%f\n",bottom_diff_data[index],top_diff[n*output_dim*K + m*K + k]);
      int data_start =  (n*output_dim*K + m*K + k)*height*width;
      int mask_start = (n*K + k)*height*width;
      bottom_diff_mask[mask_index] +=  top_diff[n*output_dim*K + m*K + k]*bottom_data[index]/mask_spat_sum[n*output_dim*K + m*K + k];
      for (int h = 0; h < height; h ++){
      	for (int w = 0; w < width ; w++ ){
      		bottom_diff_mask[mask_start + idx] -= top_diff[n*output_dim*K + m*K + k]*bottom_data[data_start+h*width+w]*bottom_mask[mask_start+h*width+w]/mask_spat_sum[n*output_dim*K + m*K + k]/mask_spat_sum[n*output_dim*K + m*K + k];
      	}
      }

     }
  }

  template <typename Dtype>
  void SOFTROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //if (!propagate_down[0]) {
    //  return;
    //}
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_mask = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff_data = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_mask= bottom[1]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const Dtype* mask_spat_sum = mask_spat_sum_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_diff_mask);
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff_data);
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SOFTROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, height_, width_, K, output_dim_, bottom_diff_data,bottom_diff_mask, bottom_data, bottom_mask, mask_spat_sum);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(SOFTROIPoolingLayer);

}  // namespace caffe
