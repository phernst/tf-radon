#include "parameter_classes.h"
#include "forward.h"
#include "texture.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <math.h>
#include <assert.h>


using namespace tensorflow;
using shape_inference::ShapeHandle;


REGISTER_OP("Forward")
   .Input("volume: float")
   .Input("angles: float")
   .Attr("vol_shape: shape")
   .Attr("proj_shape: shape")
   .Attr("det_spacing: list(float)")
   .Attr("src_dist: float")
   .Attr("det_dist: float")
   .Attr("vol_origin: list(float)")
   .Attr("voxel_dimen: list(float)")
   .Output("projections: float")
   .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
   {
      // TODO: correct shape
      TensorShapeProto sp;
      ShapeHandle sh;
      auto status = c->GetAttr( "vol_shape", &sp );
      status.Update( c->MakeShapeFromShapeProto( sp, &sh ) );
      c->set_output( 0, sh );
      return status;
   } )
;

class ForwardOp : public OpKernel
{
protected:
   // volume configuration
   std::unique_ptr<VolumeCfg> volume_cfg_;

   // projection configuration
   std::unique_ptr<ProjectionCfg> proj_cfg_;

   // texture cache
   TextureCache texture_cache_;

   static const ExecCfg defaultExecCfg()
   {
      return {8, 16, 8, 1};
   }
   
public:
   explicit ForwardOp(OpKernelConstruction* context)
      : OpKernel(context), texture_cache_{8}
   {
      TensorShape v;
      OP_REQUIRES_OK( context, context->GetAttr( "vol_shape", &v ) );

      std::vector<float> et;
      OP_REQUIRES_OK( context, context->GetAttr( "vol_origin", &et ) );

      std::vector<float> es;
      OP_REQUIRES_OK( context, context->GetAttr( "voxel_dimen", &es ) );

      volume_cfg_.reset(new VolumeCfg{
         static_cast<int>(v.dim_size( 0 )),
         static_cast<int>(v.dim_size( 1 )),
         static_cast<int>(v.dim_size( 2 )),
         et[0],
         et[1],
         et[2],
         es[0],
         es[1],
         es[2],
         true
      });

      TensorShape p;
      OP_REQUIRES_OK( context, context->GetAttr( "proj_shape", &p ) );

      std::vector<float> ds;
      OP_REQUIRES_OK( context, context->GetAttr( "det_spacing", &ds ) );

      float src_dist, det_dist;
      OP_REQUIRES_OK( context, context->GetAttr( "src_dist", &src_dist ) );
      OP_REQUIRES_OK( context, context->GetAttr( "det_dist", &det_dist ) );

      proj_cfg_.reset(new ProjectionCfg{
         static_cast<int>(p.dim_size( 0 )),
         ds[0],
         static_cast<int>(p.dim_size( 1 )),
         ds[1],
         src_dist,
         det_dist,
         0.0f,
         0.0f,
      });
   }

   void Compute(OpKernelContext* context) override
   {
      OP_REQUIRES_OK(context, Status(error::UNIMPLEMENTED, "CPU mode not implemented"));
   }

};


class ForwardCudaOp : public ForwardOp
{
public:
   explicit ForwardCudaOp(OpKernelConstruction* context) : ForwardOp(context)
   {
   }

   void Compute(OpKernelContext* context) override
   {
      // grab input 0: volume
      const auto volume = context->input(0).tensor<float, 5>();
      const int batch_size = volume.dimension(0);

      // grab input 1: angles
      const auto angles = context->input(1).tensor<float, 1>();
      proj_cfg_->n_angles = static_cast<int>(angles.dimension(0));

      const auto proj_shape = TensorShape{
          batch_size,
          1,
          proj_cfg_->n_angles,
          proj_cfg_->det_count_v,
          proj_cfg_->det_count_u
      };

      // create output
      Tensor* proj_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0,
               proj_shape, &proj_tensor));
      auto proj = proj_tensor->tensor<float, 5>();

      const auto device = context->device()->tensorflow_gpu_device_info()->gpu_id;

      radon_forward_cuda_3d(
        volume.data(), static_cast<const float*>(angles.data()), proj.data(), texture_cache_,
        *volume_cfg_, *proj_cfg_, defaultExecCfg(), batch_size, device);
   }
};

REGISTER_KERNEL_BUILDER(Name("Forward").Device(DEVICE_CPU), ForwardOp );
REGISTER_KERNEL_BUILDER(Name("Forward").Device(DEVICE_GPU), ForwardCudaOp);