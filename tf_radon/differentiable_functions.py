import tensorflow as tf
from tensorflow.python.framework import ops
import os
from os.path import join as pjoin

_path = os.path.dirname(os.path.abspath(__file__))
_cb_module = tf.load_op_library(pjoin(_path, '..', 'libconebeam.so'))
backprojection = _cb_module.backprojection
forward = _cb_module.forward


'''
    Compute the gradient of the backprojection op
    by invoking the forward projector.
'''
@ops.RegisterGradient("Backprojection")
def _backprojection_grad(op, grad):
    proj = forward(
        volume = grad,
        angles = op.inputs[1],
        vol_shape = op.get_attr("vol_shape"),
        proj_shape = op.get_attr("proj_shape"),
        det_spacing = op.get_attr("det_spacing"),
        src_dist = op.get_attr("src_dist"),
        det_dist = op.get_attr("det_dist"),
        vol_origin = op.get_attr("vol_origin"),
        voxel_dimen = op.get_attr("voxel_dimen"),
    )
    return [proj, None]


'''
    Compute the gradient of the forward projection op
    by invoking the backprojector.
'''
@ops.RegisterGradient("Forward")
def _forward_grad(op, grad):
    vol = backprojection(
        projections = grad,
        angles = op.inputs[1],
        vol_shape = op.get_attr("vol_shape"),
        proj_shape = op.get_attr("proj_shape"),
        det_spacing = op.get_attr("det_spacing"),
        src_dist = op.get_attr("src_dist"),
        det_dist = op.get_attr("det_dist"),
        vol_origin = op.get_attr("vol_origin"),
        voxel_dimen = op.get_attr("voxel_dimen"),
    )
    return [vol, None]
