# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle as pkl
from fairmotion.core import motion as motion_class
from fairmotion.utils import constants
from fairmotion.ops import conversions
from scipy.spatial.transform import Rotation as R
from .bvh import load as load_bvh

NR_JOINTS = 22
PARENTS = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
JOINTS = [
    'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe', 
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe', 
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
]
JOINT_MAPPING = {i: x for i, x in enumerate(JOINTS)}

# this are the offsets stored under `J` in the SMPL model pickle file
OFFSETS = np.array([      
       [-2.223e+02,  9.246e+01,  3.630e+02],
       [ 1.035e-01,  1.858e+00,  1.055e+01],
       [ 4.350e+01, -2.700e-05, -2.000e-06],
       [ 4.237e+01, -1.100e-05, -1.000e-05],
       [ 1.730e+01,  1.000e-06,  4.000e-06],
       [ 1.035e-01,  1.858e+00, -1.055e+01],
       [ 4.350e+01, -3.100e-05,  1.500e-05],
       [ 4.237e+01, -1.900e-05,  1.000e-05],
       [ 1.730e+01, -4.000e-06,  6.000e-06],
       [ 6.902e+00, -2.604e+00,  6.000e-06],
       [ 1.259e+01, -2.000e-06,  2.000e-06],
       [ 1.234e+01,  1.300e-05, -1.400e-05],
       [ 2.583e+01, -1.700e-05, -2.000e-06],
       [ 1.177e+01,  1.900e-05,  5.000e-06],
       [ 1.975e+01, -1.480e+00,  6.000e+00],
       [ 1.128e+01,  4.000e-06, -2.800e-05],
       [ 3.300e+01,  6.000e-06,  3.300e-05],
       [ 2.520e+01, -7.000e-06, -7.000e-06],
       [ 1.975e+01, -1.480e+00, -6.000e+00],
       [ 1.128e+01, -2.900e-05, -2.300e-05],
       [ 3.300e+01,  1.300e-05,  1.000e-05],
       [ 2.520e+01,  1.620e-04,  4.380e-04]])


def load(
    file,
    motion=None,
    scale=1.0,
    load_skel=True,
    load_motion=True,
    rotvec=True,
    v_up_skel=np.array([0.0, 1.0, 0.0]),
    v_face_skel=np.array([0.0, 0.0, 1.0]),
    v_up_env=np.array([0.0, 1.0, 0.0]),
):
    if not motion:
        motion = motion_class.Motion(fps=60)

    if load_skel:
        skel = motion_class.Skeleton(
            v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env,
        )
        for joint_name, parent_joint, offset in zip(
            JOINTS, PARENTS, OFFSETS
        ):
            joint = motion_class.Joint(name=joint_name)
            if parent_joint == -1:
                parent_joint_name = None
                joint.info["dof"] = 6  # root joint is free
                offset -= offset
            else:
                parent_joint_name = JOINTS[parent_joint]
            T1 = conversions.p2T(scale * offset)
            joint.xform_from_parent_joint = T1
            skel.add_joint(joint, parent_joint_name)
        motion.skel = skel
    else:
        assert motion.skel is not None

    if load_motion:
        assert motion.skel is not None
        # Assume 60fps
        motion.set_fps(60.0)
        dt = float(1 / motion.fps)

        hips = None
        _scale = None
        if isinstance(file, str) and file.endswith("pkl"):
            with open(file, "rb") as f:
                data = pkl.load(f, encoding="latin1")
                poses = np.array(data["smpl_poses"])  # shape (seq_length, 135)
                assert len(poses) > 0, "file is empty"
                
                smpl_poses = poses.reshape(-1, 3)
                poses = R.from_rotvec(smpl_poses).as_matrix()
                poses = poses.reshape((-1, NR_JOINTS, 3, 3))
                
                hips = data["smpl_trans"]
                _scale = data["smpl_scaling"]
        elif isinstance(file, str) and file.endswith("npy"):
            poses = np.load(file)
        elif isinstance(file, np.ndarray):
            poses = file
        else:
            raise NotImplementedError

        for pose_id, pose in enumerate(poses):
            pose_data = [
                constants.eye_T() for _ in range(len(JOINTS))
            ]

            for joint_id, joint_name in enumerate(JOINTS):
                pose_data[
                    motion.skel.get_index_joint(joint_name)
                ] = conversions.R2T(pose[joint_id])
            motion.add_one_frame(pose_data)
            
            if hips is not None and _scale is not None:
                hips_pos = np.ones((1, 4)) * _scale
                hips_pos[:, :-1] = hips[pose_id] * scale
                T = np.eye(4)
                T[:, -1] = hips_pos / _scale
                motion.poses[pose_id].set_root_transform(T, local=True)
    return motion


def save():
    raise NotImplementedError("Using bvh.save() is recommended")
