import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import pathlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.type_check import imag
import imageio
from .utils import fig2data
 
def plot(pose_frame, parents, frame_idx, save_path_dir=None, color='b', dpi=200, xlim=None, ylim=None, zlim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, p in enumerate(parents):
        if i > 0:
            ax.plot(
                [pose_frame[i, 0], pose_frame[p, 0]],
                [pose_frame[i, 2], pose_frame[p, 2]],
                [pose_frame[i, 1], pose_frame[p, 1]],
                c=color,
            )
 
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    plt.draw()

    title = f"frameID: {frame_idx}"
    plt.title(title)

    if save_path_dir is None:
        plt.close()
        return fig
    else:
        pathlib.Path(save_path_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path_dir, f"frame_{frame_idx}.png"), dpi=dpi)
        plt.close()

def plot_sequence(pose_sequence, parents, gif_path, duration=0.1):
    plt_list = []
    max_axis = np.max(np.max(pose_sequence, axis=0), axis=0)
    min_axis = np.min(np.min(pose_sequence, axis=0), axis=0)
    for idx, pose_frame in enumerate(pose_sequence):
        fig = plot(pose_frame, parents=parents, frame_idx=idx, 
                   xlim=(min_axis[0], max_axis[0]),
                   ylim=(min_axis[2], max_axis[2]),
                   zlim=(min_axis[1], max_axis[1]))
        plt_list.append(fig2data(fig))
    imageio.mimsave(gif_path, plt_list, duration=duration)


if __name__ == "__main__":
    #一个示例
    PARENTS = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 
            11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    seq = np.load("pred_poses_all.npy")[7] #[50, 22, 3]
    plot_sequence(seq, PARENTS, "test_seq.gif")