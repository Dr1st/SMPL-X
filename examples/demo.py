# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


# Parameters Meaning

# Beta 0 (Overall Body Size):
# Range: -5 to 5
# Effect: -5 for fatter, +5 for thinner and smaller

# Beta 1 (Side Compression and Stretching):
# Range: 0 to positive values
# Effect: Positive values make the body fatter

# Beta 2 (General Body Fat):
# Range: 0 to positive values
# Effect: Positive values make the body fatter

# Beta 3 (Belly Size and Body Reduction):
# Range: 0 to negative values
# Effect: Negative values make the belly much bigger and the rest of the body smaller

# Beta 4 (Chest, Hip, and Abdomen Size):
# Range: -5 to 5
# Effect: -5 for smaller, +5 for larger chest, hips, and abdomen

# Beta 5 (Big Belly with Overall Thinning):
# Range: 0 to negative values
# Effect: Negative values indicate a big belly with overall body thinning

# Beta 6 (Disproportionate Belly Size):
# Range: 0 to positive values
# Effect: Positive values indicate a particularly large belly, while other parts of the body become very thin

# Beta 7 (Vertical Body Compression):
# Range: 0 to positive values
# Effect: Positive values indicate vertical squeezing of the body

# Beta 8 (Horizontal Body Fattening):
# Range: 0 to positive values
# Effect: Positive values indicate horizontal fattening of the body

# Beta 9 (Shoulder Width):
# Range: 0 to positive values
# Effect: Positive values indicate wider shoulders



import os.path as osp
import argparse

import numpy as np
import torch

import smplx
import trimesh


def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         beta_values = None,     # Body shape parameters' values
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False,
         save_path = None):

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    print(model)

    betas, expression = None, None

    print(beta_values)

    if beta_values is not None:
        betas = torch.tensor(beta_values, dtype=torch.float32)       # Values for body shape parameters
    elif sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)

    
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)
        
    print(betas)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    # Save the model
    # print(trimesh.__version__)
    if save_path is not None:
        saved_model = trimesh.Trimesh(vertices, model.faces)
        saved_model.export(save_path)
        print(f"Model saved to {save_path}")


    # if plotting_module == 'pyrender':
    #     import pyrender
    #     # import trimesh
    #     vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    #     tri_mesh = trimesh.Trimesh(vertices, model.faces,
    #                                vertex_colors=vertex_colors)

    #     mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    #     scene = pyrender.Scene()
    #     scene.add(mesh)

    #     if plot_joints:
    #         sm = trimesh.creation.uv_sphere(radius=0.005)
    #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    #         tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    #         tfs[:, :3, 3] = joints
    #         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    #         scene.add(joints_pcl)

    #     pyrender.Viewer(scene, use_raymond_lighting=True)
    # elif plotting_module == 'matplotlib':
    #     from matplotlib import pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #     from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
    #     face_color = (1.0, 1.0, 0.9)
    #     edge_color = (0, 0, 0)
    #     mesh.set_edgecolor(edge_color)
    #     mesh.set_facecolor(face_color)
    #     ax.add_collection3d(mesh)
    #     ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

    #     if plot_joints:
    #         ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
    #     plt.show()
    # elif plotting_module == 'open3d':
    #     import open3d as o3d

    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(
    #         vertices)
    #     mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    #     mesh.compute_vertex_normals()
    #     mesh.paint_uniform_color([0.3, 0.3, 0.3])

    #     geometry = [mesh]
    #     if plot_joints:
    #         joints_pcl = o3d.geometry.PointCloud()
    #         joints_pcl.points = o3d.utility.Vector3dVector(joints)
    #         joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
    #         geometry.append(joints_pcl)

    #     o3d.visualization.draw_geometries(geometry)
    # else:
    #     raise ValueError('Unknown plotting_module: {}'.format(plotting_module))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    
    parser.add_argument('--beta-values', default=None, nargs='+', type=float,
                        help='Specific beta values to use for the model shape.')        # Body shape arguments

    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    
    parser.add_argument('--save-path', default=None, type=str,  # Add this argument
                        help='The path to save the generated model')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    save_path = args.save_path              #Get the save path

    beta_values = args.beta_values
    if beta_values is not None:
        beta_values = np.array(beta_values).reshape(1, -1)          # Values for body shape parameters

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         beta_values=beta_values,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour,
         save_path = save_path)
