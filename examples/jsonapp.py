from flask import Flask, request, render_template, redirect, url_for, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}), 400

        # Default model folder if not provided
        # model_folder = data.get('model_folder', 'D:\AvatarGen\SMPLGen\SMPL_Clean_Model')
        # model_type = data['model_type']
        # gender = data['gender']
        # num_betas = data['num_betas']
        # beta_values = data['beta_values']
        # num_expression_coeffs = data['num_expression_coeffs']
        # plotting_module = data['plotting_module']
        # ext = data['ext']
        # plot_joints = data['plot_joints'] == 'true'
        # sample_shape = data['sample_shape'] == 'true'
        # sample_expression = data['sample_expression'] == 'true'
        # use_face_contour = data['use_face_contour'] == 'true'
        # save_path = data.get('save_path')

        model_folder = 'D:\AvatarGen\SMPLGen\SMPL_Clean_Model'
        model_type = 'smpl'
        gender = data['gender']
        num_betas = '10'
        beta_values = data['beta_values']
        num_expression_coeffs = '10'
        plotting_module = 'pyrender'
        ext = 'npz'
        plot_joints = True
        sample_shape = True
        sample_expression = True
        use_face_contour = False
        save_path = data.get('save_path')


        # Construct the command
        cmd = [
            'python', './examples/demo.py',
            '--model-folder', model_folder,
            '--model-type', model_type,
            '--gender', gender,
            '--num-betas', num_betas,
            '--num-expression-coeffs', num_expression_coeffs,
            '--plotting-module', plotting_module,
            '--ext', ext
        ]

        if save_path:
            cmd.extend(['--save-path', save_path])

        if beta_values:
            cmd.extend(['--beta-values'] + beta_values.split())

        if plot_joints:
            cmd.append('--plot-joints')
        if sample_shape:
            cmd.append('--sample-shape')
        if sample_expression:
            cmd.append('--sample-expression')
        if use_face_contour:
            cmd.append('--use-face-contour')

        # Run the command
        subprocess.run(cmd)

        return jsonify({"status": "success"}), 200
    return render_template('jsonindex.html')

if __name__ == '__main__':
    app.run(debug=True)



# {
#   "model_folder": "D:/AvatarGen/SMPLGen/SMPL_Clean_Model",
#   "model_type": "smpl",
#   "gender": "neutral",
#   "num_betas": "10",
#   "beta_values": "0.0 0.0 -5.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
#   "num_expression_coeffs": "10",
#   "plotting_module": "pyrender",
#   "ext": "npz",
#   "plot_joints": "true",
#   "sample_shape": "true",
#   "sample_expression": "true",
#   "use_face_contour": "false"
# }

# {
#   "gender": "neutral",
#   "beta_values": "0.0 0.0 -5.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
# }