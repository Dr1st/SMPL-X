from flask import Flask, request, render_template, redirect, url_for
import subprocess
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_folder = request.form['model_folder']
        model_type = request.form['model_type']
        gender = request.form['gender']
        num_betas = request.form['num_betas']
        beta_values = request.form.getlist('beta_values')
        num_expression_coeffs = request.form['num_expression_coeffs']
        plotting_module = request.form['plotting_module']
        ext = request.form['ext']
        # plot_joints = request.form.get('plot_joints') == 'on'
        # sample_shape = request.form.get('sample_shape') == 'on'
        # sample_expression = request.form.get('sample_expression') == 'on'
        # use_face_contour = request.form.get('use_face_contour') == 'on'

        plot_joints = 'plot_joints' in request.form
        sample_shape = 'sample_shape' in request.form
        sample_expression = 'sample_expression' in request.form
        use_face_contour = 'use_face_contour' in request.form

        save_path = request.form['save_path']

        # Convert beta_values to a string of space-separated values
        beta_values = " ".join(beta_values)

        # Construct the command
        cmd = [
            'python', '.\examples\demo.py',
            '--model-folder', model_folder or 'D:\AvatarGen\SMPLGen\SMPL_Clean_Model', 
            '--model-type', model_type,
            '--gender', gender,
            '--num-betas', num_betas,
            '--num-expression-coeffs', num_expression_coeffs,
            '--plotting-module', plotting_module,
            '--ext', ext
            # '--save-path', save_path
        ]

        # If save path is not none, save the model
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

        return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)