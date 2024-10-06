from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SINN Class for Seismic Data Analysis
class SINN:
    def __init__(self, rel_time, data):
        self.rel_time = rel_time
        self.data = data
        abs_data = data.abs()
        mean = float(10 * abs_data.mean())
        abs_data = abs_data.transform(lambda x: x - mean / 5)
        mean /= 1.5
        self.mean = mean
        self.abs_data = abs_data
        self.on_time = int(len(self.rel_time) / 143)

    def plot(self, filename):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(self.rel_time, self.data, label="Velocity (m/s)")
        ax1.plot(self.rel_time, [self.mean for _ in range(len(self.rel_time))])
        ax1.axhline(self.mean, color='green', linestyle='--')

        on_off = True
        for i in range(len(self.abs_data)):
            if on_off:
                if float(self.abs_data[i]) >= self.mean:
                    bool_list = [float(j) >= self.mean for j in self.abs_data[i:i+self.on_time]]
                    if sum(bool_list) / len(bool_list) >= 0.65:
                        ax1.axvline(self.rel_time[i], color='red')
                        ax2.axvline(self.rel_time[i], color='red')
                        on_off = False
            else:
                if float(self.abs_data[i]) <= self.mean:
                    bool_list = [float(j) <= self.mean for j in self.abs_data[i:i+3000]]
                    if all(bool_list):
                        ax1.axvline(self.rel_time[i], color='green')
                        ax2.axvline(self.rel_time[i], color='green')
                        on_off = True

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.legend()

        f, t, Sxx = signal.spectrogram(self.data, fs=1/(self.rel_time[1] - self.rel_time[0]))

        pcm = ax2.pcolormesh(t, f, np.log10(Sxx), shading='auto', cmap='inferno')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [s]')
        fig.colorbar(pcm, ax=ax2, label='Power [(m/s)^2/sqrt(Hz)]')

        plt.savefig(f'static/{filename}')
        plt.close()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the CSV with your SINN class
        df = pd.read_csv(filepath)
        times = df["time_rel(sec)"]
        data = df["velocity(m/s)"]

        # Generate the plot and save it
        sinn = SINN(times, data)
        plot_filename = 'seismic_output.png'
        sinn.plot(plot_filename)

        return render_template('result.html', plot_url=plot_filename)

if __name__ == "__main__":
    app.run(debug=True)
