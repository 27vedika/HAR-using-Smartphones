import pickle
from plyer import accelerometer, gyroscope
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import butter, welch, filtfilt, lfilter
import math
from scipy.fft import fft
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import time
import warnings
import json
warnings.filterwarnings('ignore')


# Load pre-trained models and scalers using pickle4
with open("/storage/emulated/0/ml_project/scaler_pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("/storage/emulated/0/ml_project/pca_pkl", "rb") as pca_file:
    pca = pickle.load(pca_file)

with open("/storage/emulated/0/ml_project/svm_pkl", "rb") as model_file:
    ml_model = pickle.load(model_file)

# The rest of the code (e.g., Butterworth filter, feature extraction, and Kivy app logic) remains unchanged.
# See the previously provided example for the full implementation.


# # Butterworth filter helper functions
# def butter_lowpass(cutoff, fs, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype="low", analog=False)
#     return b, a

# def butter_filter(data, cutoff, fs, order=4):
#     b, a = butter_lowpass(cutoff, fs, order)
#     y = lfilter(b, a, data)
#     return y

# # Time-domain feature extraction
# def extract_time_domain_features(data):
#     features = [
#         np.mean(data),
#         np.std(data),
#         np.min(data),
#         np.max(data),
#         skew(data),
#         kurtosis(data),
#         np.sqrt(np.mean(np.square(data))),  # RMS
#         np.median(np.abs(data - np.median(data))),  # MAD
#         np.corrcoef(data[:-1], data[1:])[0, 1],  # Autocorrelation
#     ]
#     return features

# # Frequency-domain feature extraction
# def extract_frequency_domain_features(data, fs):
#     f, Pxx = welch(data, fs, nperseg=128)
#     energy = np.sum(Pxx)  # Signal energy
#     spectral_entropy = -np.sum(Pxx * np.log(Pxx + 1e-10))  # Spectral entropy
#     peak_frequency = f[np.argmax(Pxx)]  # Peak frequency
#     return [energy, spectral_entropy, peak_frequency]

# # Feature extraction from sliding window
# def extract_features_from_window(window_data, fs=50):
#     features = []
#     cutoff_frequency = 0.3  # Gravity cutoff frequency

#     # Process accelerometer data
#     for axis in ["accel_x", "accel_y", "accel_z"]:
#         signal = window_data[axis]
#         gravity_component = butter_filter(signal, cutoff=cutoff_frequency, fs=fs)
#         body_motion = signal - gravity_component
#         features.extend(
#             extract_time_domain_features(body_motion)
#             + extract_frequency_domain_features(body_motion, fs)
#         )

#     # Process gyroscope data
#     for axis in ["gyro_x", "gyro_y", "gyro_z"]:
#         signal = window_data[axis]
#         features.extend(
#             extract_time_domain_features(signal)
#             + extract_frequency_domain_features(signal, fs)
#         )

#     return features

# Function to apply a Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=0):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


# Function to calculate features
def calculate_features(signal):
    features = {}
    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)
    features["mad"] = np.median(np.abs(signal - np.median(signal)))
    features["max"] = np.max(signal)
    features["min"] = np.min(signal)
    features["sma"] = np.sum(np.abs(signal)) / len(signal)
    features["energy"] = np.sum(signal**2) / len(signal)
    features["iqr"] = np.percentile(signal, 75) - np.percentile(signal, 25)
    features["entropy"] = entropy(signal)
    features["arCoeff"] = np.polyfit(np.arange(len(signal)), signal, 4)
    features["skewness"] = skew(signal)
    features["kurtosis"] = kurtosis(signal)
    return features


# Function to calculate frequency domain features
def calculate_frequency_features(signal, fs):
    f, Pxx = welch(signal, fs=fs)
    features = {}
    features["mean"] = np.mean(Pxx)
    features["meanFreq"] = np.sum(f * Pxx) / np.sum(Pxx)
    features["skewness"] = skew(Pxx)
    features["kurtosis"] = kurtosis(Pxx)
    features["bandsEnergy"] = [np.sum(Pxx[i : i + 8]) for i in range(0, len(Pxx), 8)]
    features['std'] = np.std(Pxx)
    features['mad'] = np.median(np.abs(Pxx - np.median(Pxx)))  # Calculate MAD in the frequency domain
    features['max'] = np.max(Pxx)  # Calculate max in the frequency domain
    features['min'] = np.min(Pxx)  # Calculate min in the frequency domain
    features["sma"] = np.sum(np.abs(Pxx)) / len(Pxx)
    features["energy"] = np.sum(Pxx**2) / len(Pxx)
    features['iqr'] = np.percentile(Pxx, 75) - np.percentile(Pxx, 25)  # Calculate IQR in the frequency domain
    features['entropy'] = entropy(Pxx)  # Calculate entropy in the frequency domain
    features['maxInds'] = np.argmax(Pxx)
    return features

def extract_features_from_window(window_data, ob=None):
    # self.window_data["accel_x"].append(accel_x)
    # self.window_data["accel_y"].append(accel_y)
    # self.window_data["accel_z"].append(accel_z)
    # self.window_data["gyro_x"].append(gyro_x)
    # self.window_data["gyro_y"].append(gyro_y)
    # self.window_data["gyro_z"].append(gyro_z)

    tAcc_XYZ, tGyro_XYZ = [], []

    for key in window_data:
        sum = 0
        n = 0
        mean = 0
        for i in window_data[key]:
            if not(i is None or math.isnan(i) or np.isneginf(i) or np.isposinf(i)):
                sum += i
                n += 1
        
        if n > 0:
            mean = sum / n

        for i in range(len(window_data[key])):
            if window_data[key][i] is None or math.isnan(window_data[key][i]) or np.isneginf(window_data[key][i]) or np.isposinf(window_data[key][i]):
                window_data[key][i] = mean

    tAcc_XYZ.append(window_data["accel_x"])
    tAcc_XYZ.append(window_data["accel_y"])
    tAcc_XYZ.append(window_data["accel_z"])
    tGyro_XYZ.append(window_data["gyro_x"])
    tGyro_XYZ.append(window_data["gyro_y"])
    tGyro_XYZ.append(window_data["gyro_z"])

    # ob.setLabel(str(len(tAcc_XYZ[0])) + " " + str(len(tGyro_XYZ[0])))
    # time.sleep(2)
    print(str(len(tAcc_XYZ[0])) + " " + str(len(tGyro_XYZ[0])))

    # Preprocess the signals
    fs = 50  # Sampling frequency
    tAcc_XYZ_filtered = np.apply_along_axis(
        butter_lowpass_filter, 0, tAcc_XYZ, cutoff=20, fs=fs
    )
    tGyro_XYZ_filtered = np.apply_along_axis(
        butter_lowpass_filter, 0, tGyro_XYZ, cutoff=20, fs=fs
    )

    # Separate body and gravity acceleration signals
    tBodyAcc_XYZ = np.apply_along_axis(
        butter_lowpass_filter, 0, tAcc_XYZ_filtered, cutoff=0.3, fs=fs
    )
    tGravityAcc_XYZ = tAcc_XYZ_filtered - tBodyAcc_XYZ

    # Derive Jerk signals
    tBodyAccJerk_XYZ = np.diff(tBodyAcc_XYZ, axis=0)
    tBodyGyroJerk_XYZ = np.diff(tGyro_XYZ_filtered, axis=0)

    # Calculate magnitudes
    tBodyAccMag = np.linalg.norm(tBodyAcc_XYZ, axis=1)
    tGravityAccMag = np.linalg.norm(tGravityAcc_XYZ, axis=1)
    tBodyAccJerkMag = np.linalg.norm(tBodyAccJerk_XYZ, axis=1)
    tBodyGyroMag = np.linalg.norm(tGyro_XYZ_filtered, axis=1)
    tBodyGyroJerkMag = np.linalg.norm(tBodyGyroJerk_XYZ, axis=1)

    # ob.setLabel("Before FFT")
    print("Before FFT")

    # Apply FFT
    fBodyAcc_XYZ = np.apply_along_axis(fft, 0, tBodyAcc_XYZ, n=128)
    fBodyAccJerk_XYZ = np.apply_along_axis(fft, 0, tBodyAccJerk_XYZ, n=128)
    fBodyGyro_XYZ = np.apply_along_axis(fft, 0, tGyro_XYZ_filtered, n=128)
    fBodyAccMag = fft(tBodyAccMag, n=128)
    fBodyAccJerkMag = fft(tBodyAccJerkMag, n = 128)
    fBodyGyroMag = fft(tBodyGyroMag, n=128)
    fBodyGyroJerkMag = fft(tBodyGyroJerkMag, n=128)

    # ob.setLabel("After FFT")
    print("After FFT")

    # Extract features
    features = {}
    for axis in ["X", "Y", "Z"]:
        features[f"tBodyAcc-{axis}"] = calculate_features(
            tBodyAcc_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]]
        )
        features[f"tGravityAcc-{axis}"] = calculate_features(
            tGravityAcc_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]]
        )
        features[f"tBodyAccJerk-{axis}"] = calculate_features(
            tBodyAccJerk_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]]
        )
        features[f"tBodyGyro-{axis}"] = calculate_features(
            tGyro_XYZ_filtered[:, {"X": 0, "Y": 1, "Z": 2}[axis]]
        )
        features[f"tBodyGyroJerk-{axis}"] = calculate_features(
            tBodyGyroJerk_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]]
        )
        features[f"fBodyAcc-{axis}"] = calculate_frequency_features(
            fBodyAcc_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]], fs
        )
        features[f"fBodyAccJerk-{axis}"] = calculate_frequency_features(
            fBodyAccJerk_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]], fs
        )
        features[f"fBodyGyro-{axis}"] = calculate_frequency_features(
            fBodyGyro_XYZ[:, {"X": 0, "Y": 1, "Z": 2}[axis]], fs
        )

    features["tBodyAccMag"] = calculate_features(tBodyAccMag)
    features["tGravityAccMag"] = calculate_features(tGravityAccMag)
    features["tBodyAccJerkMag"] = calculate_features(tBodyAccJerkMag)
    features["tBodyGyroMag"] = calculate_features(tBodyGyroMag)
    features["tBodyGyroJerkMag"] = calculate_features(tBodyGyroJerkMag)
    features["fBodyAccMag"] = calculate_frequency_features(fBodyAccMag, fs)
    features["fBodyAccJerkMag"] = calculate_frequency_features(fBodyAccJerkMag, fs)
    features["fBodyGyroMag"] = calculate_frequency_features(fBodyGyroMag, fs)
    features["fBodyGyroJerkMag"] = calculate_frequency_features(fBodyGyroJerkMag, fs)

    # Calculate additional vectors
    gravityMean = np.mean(tGravityAcc_XYZ, axis=0)
    tBodyAccMean = np.mean(tBodyAcc_XYZ, axis=0)
    tBodyAccJerkMean = np.mean(tBodyAccJerk_XYZ, axis=0)
    tBodyGyroMean = np.mean(tGyro_XYZ_filtered, axis=0)
    tBodyGyroJerkMean = np.mean(tBodyGyroJerk_XYZ, axis=0)

    # ob.setLabel("Before Angle calc")
    print("Before Angle calc")

    # Calculate angles
    features["angle(tBodyAccMean,gravity)"] = np.arccos(
        np.dot(tBodyAccMean, gravityMean)
        / (np.linalg.norm(tBodyAccMean) * np.linalg.norm(gravityMean))
    )
    features["angle(tBodyAccJerkMean,gravityMean)"] = np.arccos(
        np.dot(tBodyAccJerkMean, gravityMean)
        / (np.linalg.norm(tBodyAccJerkMean) * np.linalg.norm(gravityMean))
    )
    features["angle(tBodyGyroMean,gravityMean)"] = np.arccos(
        np.dot(tBodyGyroMean, gravityMean)
        / (np.linalg.norm(tBodyGyroMean) * np.linalg.norm(gravityMean))
    )
    features["angle(tBodyGyroJerkMean,gravityMean)"] = np.arccos(
        np.dot(tBodyGyroJerkMean, gravityMean)
        / (np.linalg.norm(tBodyGyroJerkMean) * np.linalg.norm(gravityMean))
    )
    features["angle(X,gravityMean)"] = np.arccos(
        gravityMean[0] / np.linalg.norm(gravityMean)
    )
    features["angle(Y,gravityMean)"] = np.arccos(
        gravityMean[1] / np.linalg.norm(gravityMean)
    )
    features["angle(Z,gravityMean)"] = np.arccos(
        gravityMean[2] / np.linalg.norm(gravityMean)
    )

    # ob.setLabel("After Angle calc")
    print("After Angle Calc")

    # Organize features into a NumPy array in the order specified by features.txt
    from final_features import feature_list

    # Create a NumPy array to store the features in the specified order
    feature_array = np.zeros(len(feature_list))

    # # Populate the feature array
    for i, feature_name in enumerate(feature_list):
        print(feature_name)
        if "BodyBody" in feature_name:
            feature_name = feature_name[:5] + feature_name[9:]

        if "arCoeff" in feature_name:
            try:
                if "," in feature_name.split("arCoeff()-")[1]:
                    signal, axis_coeff = feature_name.split("arCoeff()-")
                    signal, axis_coeff = signal.strip('-'), axis_coeff.strip('-')
                    axis, coeff_idx = axis_coeff.split(",")
                    coeff_idx = int(coeff_idx) - 1
                    feature_array[i] = features[f"{signal}-{axis}"]["arCoeff"][coeff_idx]
            except IndexError:
                signal, coeff_idx = feature_name.split("arCoeff()")
                signal = signal.strip('-')
                coeff_idx = int(coeff_idx) - 1
                feature_array[i] = features[signal]["arCoeff"][coeff_idx]

        elif "correlation" in feature_name:
            axes = feature_name.split("correlation()-")[1].split(",")
            feature_array[i] = np.corrcoef(
                features[f"tBodyAcc-{axes[0]}"]["mean"],
                features[f"tBodyAcc-{axes[1]}"]["mean"],
            )[0, 1]
        elif "bandsEnergy" in feature_name:
            signal, band = feature_name.split("bandsEnergy()-")
            signal = signal.strip('-')
            band_start, band_end = map(int, band.split(","))
            if signal.startswith('f'):
                feature_array[i] = (
                    np.sum(features[f"{signal}-X"]["bandsEnergy"][band_start:band_end]) +
                    np.sum(features[f"{signal}-Y"]["bandsEnergy"][band_start:band_end]) +
                    np.sum(features[f"{signal}-Z"]["bandsEnergy"][band_start:band_end])
                ) / 3
            else:
                feature_array[i] = np.sum(features[signal]["bandsEnergy"][band_start:band_end])

        elif "angle" in feature_name:
            feature_array[i] = features[feature_name]

        else:
            parts = feature_name.split("-")
            if len(parts) == 2:
                signal, metric = parts
                if metric == "sma()":
                    if signal in ["tBodyAccMag", "tGravityAccMag", "tBodyAccJerkMag", "tBodyGyroMag", "tBodyGyroJerkMag"]:
                        feature_array[i] = features[signal][metric[:-2]]
                    elif signal in ["fBodyAccMag", "fGravityAccMag", "fBodyAccJerkMag", "fBodyGyroMag", "fBodyGyroJerkMag"]:
                        feature_array[i] = np.abs(features[signal]["meanFreq"])
                    else:
                        feature_array[i] = (
                            features[f"{signal}-X"][metric[:-2]]
                            + features[f"{signal}-Y"][metric[:-2]]
                            + features[f"{signal}-Z"][metric[:-2]]
                        ) / 3
                else:
                    feature_array[i] = features[signal][metric[:-2] if metric[-2:] == '()' else metric]
            else:
                signal = "-".join(parts[:-2])
                metric = parts[-2]
                axis = parts[-1]
                feature_array[i] = features[signal + "-" + axis][metric[:-2] if metric[-2:] == '()' else metric]


    # # Print the feature array
    print(len(feature_list))
    print(len(feature_array))

    valid_values = feature_array[~np.isnan(feature_array) & ~np.isinf(feature_array)]

    mean = np.nanmean(valid_values)
    print(mean)
    for i in range(len(feature_array)):
        if feature_array[i] is None or math.isnan(feature_array[i]) or np.isneginf(feature_array[i]) or np.isposinf(feature_array[i]):
            feature_array[i] = mean
    
    # feature_array = np.nan_to_num(feature_array, nan=mean)
    # print(mean)
    return feature_array


# Main Kivy App
class ActivityRecognitionApp(App):
    def setLabel(self, prompt):
        self.label.text = prompt

    def build(self):
        self.is_collecting = False
        self.window_data = {"accel_x": [], "accel_y": [], "accel_z": [], "gyro_x": [], "gyro_y": [], "gyro_z": []}
        self.layout = BoxLayout(orientation="vertical")
        self.label = Label(text="Press 'Start' to begin recording.")
        self.start_button = Button(text="Start", on_press=self.start_recording)
        self.stop_button = Button(text="Stop", on_press=self.stop_recording)
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.start_button)
        self.layout.add_widget(self.stop_button)
        return self.layout

    def start_recording(self, instance):
        self.is_collecting = True
        self.label.text = "Recording... Collecting data."
        accelerometer.enable()
        gyroscope.enable()  # Enable gyroscope
        Clock.schedule_interval(self.collect_data, 0.05)  # 50 Hz

    def stop_recording(self, instance):
        f = open("sensor_data.txt", "w")
        json.dump(self.window_data, f)
        f.close()

        self.is_collecting = False
        self.label.text = "Stopped recording. Processing data."
        Clock.unschedule(self.collect_data)
        accelerometer.disable()
        gyroscope.disable()  # Disable gyroscope

        if len(self.window_data["accel_x"]) >= 128:
            features = extract_features_from_window(self.window_data, self)
            normalized_features = scaler.transform([features])  # Normalize
            reduced_features = pca.transform(normalized_features)  # Dimensionality reduction
            prediction = ml_model.predict(reduced_features)[0]  # Predict activity
            self.label.text = f"Predicted Activity: {prediction}"
            
            targets = ["STANDING", "WALKING_UPSTAIRS", "WALKING", "WALKING_DOWNSTAIRS", "SITTING", "LAYING"]
            predicted_activity = targets[prediction]
            self.label.text = f"Predicted Activity: {predicted_activity}"

            f = open("classification.txt", 'w')
            f.write(f"Predicted Activity: {prediction}")
            f.close()

    def collect_data(self, dt):
        if not self.is_collecting:
            return

        try:
            accel_data = accelerometer.acceleration
            gyro_data = gyroscope.rotation

            if accel_data is None or gyro_data is None:
                return

            accel_x, accel_y, accel_z = accel_data
            gyro_x, gyro_y, gyro_z = gyro_data

            # Append data to sliding window
            self.window_data["accel_x"].append(accel_x)
            self.window_data["accel_y"].append(accel_y)
            self.window_data["accel_z"].append(accel_z)
            self.window_data["gyro_x"].append(gyro_x)
            self.window_data["gyro_y"].append(gyro_y)
            self.window_data["gyro_z"].append(gyro_z)

            # Maintain window size of 128 samples
            # for key in self.window_data:
            #     if len(self.window_data[key]) > 128:
            #         self.window_data[key].pop(0)

            # # If enough data is collected, process it
            #if len(self.window_data["accel_x"]) == 128:
                #features = extract_features_from_window(self.window_data, self)
               # features = features.reshape(1,-1)
              #  normalized_features = scaler.transform([features])  # Normalize
              #  reduced_features = pca.transform(normalized_features)  # Dimensionality reduction
                #prediction = ml_model.predict(reduced_features)[0]  # Predict activity
                

        except Exception as e:
            self.label.text = f"Error: {e}"

# Run the app
if __name__ == "__main__":
    ActivityRecognitionApp().run()

    # import json

    # f = open("sensor_data.txt")
    # data = json.load(f)
    # f.close()

    # feature_vector = (extract_features_from_window(data))
    # feature_vector = feature_vector.reshape(1,-1)
    # normalized_features = scaler.transform(feature_vector)  # Normalize
    # reduced_features = pca.transform(normalized_features)  # Dimensionality reduction
    # prediction = ml_model.predict(reduced_features)[0]  # Predict activity
    # label = f"Predicted Activity: {prediction}"
    # print(label)
