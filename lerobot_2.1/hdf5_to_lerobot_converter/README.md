# HDF5 to LeRobot Dataset Converter

## 🚀 Purpose

Convert robotics datasets stored in **custom HDF5 formats** into the standardized [LeRobot](https://huggingface.co/lerobot) structure for Hugging Face Hub.

Designed for researchers who need:
- Flexible conversion without rigid schema requirements  
- Custom state/action processing  
- Batch support for large datasets  

---

## 📥 Installation (Conda)

1. **Download the `environment.yml`** from this repository.

2. **Create the Conda environment**:

   ```bash
   conda env create -f environment.yml
   conda activate lerobot
   ```

---

## ⚙️ Configuration

Create a `config.yaml` file like the following:

```yaml
repo_id: my-username/my-dataset-name
raw_data_dir: /path/to/hdf5/episodes
output_dir: /custom/output/path  # optional, default: ~/.cache/huggingface/lerobot/<repo_id>
fps: 30                           # optional, frames per second for video encoding
batch_size: 1                     # optional, number of episodes per dataset batch
```

---

## 🧪 Usage

Run the converter with your config file:

```bash
python convert_lerobot.py --config config.yaml
```

This will:

1. Load all files named like `episode_*.hdf5` from the `raw_data_dir`  
2. Decode and process states, actions, and images for each episode  
3. Save the data into Hugging Face-compatible `.lerobot` format in batches  
4. Generate videos for each camera stream  

---

## 📁 Output Structure

Output will be saved in:

```
~/.cache/huggingface/lerobot/<repo_id>/
```

Or under your specified `output_dir`. Each batch will include:
- `.lerobot` batch file
- video files per camera (e.g., `camera_high.mp4`)
- metadata for episodes and frame alignment

---

## 🧠 Assumptions & Defaults

This script assumes:
- **Episode filenames** are in the format: `episode_00001.hdf5`, `episode_00002.hdf5`, etc.
- HDF5 file structure includes:
  - `observations/qpos`: robot joint positions
  - `observations/images/camera_high` and `camera_wrist_right`: sequences of JPEG-compressed images
  - `task`: stored as an HDF5 file attribute
- **Actions** are derived by shifting the `qpos` sequence (i.e., `action[t] = qpos[t+1]`)
- Videos are saved per camera using `cv2.VideoWriter`

---

## 🛠️ Customization

To adapt the conversion for your own dataset format, modify the `LeRobotDatasetConverter` class:

- ✅ Add or remove cameras: update the `image_data` block in `process_episode`
- ⚙️ Adjust joint state/action dimensions: edit the `features` dictionary
- 📏 Customize batching: set a different `batch_size` in `config.yaml`
- 🔄 Modify action logic: replace the qpos-shift logic if your dataset provides real actions

---

## ☁️ Uploading to Hugging Face Hub

Once you've generated the dataset:

```bash
lerobot-cli push /path/to/output --repo-id my-username/my-dataset-name
```

This will upload your `.lerobot` files and videos to the Hub for public or private access via the LeRobot API.

---

## 🐛 Troubleshooting

- **No episode files found**  
  Ensure the directory `raw_data_dir` contains files named `episode_*.hdf5`.

- **ValueError: Cannot divide X episodes into batches of Y**  
  Your `batch_size` must evenly divide the number of HDF5 episodes.

- **Error processing [filename]**  
  Possible corrupted image buffers or missing fields in the HDF5 structure. Check logs for stack traces.

- **Video writing errors**  
  Ensure OpenCV is installed with FFmpeg support (`opencv-python` from pip works in most cases).

---

## 📦 Dependencies

Required packages include:
- `lerobot`
- `h5py`
- `opencv-python`
- `PyYAML`
- `numpy`
- `tqdm`

These are included in the provided `environment.yml`.

---

## 📄 License

MIT License. See the `LICENSE` file for more details.
