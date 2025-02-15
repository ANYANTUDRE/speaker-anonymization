from anonymization.pipelines.sttts_pipeline import STTTSPipeline
import torch
import yaml
from pathlib import Path

# Load datasets.yaml first
datasets_path = Path("configs/datasets.yaml")  # Adjust path if needed
if datasets_path.exists():
    with open(datasets_path, "r") as datasets_file:
        datasets_config = yaml.safe_load(datasets_file)
else:
    raise FileNotFoundError(f"Datasets file not found at {datasets_path}")

# Load main configuration file
config_path = Path("configs/anon/anon_ims_sttts_pc_whisper.yaml")
if config_path.exists():
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
else:
    raise FileNotFoundError(f"Config file not found at {config_path}")

# Merge datasets into config
print("Merge datasets into config")
config["datasets"] = datasets_config  # Replace the !include reference

# Set up device for computation
devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]

# Initialize and run the pipeline
pipeline = STTTSPipeline(config=config, force_compute=True, devices=devices, config_name="anon_ims_sttts_pc_whisper")
print("hellooooo...")

input_audio = "audio.mp3"
print("runninnng pipeline...")
anonymized_audio = pipeline.run_single_audio(input_audio)
print("Anonymized audio file is at:", anonymized_audio)
