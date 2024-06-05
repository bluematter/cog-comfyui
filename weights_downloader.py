import subprocess
import time
import os
from weights_manifest import WeightsManifest


class WeightsDownloader:
    supported_filetypes = [
        ".ckpt",
        ".safetensors",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".torchscript",
    ]

    def __init__(self):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map

    def get_weights_by_type(self, type):
        return self.weights_manifest.get_weights_by_type(type)

    def download_weights(self, weight_str):
        if weight_str in self.weights_map:
            if self.weights_manifest.is_non_commercial_only(weight_str):
                print(
                    f"⚠️  {weight_str} is for non-commercial use only. Unless you have obtained a commercial license.\nDetails: https://github.com/fofr/cog-comfyui/blob/main/weights_licenses.md"
                )
            self.download_if_not_exists(
                weight_str,
                self.weights_map[weight_str]["url"],
                self.weights_map[weight_str]["dest"],
            )
        else:
            raise ValueError(
                f"{weight_str} unavailable. View the list of available weights: https://github.com/fofr/cog-comfyui/blob/main/supported_weights.md"
            )

    def download_if_not_exists(self, weight_str, url, dest):
        if dest.endswith(weight_str):
            path_string = dest
        else:
            path_string = os.path.join(dest, weight_str)

        if not os.path.exists(path_string):
            self.download(weight_str, url, dest)

    def download(self, weight_str, url, dest):
        # Ensure the destination directory exists
        if "/" in weight_str:
            subfolder = weight_str.rsplit("/", 1)[0]
            dest = os.path.join(dest, subfolder)
        os.makedirs(dest, exist_ok=True)

        print(f"⏳ Downloading {weight_str} to {dest}")
        start = time.time()

        # Determine the file path
        file_name = os.path.basename(url)
        file_path = os.path.join(dest, file_name)

        # Using pget to download the file with adjusted chunk size and concurrency
        pget_command = [
            "pget", "--log-level", "warn", "-c", "10", "-m", "50M", "-xf", url, file_path
        ]
        subprocess.check_call(pget_command, close_fds=False)

        # Verify the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Untar the file if it is a tar file
        if file_path.endswith(".tar"):
            print(f"⏳ Extracting {file_path}")
            tar_command = [
                "tar", "-xvf", file_path, "-C", dest
            ]
            subprocess.check_call(tar_command, close_fds=False)

        elapsed_time = time.time() - start
        try:
            file_size_bytes = os.path.getsize(file_path)
            file_size_megabytes = file_size_bytes / (1024 * 1024)
            print(
                f"⌛️ Downloaded {weight_str} in {elapsed_time:.2f}s, size: {file_size_megabytes:.2f}MB"
            )
        except FileNotFoundError:
            print(f"⌛️ Downloaded {weight_str} in {elapsed_time:.2f}s")
