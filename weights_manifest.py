import subprocess
import time
import os
import json
import custom_node_helpers as helpers
from config import config

UPDATED_WEIGHTS_MANIFEST_URL = (
    "https://raw.githubusercontent.com/fofr/cog-comfyui/main/weights.json"
)

UPDATED_WEIGHTS_MANIFEST_PATH = "updated_weights.json"
WEIGHTS_MANIFEST_PATH = "weights.json"
BASE_URL = config["WEIGHTS_BASE_URL"]
BASE_PATH = "ComfyUI/models"


class WeightsManifest:
    @staticmethod
    def base_url():
        return BASE_URL

    def __init__(self):
        self.download_latest_weights_manifest = (
            os.getenv("DOWNLOAD_LATEST_WEIGHTS_MANIFEST",
                      "false").lower() == "true"
        )
        self.weights_manifest = self._load_weights_manifest()
        self.weights_map = self._initialize_weights_map()

    def _load_weights_manifest(self):
        if self.download_latest_weights_manifest:
            self._download_updated_weights_manifest()
        return self._merge_manifests()

    def _download_updated_weights_manifest(self):
        if not os.path.exists(UPDATED_WEIGHTS_MANIFEST_PATH):
            print(
                f"Downloading updated weights manifest from {UPDATED_WEIGHTS_MANIFEST_URL}"
            )
            start = time.time()
            try:
                subprocess.check_call(
                    [
                        "pget",
                        "--log-level",
                        "warn",
                        "-f",
                        UPDATED_WEIGHTS_MANIFEST_URL,
                        UPDATED_WEIGHTS_MANIFEST_PATH,
                    ],
                    close_fds=False,
                    timeout=5,
                )
                print(
                    f"Downloading {UPDATED_WEIGHTS_MANIFEST_URL} took: {(time.time() - start):.2f}s"
                )
            except subprocess.CalledProcessError:
                print(f"Failed to download {UPDATED_WEIGHTS_MANIFEST_URL}")
                pass
            except subprocess.TimeoutExpired:
                print(
                    f"Download from {UPDATED_WEIGHTS_MANIFEST_URL} timed out")
                pass

    def _merge_manifests(self):
        if os.path.exists(WEIGHTS_MANIFEST_PATH):
            with open(WEIGHTS_MANIFEST_PATH, "r") as f:
                original_manifest = json.load(f)
        else:
            original_manifest = {}

        if not os.path.exists(UPDATED_WEIGHTS_MANIFEST_PATH):
            return original_manifest

        with open(UPDATED_WEIGHTS_MANIFEST_PATH, "r") as f:
            updated_manifest = json.load(f)

        for key in updated_manifest:
            if key in original_manifest:
                for item in updated_manifest[key]:
                    if item not in original_manifest[key]:
                        print(f"Adding {item} to {key}")
                        original_manifest[key].append(item)
            else:
                original_manifest[key] = updated_manifest[key]

        return original_manifest

    def _generate_weights_map(self, keys, dest):
        custom_weights = {
            "control_openpose-fp16.safetensors": "https://weights.basedlabs.ai/control_openpose-fp16.safetensors.tar",
            "zavychromaxl_v70.safetensors": "ComfyUI/models/checkpoints/zavychromaxl_v70.safetensors",
            # "zavychromaxl_v31.safetensors": "https://weights.replicate.delivery/default/basedlabs-media/zavychromaxl_v31.safetensors.tar",
            "kitty.safetensors": "https://weights.replicate.delivery/default/basedlabs-media/kitty.safetensors.tar",
            "rife47.pth": "https://weights.replicate.delivery/default/basedlabs-media/rife47.pth.tar",
            "svd_xt_1_1.safetensors": "https://weights.replicate.delivery/default/basedlabs-media/svd_xt_1_1.safetensors.tar",
            "ral-spike-sdxl.safetensors": "ComfyUI/models/loras/ral-spike-sdxl.safetensors"
        }

        return {
            key: {
                "url": custom_weights[key] if key in custom_weights else f"{BASE_URL}/{dest}/{key}.tar",
                "dest": f"{BASE_PATH}/{dest}"
            }
            for key in keys
        }

    def _initialize_weights_map(self):
        weights_map = {}
        for key in self.weights_manifest.keys():
            if key.isupper():
                weights_map.update(
                    self._generate_weights_map(
                        self.weights_manifest[key], key.lower())
                )

        for module_name in dir(helpers):
            module = getattr(helpers, module_name)
            if hasattr(module, "weights_map"):
                weights_map.update(module.weights_map(BASE_URL))

        return weights_map

    def non_commercial_weights(self):
        return [
            "inswapper_128.onnx",
            "inswapper_128_fp16.onnx",
            "proteus_v02.safetensors",
            "RealVisXL_V3.0_Turbo.safetensors",
            "sd_xl_turbo_1.0.safetensors",
            "sd_xl_turbo_1.0_fp16.safetensors",
            "svd.safetensors",
            "svd_xt.safetensors",
            "turbovisionxlSuperFastXLBasedOnNew_tvxlV32Bakedvae",
            "copaxTimelessxlSDXL1_v8.safetensors",
            "MODILL_XL_0.27_RC.safetensors",
            "epicrealismXL_v10.safetensors",
            "RMBG-1.4/model.pth",
        ]

    def is_non_commercial_only(self, weight_str):
        return weight_str in self.non_commercial_weights()

    def get_weights_by_type(self, weight_type):
        return self.weights_manifest.get(weight_type, [])
