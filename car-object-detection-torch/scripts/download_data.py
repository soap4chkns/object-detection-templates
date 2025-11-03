import os
import sys

import kaggle

OWNER_NAME = "sshikamaru"
DATASET_NAME = "car-object-detection"

SECRET_PATH = os.path.join(os.path.expanduser("~"), ".kaggle/kaggle.json")
OUTPUT_PATH = "./data"

if not os.path.exists(SECRET_PATH):
    try:
        from google.colab import userdata  # type: ignore

        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_API_KEY")
    except ImportError:
        raise ImportError("Intended for development only in google colab")
    except Exception:
        raise Exception("error setting up google colab kaggle credentials")

if os.path.exists(OUTPUT_PATH):
    print("Warning: dataset already exists. skipping download")
    sys.exit(-1)

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    os.path.join(OWNER_NAME, DATASET_NAME),
    path=".",
    unzip=True,
)
