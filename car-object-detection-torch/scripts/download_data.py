import json
import os
import sys

OWNER_NAME = "sshikamaru"
DATASET_NAME = "car-object-detection"

SECRET_PATH = os.path.join(os.path.expanduser("~"), ".config", "kaggle", "kaggle.json")
OUTPUT_PATH = "./data"

if not os.path.exists(SECRET_PATH):
    # the kaggle api requires the ~/.kaggle/kaggle.json file exist
    # even if credentials are sourced from environment variables
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".config", "kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    try:
        # manually create the kaggle.json file
        username = userdata.get("KAGGLE_USERNAME")  # pyright: ignore[reportUndefinedVariable] # noqa: F821
        key = userdata.get("KAGGLE_API_KEY")  # pyright: ignore[reportUndefinedVariable] # noqa: F821
        kaggle_creds = {"username": username, "key": key}

        with open(SECRET_PATH, "w") as fh:
            json.dump(kaggle_creds, fh)

    except ImportError:
        raise ImportError("Intended for development only in google colab")
    except Exception:
        raise Exception("error setting up google colab kaggle credentials")

if os.path.exists(OUTPUT_PATH):
    print("Warning: dataset already exists. skipping download")
    sys.exit(-1)

# cannot import kaggle without the presence of ~/.kaggle first
import kaggle  # noqa: E402

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    os.path.join(OWNER_NAME, DATASET_NAME),
    path=".",
    unzip=True,
)
