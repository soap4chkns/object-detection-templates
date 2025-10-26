# dataset path identifier 
USERNAME="sshikamaru"
DATASET_NAME="car-object-detection"

download_dataset() {
    if [ -d "data" ]; then
        echo "dataset already downloaded. skipping download ..."
        return
    fi

    DATASET_PATH="${1}/${2}"
    kaggle datasets download -d $DATASET_PATH --path . --unzip -w
    if [ $? -ne 0 ]; then
        echo "Failed to download dataset"
    fi
}

download_dataset $USERNAME $DATASET_NAME
