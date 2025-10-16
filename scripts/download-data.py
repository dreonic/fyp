import kagglehub
import shutil
import os

def download_kaggle_dataset(dataset_name, desired_location=None):
    path = kagglehub.dataset_download(dataset_name)

    if desired_location:
        if not os.path.exists(desired_location):
            os.makedirs(desired_location)
        
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(desired_location, item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.move(s, d)
        
        shutil.rmtree(path)  # Remove the now empty extracted folder
        print(f"Files moved to {desired_location}")

def main(args):
    dataset_name = args.dataset
    desired_location = args.destination or None

    assert dataset_name, "Dataset name is required."

    download_kaggle_dataset(dataset_name, desired_location)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Kaggle Dataset")
    parser.add_argument('--dataset', type=str, required=True, help='Kaggle dataset name (e.g., plameneduardo/a-covid-multiclass-dataset-of-ct-scans)')
    parser.add_argument('--destination', type=str, help='Desired location to save the dataset')

    args = parser.parse_args()
    main(args)
