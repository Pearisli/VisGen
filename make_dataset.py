import os
import shutil
import argparse

def organize_images(source_dir: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        category = filename.split('_')[-1].split('.')[0]

        category_folder = os.path.join(target_dir, category)
        os.makedirs(category_folder, exist_ok=True)

        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(category_folder, filename)
        shutil.move(source_path, target_path)

    print("Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Organize images into category folders based on filename suffix.")
    parser.add_argument('source_dir', type=str, help="Path to the source directory containing the images.")
    parser.add_argument('target_dir', type=str, help="Path to the target directory where images will be organized.")

    args = parser.parse_args()

    organize_images(args.source_dir, args.target_dir)

if __name__ == "__main__":
    main()
