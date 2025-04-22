import os
import shutil

def merge_subfolders(source_root, target_root):
    for subfolder in os.listdir(source_root):  # x40, x100, etc.
        subfolder_path = os.path.join(source_root, subfolder)
        if os.path.isdir(subfolder_path):
            for label in os.listdir(subfolder_path):  # benign, malignant
                label_path = os.path.join(subfolder_path, label)
                target_label_path = os.path.join(target_root, label)
                os.makedirs(target_label_path, exist_ok=True)

                for img_file in os.listdir(label_path):
                    src_file = os.path.join(label_path, img_file)
                    new_name = f"{subfolder}_{img_file}"
                    dst_file = os.path.join(target_label_path, new_name)
                    shutil.copy2(src_file, dst_file)

# Ejemplo de uso:
merge_subfolders("./images/binary_scenario/train", "./images/binary_scenario_merged/train")
merge_subfolders("./images/binary_scenario/test", "./images/binary_scenario_merged/test")
merge_subfolders("./images/binary_scenario/val", "./images/binary_scenario_merged/val")