import os
import re
import glob


def generate_csv_from_dir(data_dir, file_path):
    output_list = []
    original_dir = os.path.join(data_dir, "images", "original")
    blended_dir = os.path.join(data_dir, "images", "blended")
    mask_dir = os.path.join(data_dir, "mask")

    # add original images to output_list
    for image_path in glob.glob(os.path.join(original_dir, "*")):
        output_list.append([image_path, "", "0"])

    # match blended images and mask images
    for image_path in glob.glob(os.path.join(blended_dir, "*")):
        image_name = re.match(os.path.join(blended_dir, r'(\d*).png'), image_path).group(1)
        frame_image_path = os.path.join(blended_dir, image_name + '.png')
        mask_image_path = os.path.join(mask_dir, 'mask_' + image_name + '.png')
        output_list.append([frame_image_path, mask_image_path, "1"])

    # output to csv
    output_line_list = [",".join(output) for output in output_list]
    with open(file_path, "w+") as csv_file:
        csv_file.write("\n".join(output_line_list))


root_dir = os.path.expanduser("~/zhiwen/XRay/Data/data_for_fake_imagery")
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")

output_dir = os.path.expanduser("~/zhiwen/XRay/Code/data")
train_csv_path = os.path.join(output_dir, "train_image_selected.csv")
val_csv_path = os.path.join(output_dir, "val_image_selected.csv")

generate_csv_from_dir(train_dir, train_csv_path)
generate_csv_from_dir(val_dir, val_csv_path)
