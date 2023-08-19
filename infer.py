from PIL import Image
from tqdm import tqdm

from utils.inference_handler import setup_model, get_palette, get_mask_array, get_inference_results

import os

input_path = "input_images"
output_path = "_results"

do_palette = True
checkpoints_path = "trained_checkpoint/checkpoint_u2net.pth"
device = "cpu"

u2_net = setup_model(checkpoints_path, device)

palette = get_palette(4)

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)

    # setup results list
    results = []

    # setup inference dataset
    images_list = sorted(os.listdir(input_path))
    images_list = [x for x in images_list if x != ".keep"]
    pbar = tqdm(total=len(images_list))

    for image_name in images_list:
        # prep image and infer
        image = Image.open(os.path.join(input_path, image_name))
        output_arr = get_mask_array(image, u2_net)

        # generate output image
        output_image = Image.fromarray(output_arr.astype("uint8"), mode="L")
        if do_palette:
            output_image.putpalette(palette)

        # save output image
        root, _ = os.path.splitext(image_name)
        output_name = os.path.join(output_path, root + ".png")
        output_image.save(output_name)

        # generate result
        result = get_inference_results(output_image)
        results.append({output_name: result})

        # save input to directory
        image.save(os.path.join(output_path, root + "_orig.png"))
        pbar.update(1)

    pbar.close()
    print(results)
