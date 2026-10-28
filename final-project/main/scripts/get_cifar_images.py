"""Extract cifar-10 images"""

import os
import pickle
from argparse import ArgumentParser

import numpy as np
from PIL import Image

int_to_batch = {
    1: "data_batch_1",
    2: "data_batch_2",
    3: "data_batch_3",
    4: "data_batch_4",
    5: "data_batch_5",
}


def main(cifar_dir, output_dir, batches):
    cifar_batches = []
    for d in batches:
        if int(d) in int_to_batch:
            cifar_batches.append(int_to_batch[int(d)])

    images = None

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for batch in cifar_batches:
        with open(
            os.path.join(cifar_dir, batch),
            "rb",
        ) as fo:
            d = pickle.load(fo, encoding="bytes")
        if images is None:
            images = np.array(d[b"data"])
        else:
            images = np.concatenate([images, np.array(d[b"data"])])

    for i, image in enumerate(images):
        r = image[0:1024].reshape((32, 32))
        g = image[1024:2048].reshape((32, 32))
        b = image[2048:3072].reshape((32, 32))

        img_reshaped = np.stack([r, g, b], axis=2).astype(np.uint8)

        img = Image.fromarray(img_reshaped)
        img.save(os.path.join(output_dir, f"image_{i}.jpg"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cifar-dir", type=str, required=True, help="Location of CIFAR10 root dir"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Location of output dir"
    )
    parser.add_argument(
        "--batches",
        type=str,
        required=False,
        help="Which batches to use",
        default="12345",
    )

    args = parser.parse_args()
    main(args.cifar_dir, args.output_dir, args.batches)
