import numpy as np
from typing import Tuple
from PIL import Image
from tqdm import tqdm


def max_plus_transform_blockwise(image: np.ndarray, num_bits: int = 8) -> np.ndarray:
    """
    经典等价实现：对输入灰度图执行与量子电路等价的 Max-Plus 小波变换（逐 2x2 块）。

    等价规则（与当前量子实现一致）：
      - 记 2x2 块中像素为 a=image[i,j], b=image[i,j+1], c=image[i+1,j], d=image[i+1,j+1]
      - LL = max(a, b, c, d)
      - LH = (b - a) mod 2^num_bits
      - HL = (max(c, d) - b) mod 2^num_bits
      - HH = (d - c) mod 2^num_bits

    输出图像采用与量子后处理一致的布局：
      output[i, j]     = HL
      output[i, j+1]   = HH
      output[i+1, j]   = LL
      output[i+1, j+1] = LH

    要求输入的高宽均为偶数。
    """
    assert image.ndim == 2, "只支持单通道灰度图"
    h, w = image.shape
    assert h % 2 == 0 and w % 2 == 0, "图像高宽必须为偶数"

    mod_mask = (1 << num_bits) - 1
    img = image.astype(np.uint16)  # 临时扩位，避免 Python 负数与溢出问题
    out = np.zeros_like(img, dtype=np.uint16)

    for i in tqdm(range(0, h, 2), desc="Classical Max-Plus"):
        for j in range(0, w, 2):
            a = int(img[i, j])
            b = int(img[i, j + 1])
            c = int(img[i + 1, j])
            d = int(img[i + 1, j + 1])

            ll = max(a, b, c, d)
            lh = (b - a) & mod_mask
            md = max(c, d)
            hl = (md - b) & mod_mask
            hh = (d - c) & mod_mask

            out[i, j] = hl
            out[i, j + 1] = hh
            out[i + 1, j] = ll
            out[i + 1, j + 1] = lh

    # 裁回到 uint8（或按位宽返回）
    if num_bits <= 8:
        return out.astype(np.uint8)
    elif num_bits <= 16:
        return out.astype(np.uint16)
    else:
        return out.astype(np.uint32)


def load_grayscale_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert('L'))


def save_grayscale_image(path: str, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="经典等价的 Max-Plus 小波变换（逐 2x2 块）")
    parser.add_argument("input", type=str, help="输入灰度图路径（宽高需为偶数）")
    parser.add_argument("output", type=str, help="输出图路径")
    parser.add_argument("--bits", type=int, default=8, help="位宽（与量子寄存器位数等价，默认8位）")
    args = parser.parse_args()

    img = load_grayscale_image(args.input)
    transformed = max_plus_transform_blockwise(img, num_bits=args.bits)
    save_grayscale_image(args.output, transformed)
    print(f"Saved: {args.output}")


