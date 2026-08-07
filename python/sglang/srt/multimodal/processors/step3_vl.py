import math
import re
from itertools import product
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from transformers import BatchFeature, ProcessorMixin, TensorType

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.step3_vl import Step3VLForConditionalGeneration
from sglang.srt.models.step3_vl_10b import StepVLForConditionalGeneration
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)

# ============================================================================
# Step3-VL 多模态图像预处理器
#
# 整体思路（类似 LLaVA-1.6 AnyRes / InternVL 的动态分辨率方案）：
#   1. 一张原图会被处理成「1 张全局缩略图 + N 张局部高清 patch」；
#   2. 全局图缩放到 728x728，编码成 169 个视觉 token（13x13）；
#   3. 每个局部 patch 缩放到 504x504，编码成 81 个视觉 token（9x9）；
#   4. 在文本里把用户写的 1 个 <im_patch> 占位符，展开成上面这一整串
#      特殊 token（<patch_start>...<patch_end><im_start>...<im_end>），
#      这样 LLM 就能在正确的位置「看到」图片。
#
# 一个完整例子（输入 1512x1008 的图）：
#   determine_window_size(1512, 1008) -> 504     （长/短 = 1.5，不 >4，取 504）
#   get_image_size_for_crop -> (1512, 1008)      （已是 504 的整数倍）
#   slide_window -> 3x2 = 6 个 patch，x_num=3, y_num=2
#   换行掩码：第 0 行末尾 patch#2 标 True，第 1 行是最后一行不标
#   -> 最终 token 数 = 6*(81+2) + (169+2) + 1 = 670
# ============================================================================

Step3Image = Union[Image.Image, torch.Tensor]
# (全局图, 局部 patch 列表, 每个 patch 后是否要插 <patch_newline> 的掩码)
ImageWithPatches = tuple[Step3Image, list[Step3Image], list[int] | None]


class GPUToTensor(torch.nn.Module):
    """把任意输入图像（PIL / numpy HWC / torch CHW）统一成 CHW 的 float32 tensor，
    数值归一化到 [0, 1]，并在有 GPU 时搬到 CUDA 上（后续 resize/normalize 走 GPU 更快）。

    例子：
        PIL RGB 图 (W=800, H=600)      -> tensor[3, 600, 800] float32, 值域 [0,1]
        numpy 灰度图 (600, 800)         -> 先扩成 (600, 800, 3)，再转成 [3, 600, 800]
        torch uint8 tensor [1, H, W]   -> 复制通道成 [3, H, W]，再 /255
    """

    def forward(
        self, raw_image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(raw_image, torch.Tensor):
            # 分支 1：已经是 tensor（例如上游已解码好的图像），只做通道/dtype 规整
            image_tensor = raw_image
            if image_tensor.ndim != 3:
                raise TypeError(
                    f"Expected CHW image tensor, got shape {tuple(image_tensor.shape)}"
                )
            if image_tensor.shape[0] == 1:
                # 灰度图 [1,H,W] -> 复制成三通道 [3,H,W]
                image_tensor = image_tensor.repeat(3, 1, 1)
            elif image_tensor.shape[0] != 3:
                raise TypeError(
                    f"Expected CHW image tensor with 1 or 3 channels, got shape {tuple(image_tensor.shape)}"
                )
            if image_tensor.dtype == torch.uint8:
                # uint8 [0,255] -> float32 [0,1]，和 PIL 分支的 ToTensor() 行为对齐
                image_tensor = image_tensor.to(torch.float32).div(255)
            elif not image_tensor.is_floating_point():
                image_tensor = image_tensor.to(torch.float32)
            # 注意：这个分支不搬 GPU，保持调用方给的 device
            return image_tensor.contiguous()
        if isinstance(raw_image, Image.Image):
            # 分支 2：PIL 图。ToTensor() 已包含 HWC->CHW + /255
            image_tensor = transforms.ToTensor()(raw_image)
            if torch.cuda.is_available():
                image_tensor = image_tensor.to(torch.device("cuda"))
            return image_tensor
        # 分支 3：numpy HWC（或 HW 灰度）
        if raw_image.ndim == 2:
            # (H, W) -> (H, W, 3)
            raw_image = raw_image[:, :, None].repeat(3, -1)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        image_tensor = torch.from_numpy(raw_image).to(device)
        # HWC -> CHW
        image_tensor = torch.permute(image_tensor, (2, 0, 1)).contiguous()
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32).div(255)
        return image_tensor


class Step3VisionProcessor:
    """把一张图变成模型输入的 pixel_values（归一化 + 缩放到固定边长）。

    内部维护两条 torchvision 流水线：
      - transform:       全局缩略图用，缩放到 size x size（默认 728）
      - patch_transform: 局部 patch 用，缩放到 patch_size x patch_size（默认 504）

    注意执行顺序是 ToTensor -> Normalize -> Resize（先归一化再缩放），
    与 HF 的常见写法（先 resize 再 normalize）相反，这是 Step3 官方实现的行为，
    改动会导致数值和权重训练时不一致。

    mean/std 用的是 CLIP 的经典统计量。
    """

    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose(
            [
                GPUToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize(
                    (size, size),
                    interpolation=(
                        InterpolationMode.BICUBIC
                        if interpolation_mode == "bicubic"
                        else InterpolationMode.BILINEAR
                    ),
                    antialias=True,
                ),
            ]
        )

        self.patch_transform = (
            transforms.Compose(
                [
                    GPUToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(
                        (patch_size, patch_size),
                        interpolation=(
                            InterpolationMode.BICUBIC
                            if interpolation_mode == "bicubic"
                            else InterpolationMode.BILINEAR
                        ),
                        antialias=True,
                    ),
                ]
            )
            if patch_size is not None
            else None
        )

    def __call__(self, image, is_patch=False):
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        else:
            return {"pixel_values": self.transform(image).unsqueeze(0)}


class ImagePatcher:
    """把一张图按滑窗切成若干不重叠的正方形 patch（动态分辨率的核心，纯几何逻辑）。

    完整流程（见 __call__）：
        原图
         -> square_pad             极端长宽比时补成正方形
         -> resize(preprocess)     长边限制到 3024 以内
         -> determine_window_size  决定窗口边长（返回 0 表示不切 patch）
         -> resize(crop)           尺寸对齐到窗口的整数倍
         -> slide_window           按窗口逐行逐列裁出 patch
        返回 (全局图, patch 列表, 换行掩码)

    这个类同时被两个地方用：
      - __call__ 真正切图（推理时）
      - get_num_patches 只算数量（调度器提前预估 token 数、分配 KV cache 时）
    两条路径必须严格一致，否则预估长度和实际长度对不上会 assert 失败。
    """

    def get_image_size(self, img: Step3Image) -> tuple[int, int]:
        """统一返回 (width, height)。
        PIL 的 .size 本来就是 (W, H)；tensor 是 CHW，所以取 shape[-1]=W, shape[-2]=H。
        """
        if isinstance(img, Image.Image):
            return img.size
        if isinstance(img, torch.Tensor):
            if img.ndim != 3:
                raise TypeError(
                    f"Expected CHW image tensor, got shape {tuple(img.shape)}"
                )
            return int(img.shape[-1]), int(img.shape[-2])
        raise TypeError(f"Unsupported image type: {type(img)}")

    def determine_window_size(self, long: int, short: int) -> int:
        """决定切 patch 的窗口边长；返回 0 表示「这张图不切 patch，只用全局图」。

        规则：
          - 长边 <= 728（正好是全局图分辨率）：
              长宽比 <= 1.5 -> 0      图本身够小、比例正常，全局图已经够看
              长宽比 >  1.5 -> short  用短边做窗口，沿长边切几刀，避免被压扁失真
          - 长边 > 728：
              长宽比 <= 4 -> 504              标准 patch 尺寸
              长宽比 >  4 -> min(short, 504)  超细长条用短边，避免窗口比图还宽

        例子：
          (640, 480)   -> 0     长边 640<=728 且比例 1.33<=1.5，不切
          (700, 200)   -> 200   长边 <=728 但比例 3.5>1.5，用短边 200 做窗口
          (1512, 1008) -> 504   长边 >728，比例 1.5<=4
          (4000, 200)  -> 200   比例 20>4，取 min(200, 504)
        """
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        """在 width x height 的画布上按 (size, step) 滑窗，返回所有窗口框和行列数。

        返回：
          - windows: [(x, y, w, h), ...]，顺序是「先行后列」（逐行从左到右）
          - (x_num, y_num): 横向窗口数、纵向窗口数

        例子：width=1512, height=1008, size=step=504（本实现里 step==size，即不重叠）
          x_num = ceil((1512-504)/504 + 1) = 3
          y_num = ceil((1008-504)/504 + 1) = 2
          -> 6 个框：(0,0) (504,0) (1008,0) (0,504) (504,504) (1008,504)

        注：img_rate_thr 只做了断言校验，实际未参与任何过滤（沿用自上游实现）。
        """
        assert 1 >= img_rate_thr >= 0, "The `img_rate_thr` should lie in 0~1"
        windows = []
        # Sliding windows.
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            # 横向起点：0, step, 2*step, ...
            x_num = 1 if width <= size_w else math.ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            # 最后一个窗口若超出右边界，就贴着右边界对齐（会和前一个窗口重叠一点）
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            # 纵向同理
            y_num = 1 if height <= size_h else math.ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            # product(y_start, x_start) 先遍历 y 再遍历 x，得到 (y, x) 对，
            # 保证输出顺序是「逐行从左到右」；随后交换两列变成 (x, y)。
            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            # 拼成 [x1, y1, x2, y2] 形式（start + size 得到右下角）
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        # [x1, y1, x2, y2] -> (x, y, w, h)
        # 注意：x_num / y_num 取的是循环里最后一组 size 的值，
        # 本文件所有调用都只传 1 组 size，所以没问题。
        return [
            (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
            for box in windows
        ], (x_num, y_num)

    def square_pad(self, img: Step3Image) -> Step3Image:
        """右侧/下方补黑边，把图补成正方形（内容左上角对齐）。

        例子：(200, 50) -> (200, 200)，右边不动、下方补 150 行黑。
        """
        w, h = self.get_image_size(img)
        if w == h:
            return img
        size = max(w, h)
        if isinstance(img, Image.Image):
            padded = Image.new(img.mode, (size, size), 0)
            padded.paste(img, (0, 0))
            return padded
        # F.pad 的顺序是 (left, right, top, bottom)
        return torch.nn.functional.pad(img, (0, size - w, 0, size - h), value=0)

    def get_image_size_for_padding(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:
        """判断是否需要 square_pad：只有「又小又极端细长」的图才补。

        条件：短边 < 32 且 长宽比 > 4（或 < 1/4）
        例子：
          (400, 20)  -> (400, 400)   短边 20<32 且比例 20>4，需要补成正方形
          (400, 100) -> (400, 100)   短边 100>=32，不补
          (100, 20)  -> (100, 100)   短边 20<32 且比例 5>4，补

        动机：这类图缩放到 728x728 会被拉伸到完全无法辨认，补边比拉伸损失小。
        """
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:
        """限制长边不超过 3024（等比缩放），防止超大图切出过多 patch 撑爆显存。

        例子：
          (6000, 3000) -> scale = 3024/6000 = 0.504 -> (3024, 1512)
          (2000, 1000) -> 原样返回
        """
        if max(img_height, img_width) > 3024:
            scale_factor = 3024 / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
            return img_width, img_height
        else:
            return img_width, img_height

    def get_image_size_for_crop(
        self, img_width: int, img_height: int, window_size: int
    ):
        """把尺寸对齐到 window_size 的整数倍，这样滑窗能刚好铺满、不留边角。

        取整策略：小数部分 > 0.2 才向上进位，否则向下截断
        （避免为了一条细边多切一整行 patch）。
        某一边比窗口还小时保持原样（此时该方向只会有 1 个窗口）。

        例子（window_size=504）：
          (1512, 1008) -> (1512, 1008)   3.0 / 2.0，正好整数倍
          (1600, 1008) -> (1512, 1008)   1600/504=3.17，小数 0.17<=0.2 -> 截断为 3
          (1700, 1008) -> (2016, 1008)   1700/504=3.37，小数 0.37>0.2  -> 进位为 4
          (300, 1008)  -> (300, 1008)    宽 300<504，宽方向保持原样
        """
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def resize(self, img: Step3Image, size: tuple[int, int]) -> Step3Image:
        """统一的 resize 入口，size 是 (width, height)。
        torchvision 的 F.resize 要 (height, width)，所以要反过来传。
        """
        if isinstance(img, Image.Image):
            return img.resize(size, Image.Resampling.BILINEAR)
        return F.resize(
            img,
            [size[1], size[0]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).contiguous()

    def patch_crop(
        self, img: Step3Image, i: int, j: int, th: int, tw: int
    ) -> Step3Image:
        """裁剪一块区域。参数是 torchvision 风格：i=top(y), j=left(x), th=高, tw=宽。
        PIL 的 crop 要 (left, upper, right, lower)，所以顺序要换。
        """
        if isinstance(img, Image.Image):
            return img.crop((j, i, j + tw, i + th))
        return img[:, i : i + th, j : j + tw].contiguous()

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        """不真正切图，只算出 (patch 数, 换行符数)。
        调度器用它提前算 prompt 长度，必须和 __call__ 的结果完全一致。

        返回的第二项 full_rows 是「需要插 <patch_newline> 的行数」：
        除最后一行外每行末尾插一个。

        例子（1512x1008，window=504，3x2=6 个 patch）：
          full_rows = (6-1)//3 + 1 = 2
          6 % 3 == 0 -> full_rows -= 1 -> 1
          即只在第 0 行末尾插换行；最后一行不插。
        """
        img_width, img_height = self.get_image_size_for_padding(img_width, img_height)
        img_width, img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        window_size = self.determine_window_size(
            max(img_height, img_width), min(img_height, img_width)
        )
        if window_size == 0:
            # 不切 patch：0 个 patch、0 个换行
            return 0, 0
        else:
            img_width, img_height = self.get_image_size_for_crop(
                img_width, img_height, window_size
            )
            center_list, (x_num, y_num) = self.slide_window(
                img_width,
                img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            # 向上取整得到总行数
            full_rows = (len(center_list) - 1) // x_num + 1
            # 若最后一行是「满行」，它的行尾换行符会被 __call__ 丢弃，这里要对齐地减 1
            if len(center_list) > 0 and len(center_list) % x_num == 0:
                full_rows -= 1
            return len(center_list), full_rows

    def __call__(
        self, img: Step3Image
    ) -> tuple[Step3Image, list[Step3Image], list[bool] | None]:
        """真正执行切图。

        返回 (全局图, patch 列表, 换行掩码)：
          - 全局图：经过 pad/长边限制后的整图，后续会被缩到 728
          - patch 列表：若干 window_size x window_size 的裁块，后续会被缩到 504
          - 换行掩码：与 patch 列表等长的 bool 列表，True 表示该 patch 后要插
            <patch_newline>（让模型感知二维布局）；不切 patch 时为 None

        例子（1512x1008）：
          -> (1512x1008 整图, [6 个 504x504 patch], [F,F,T,F,F,F])
             索引 2 是第 0 行最后一个 patch，所以标 True；
             索引 5 虽然也是行尾，但它是整张图最后一个，换行没意义，被丢弃。
        """
        img_width, img_height = self.get_image_size(img)
        # 步骤 1：极端细长小图 -> 补成正方形
        new_img_width, new_img_height = self.get_image_size_for_padding(
            img_width, img_height
        )
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = self.get_image_size(img)

        # 步骤 2：长边限制到 3024
        new_img_width, new_img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        img = self.resize(img, (new_img_width, new_img_height))
        # 步骤 3：决定窗口大小
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width), min(new_img_height, new_img_width)
        )
        if window_size == 0:
            # 小图/正常比例：只返回整图，不切 patch
            return img, [], None
        else:
            # 步骤 4：对齐到窗口整数倍，得到一张「专供裁剪」的图。
            # 注意全局图 img 仍是步骤 2 的版本，不受这次 resize 影响。
            new_img_width, new_img_height = self.get_image_size_for_crop(
                new_img_width, new_img_height, window_size
            )
            if (new_img_width, new_img_height) != (img_width, img_height):
                img_for_crop = self.resize(img, (new_img_width, new_img_height))
            else:
                img_for_crop = img

            patches = []
            newlines = []
            # 步骤 5：逐个窗口裁剪，并记录行尾位置
            center_list, (x_num, y_num) = self.slide_window(
                new_img_width,
                new_img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)
                patches.append(big_patch)
                # 每 x_num 个 patch 就是一行的结尾
                if (patch_id + 1) % x_num == 0:
                    newlines.append(patch_id)

            # 最后一个 patch 后面不需要换行（后面紧跟的是全局图 token）
            if newlines and newlines[-1] == len(patches) - 1:
                newlines.pop()

            return (
                img,
                patches,
                # 把「行尾索引列表」转成等长 bool 掩码，方便下游按位判断
                (
                    [i in newlines for i in range(len(patches))]
                    if len(patches) > 0
                    else None
                ),
            )


class Step3VLProcessor:
    """HuggingFace 风格的 processor：同时产出 pixel_values 和展开后的 input_ids。

    token 布局（每张图）：
        <patch_start> 81*<im_patch> <patch_end>   \\ 每个局部 patch 一组
        [<patch_newline>]                          \\ 行尾才有
        ...
        <im_start> 169*<im_patch> <im_end>        \\ 最后才是全局图

    注意顺序是「局部 patch 在前、全局图在后」。

    例子（1512x1008，6 个 patch，1 个换行）：
        6*(1+81+1) + 1 + (1+169+1) = 498 + 1 + 171 = 670 个 token
    """

    def __init__(
        self,
        config,
        tokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        # 传进来的可能是完整 processor，也可能是裸 tokenizer，这里统一取 tokenizer
        if isinstance(tokenizer, ProcessorMixin):
            tokenizer = tokenizer.tokenizer
        self.tokenizer = tokenizer

        self.image_size = 728  # 全局图边长
        self.patch_size = 504  # 局部 patch 边长
        self.image_preprocessor = Step3VisionProcessor(
            self.image_size, "bilinear", self.patch_size
        )

        # 728 -> ViT 下采样后是 13x13 = 169 个视觉 token
        self.num_image_feature_size = 169
        # 504 -> 9x9 = 81 个视觉 token
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size

        self.patcher = ImagePatcher()

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        """给定原图尺寸，算出它会占多少个 token（调度器预估用，不需要真的解码图片）。

        公式拆解：
          num_patches * (81 + 2)   每个 patch：81 个特征 + <patch_start> + <patch_end>
          + 169 + 2                全局图：169 个特征 + <im_start> + <im_end>
          + num_newlines           行尾的 <patch_newline>

        例子：1512x1008 -> 6*(81+2) + 171 + 1 = 670
              640x480   -> 0 + 171 + 0 = 171（不切 patch）
        """
        num_patches, num_newlines = self.patcher.get_num_patches(img_width, img_height)

        return (
            num_patches * (self.num_patch_feature_size + 2)
            + self.num_image_feature_size
            + 2
            + num_newlines
        )

    def _split_images(self, images: list[Image.Image]) -> list[ImageWithPatches]:
        """对 batch 里每张图分别跑一遍 ImagePatcher。"""
        result = []
        for img in images:
            result.append(self.patcher(img))
        return result

    def _convert_images_to_pixel_values(
        self,
        images: list[Step3Image],
        is_patch: bool = False,
    ) -> list[torch.Tensor]:
        """批量做归一化 + 缩放。is_patch 决定缩到 504 还是 728。
        返回的每个元素形状是 [1, 3, S, S]。
        """
        return [
            self.image_preprocessor(img, is_patch=is_patch)["pixel_values"]
            for img in images
        ]

    def _get_patch_repl(
        self,
        num_patches: int,
        patch_newline_mask: list[bool] | None,
    ) -> tuple[str, list[int]]:
        """生成所有局部 patch 对应的文本片段和 token id 列表。

        例子（num_patches=2, mask=[True, False]）：
          text = "<patch_start>{81个<im_patch>}<patch_end><patch_newline>"
                 "<patch_start>{81个<im_patch>}<patch_end>"
          ids  = [ps, img*81, pe, nl, ps, img*81, pe]   共 83+1+83 = 167 个
        """
        text = ""
        token_ids = []
        for i in range(num_patches):
            # num_patches > 0 时 mask 一定非 None（由 ImagePatcher 保证）
            assert len(patch_newline_mask) == num_patches
            text += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            token_ids.extend(
                [self.tokenizer.convert_tokens_to_ids("<patch_start>")]
                + [self.image_token_id] * self.num_patch_feature_size
                + [self.tokenizer.convert_tokens_to_ids("<patch_end>")]
            )
            if patch_newline_mask and patch_newline_mask[i]:
                text += "<patch_newline>"
                token_ids.append(
                    self.tokenizer.convert_tokens_to_ids("<patch_newline>")
                )
        return text, token_ids

    def _get_image_repl(
        self,
        num_images: int,
    ) -> tuple[str, list[int]]:
        """生成全局图对应的文本片段和 token id（<im_start> + 169 + <im_end>）。
        num_images 在本文件的调用里恒为 1（每张图单独处理）。
        """
        text = f"<im_start>{self.image_feature_placeholder}<im_end>"
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<im_start>")]
            + [self.image_token_id] * self.num_image_feature_size
            + [self.tokenizer.convert_tokens_to_ids("<im_end>")]
        )
        return text * num_images, token_ids * num_images

    def _get_image_repl_features(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: Optional[list[bool]],
    ) -> tuple[str, list[int]]:
        """拼接：先所有局部 patch，再全局图。"""
        if num_patches > 0:
            patch_repl, patch_repl_ids = self._get_patch_repl(
                num_patches, patch_new_line_idx
            )
        else:
            # 小图不切 patch，只有全局图部分
            patch_repl = ""
            patch_repl_ids = []
        image_repl, image_repl_ids = self._get_image_repl(num_images)
        return patch_repl + image_repl, patch_repl_ids + image_repl_ids

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        """把 text 中第 i 个 placeholder 替换成 repls[i]（按顺序一一对应）。

        例子：
          text  = "看图 A：<im_patch> 再看图 B：<im_patch>"
          repls = ["<A展开>", "<B展开>"]
          -> "看图 A：<A展开> 再看图 B：<B展开>"

        占位符个数和图片数不一致会直接报错，避免图文错位。
        """
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        *args,
        **kwargs,
    ) -> BatchFeature:
        """HF processor 标准入口：文本 + 图片 -> BatchFeature。

        输出字段：
          input_ids / attention_mask   来自 tokenizer
          pixel_values                 [num_images, 3, 728, 728]     所有全局图
          patch_pixel_values           [total_patches, 3, 504, 504]  所有图的 patch 拼一起
          num_patches                  list[int]，每张图各有几个 patch（模型侧据此切分）
          patch_newline_mask           bool tensor，所有 patch 的换行标记拼一起

        例子（2 张图，分别切出 6 和 0 个 patch）：
          pixel_values       -> [2, 3, 728, 728]
          patch_pixel_values -> [6, 3, 504, 504]
          num_patches        -> [6, 0]
          patch_newline_mask -> tensor([F,F,T,F,F,F])
        """
        # 参数归一化：统一成 list
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            # 纯文本请求，走 tokenizer 就行
            image_inputs = {}
            text_inputs = self.tokenizer(text)
        else:
            splitted_images_data = self._split_images(images)
            pixel_values_lst = []
            patch_pixel_values_lst = []
            patch_newline_mask_lst = []
            image_repl_str_lst = []
            image_repl_ids_lst = []
            num_patches = []
            for (
                raw_img,
                img_patches,
                patch_newline_mask,
            ) in splitted_images_data:  # noqa: E501
                # 全局图 -> 728
                pixel_values_lst.extend(self._convert_images_to_pixel_values([raw_img]))

                # 局部 patch -> 504（可能一个都没有）
                if len(img_patches) > 0:
                    patch_pixel_values_lst.extend(
                        self._convert_images_to_pixel_values(img_patches, is_patch=True)
                    )
                num_patches.append(len(img_patches))

                # 生成这张图对应的占位 token 串
                image_repl_str, image_repl_ids = self._get_image_repl_features(
                    1, len(img_patches), patch_newline_mask
                )
                image_repl_str_lst.append(image_repl_str)
                image_repl_ids_lst.extend(image_repl_ids)

                if patch_newline_mask is not None:
                    patch_newline_mask_lst.extend(patch_newline_mask)

            image_inputs = {
                "pixel_values": torch.cat(pixel_values_lst),
                "num_patches": num_patches,
            }
            if patch_pixel_values_lst:
                image_inputs["patch_pixel_values"] = torch.cat(patch_pixel_values_lst)
            if patch_newline_mask_lst:
                image_inputs["patch_newline_mask"] = torch.tensor(
                    patch_newline_mask_lst, dtype=torch.bool
                )

            text = [
                # 把每条文本里的 <im_patch> 依次换成完整展开串，然后再整体分词。
                # 注意：上面攒的 image_repl_ids_lst 其实没被用上（属于冗余计算）。
                self.replace_placeholder(t, self.image_token, image_repl_str_lst)
                for t in text
            ]
            text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )


################################################


class Step3VLImageProcessor(SGLangBaseProcessor):
    """SGLang 侧的适配层：把上面的 Step3VLProcessor 接到 SGLang 的多模态流水线上。

    `models` 声明这个 processor 服务于哪些模型类，SGLang 启动时据此自动注册
    （见 sglang/srt/multimodal/processors/base_processor.py 的注册机制）。
    """

    models = [
        Step3VLForConditionalGeneration,
        StepVLForConditionalGeneration,
        Step3p7ForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # TODO, check _processor is tokenizer or processor.
        # _processor 由上游 HF AutoProcessor/AutoTokenizer 加载而来，
        # 这里再包一层自定义的 Step3VLProcessor
        processor = Step3VLProcessor(hf_config, _processor)
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.IM_TOKEN = "<im_patch>"
        self.IM_TOKEN_ID = self._processor.tokenizer.get_vocab()[self.IM_TOKEN]
        # 告诉通用流水线：图片占位符长什么样、对应哪个 token id，
        # 以便在 prompt 中定位并替换成实际的视觉特征
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IM_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(r"(?:<im_patch>)"),
        ).build(_processor)

        # NOTE: 下面两行是无用的局部变量（并未赋给 self），属于遗留代码
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    # NOTE: preprocess / __call__ 引用了不存在的 self.transform，是死代码。
    # 实际预处理走的是 self._processor（即 Step3VLProcessor）。
    def preprocess(self, image):
        return {"pixel_values": self.transform(image).unsqueeze(0)}

    def __call__(self, image):
        return self.preprocess(image)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text: str | List[int],
        request_obj,
        *args,
        **kwargs,
    ):
        """SGLang 调用的统一异步入口。

        两步：
          1. load_mm_data:  异步下载/解码图片（URL、base64、本地路径都在这里处理），
                            并按 mm_tokens 定位 prompt 中的占位符
          2. process_and_combine_mm_data: 调用 self._processor（Step3VLProcessor）
                            做真正的预处理，打包成 mm_items，
                            并拼出展开后的 input_ids

        ------------------------------------------------------------------
        第 2 步展开：process_and_combine_mm_data 到底做了什么
        （实现在 base_processor.py 的 BaseMultimodalProcessor 里，此处不重写）

        以「一段文本 + 1 张 1512x1008 的图」为例，走一遍全过程：

        (a) base_output.organize_results()
            把 load_mm_data 的结果按模态归类。这里得到
              raw_images = [PIL.Image(1512x1008)]，raw_audios/raw_videos 为空。
            若全是纯文本（没有任何多模态数据），直接 tokenizer 编码后返回
            ([], input_ids, {})，后面全部跳过。

        (b) _process_and_collect_mm_items(...) -> process_mm_data(...)
            这一步才真正调到 self._processor，也就是本文件的 Step3VLProcessor。
            process_mm_data 负责把参数塞成 HF processor 的调用约定：
              images -> kwargs["images"]，并把 self.image_config 合进 images_kwargs；
              videos -> kwargs["videos"]；audios 按模型名走 "audio"/"audios"。
            然后 `self._processor(text=..., images=..., return_tensors="pt")`，
            落到 Step3VLProcessor.__call__：
              - 切图：1 个 728x728 全局图 + N 个 504x504 局部 patch
              - 把 prompt 里的 1 个 <im_patch> 占位符，展开成
                局部 patch 段(每个 81 token) + 全局图段(169 token) + 换行
              - 返回 BatchFeature{input_ids, attention_mask, pixel_values, ...}

        (c) collect_mm_items_from_processor_output(ret)
            遍历 BatchFeature 的每个字段，用 ATTR_NAME_TO_MODALITY 表反查它属于
            哪个模态（pixel_values -> IMAGE，input_features -> AUDIO ...），
            同模态的字段塞进同一个 MultimodalDataItem。
            input_ids / format / hash / pad_value / offsets 是元数据，跳过不塞。
            注意：此时**每个模态只有 1 个 item**（所有图的 pixel_values 拼在一起）。

        (d) 回填 offsets
            对每个 item，用 mm_tokens 查到该模态的占位 token id（这里是
            IM_TOKEN_ID），在展开后的 input_ids 里扫描出所有连续区间：
              get_mm_items_offset(input_ids, mm_token_id)
            得到形如 [(12, 424)] 的 (start, end) 闭区间列表 —— 这些位置将来
            要被替换成视觉 embedding。

        (e) get_new_expanded_mm_items(all_collected_items)
            把 (c) 里「一个模态一个 item」再**按图/按视频拆成一个个独立 item**。
            目的是提升 radix cache 粒度：3 张图拆成 3 个 item 后，改动第 3 张
            图不会让前 2 张的缓存失效。拆完再 set_pad_value() 算哈希占位值。

        (f) 返回 (mm_items, input_ids, ret)
            input_ids 就是展开后的完整序列（占位符已按实际 token 数铺开），
            长度与最终喂给 LLM 的序列一致。

        ------------------------------------------------------------------
        几个容易踩的点：

        - Step3VLProcessor 是 HF 风格的 processor（本文件里自己实现的），
          而 Step3VLImageProcessor（本类）是 SGLang 的适配层。真正干活的是前者，
          本类只负责「异步加载 + 调度 + 打包成 SGLang 的数据结构」。

        - self.transform 那两个方法（preprocess/__call__）是死代码，
          走的根本不是它们，别被误导。

        - SGLANG_MM_AVOID_RETOKENIZE 那条分支只在「纯图 + 传入的是 token id 列表」
          时生效，用于避免 decode->re-tokenize 造成的 token 漂移。
          Step3 的常规文本输入不会走到。
        """
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # ret 是 Step3VLProcessor 的原始 BatchFeature 输出，此处不再需要
        # （信息已经分别进了 mm_items 和 input_ids），故直接丢弃。
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )
