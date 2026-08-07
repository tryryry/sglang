import math
import os
import re
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.interns2preview import InternS2PreviewForConditionalGeneration
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.utils import logger

# ============================================================================
# Qwen-VL / Qwen-Omni 系列多模态预处理器
#
# 与 Step3-VL 的根本区别：Qwen-VL 走的是「原生动态分辨率」(Naive Dynamic
# Resolution)，**不切 patch、也没有全局缩略图**。
#
#   Step3-VL：固定 728 全局图(169 token) + 若干 504 局部 patch(各 81 token)
#   Qwen-VL ：整图直接缩放到「长宽都是 28 的倍数」，token 数随图大小浮动
#
# 为什么是 28？   28 = patch_size(14) x spatial_merge_size(2)
#   ViT 以 14x14 像素为一个 patch，再把相邻 2x2 个 patch 合并成 1 个 LLM token，
#   所以 28x28 像素 <=> 1 个 LLM token。
#
#   => token 数 = (H/28) x (W/28)
#
# 例子（1024x768 的图）：
#   smart_resize(768, 1024) -> (756, 1036)   # 都对齐到 28 的倍数
#   token 数 = 756/28 * 1036/28 = 27 * 37 = 999
#
# 第二个关键点：M-RoPE（多模态旋转位置编码）
#   普通 LLM 的位置是 1 维标量；Qwen-VL 用 3 维 (t, h, w)：
#     纯文本 token   -> t=h=w 同步递增，退化成普通 RoPE
#     图像 token     -> t 固定，h/w 按其在图中的二维坐标铺开
#     视频 token     -> t 随帧递增，h/w 同上
#   这样模型能同时感知「时间先后」和「空间上下左右」。
#   本文件里大量代码（get_rope_index / mrope_positions）都在算这个。
#
# 第三个关键点：视频抽帧
#   按 FPS 重采样 -> 限制帧数 -> 按「总像素预算 / 帧数」动态压缩每帧分辨率，
#   帧越多、每帧就越小，保证总 token 数可控。
#
# 本文件结构：
#   smart_resize / *_by_factor   尺寸对齐到 28 倍数的数学工具
#   smart_nframes / preprocess_video   视频抽帧与缩放
#   QwenVLImageProcessor         主处理器（图/视频/音频 + M-RoPE 计算）
# ============================================================================

# 28 = patch_size(14) * spatial_merge_size(2)，即「1 个 LLM token 对应的像素边长」
IMAGE_FACTOR = 28
# 一张图最少 4 个 token（4 = 2x2 的最小网格）
MIN_PIXELS = 4 * 28 * 28
# 一张图最多 16384 个 token（默认值，可用 SGLANG_IMAGE_MAX_PIXELS 调）
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
# 长宽比超过 200:1 直接报错（这种图缩放后必然有一边退化成 1 个 token）
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )
# 视频的总 token 预算（所有帧加起来），默认约 128000*0.9 ≈ 115200 个 token
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28  # 单帧下限 128 token
VIDEO_MAX_PIXELS = 768 * 28 * 28  # 单帧上限 768 token
FRAME_FACTOR = 2  # 帧数必须是 2 的倍数（ViT 时间维两帧一组）
FPS = 2.0  # 默认按 2 帧/秒重采样
FPS_MIN_FRAMES = 4  # 最少抽 4 帧
FPS_MAX_FRAMES = 768  # 最多抽 768 帧


_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
if _is_cpu and _is_cpu_amx_available:
    # CPU 推理场景：用 AMX 指令集加速的实现替换掉 HF 原生的图像预处理（猴子补丁）
    try:
        import transformers

        from sglang.srt.layers.amx_utils import fast_preprocess_cpu

        transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess = (
            fast_preprocess_cpu
        )
    except Exception as e:
        logger.warning(
            f"Failed to hack Qwen2VLImageProcessorFast with AMX optimization: {e}"
        )


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    ------------------------------------------------------------------
    中文详解：这是 Qwen-VL「原生动态分辨率」的核心。

    三条约束：
      1) H、W 都对齐到 factor(28) 的倍数  -> 才能整除成完整的 LLM token
      2) 总像素落在 [min_pixels, max_pixels] -> 控制单图 token 数上下限
      3) 尽量保持原始长宽比               -> 不拉伸变形

    算法：先各自四舍五入到 28 倍数；若超上限就按 sqrt(实际/上限) 等比缩小
    再向下取整；若低于下限就按 sqrt(下限/实际) 等比放大再向上取整。
    用 sqrt 是因为面积和边长是平方关系。

    例子 A（常规图 1024x768，不触发上下限）：
      h_bar = round(768/28)*28  = 27*28 = 756
      w_bar = round(1024/28)*28 = 37*28 = 1036
      面积 783,216 在区间内 -> 直接返回 (756, 1036)
      token 数 = 27 * 37 = 999

    例子 B（超大图 8000x6000，触发 max_pixels=16384*784）：
      h_bar=6000->5992, w_bar=8000->7994，面积 4789万 >> 上限 1284万
      beta = sqrt(4800万/1284万) ≈ 1.93
      h_bar = floor(6000/1.93 /28)*28 = 3108
      w_bar = floor(8000/1.93 /28)*28 = 4144
      token 数 = 111 * 148 = 16428（略超是因为 beta 用原始尺寸算的近似）

    例子 C（极小图 10x10，触发 min_pixels=4*784=3136）：
      h_bar = w_bar = max(28, round(10/28)*28) = max(28, 0) = 28
      面积 784 < 3136 -> beta = sqrt(3136/100) = 5.6
      h_bar = ceil(10*5.6 /28)*28 = 56，w_bar 同理 = 56
      token 数 = 2 * 2 = 4
    ------------------------------------------------------------------
    """
    # 长宽比过于极端时，缩放后必然有一边退化成 1 个 token，直接拒绝
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    # 第一步：各自四舍五入到 factor 倍数（max(factor, ...) 保证至少 1 格，不会变成 0）
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        # 超上限：等比缩小。beta 是边长缩放系数，用 floor 保证缩完不会再超
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        # 低于下限：等比放大。用 ceil 保证放完不会还不够
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    """四舍五入到 factor 的倍数。例：round_by_factor(768, 28) = 756"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """向上取整到 factor 的倍数。例：ceil_by_factor(768, 28) = 784"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """向下取整到 factor 的倍数。例：floor_by_factor(768, 28) = 756"""
    return math.floor(number / factor) * factor


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.

    ------------------------------------------------------------------
    中文详解：决定一个视频要抽多少帧。两种模式二选一：
      - nframes: 直接指定帧数（只对齐到 FRAME_FACTOR=2 的倍数）
      - fps:     按目标帧率重采样，再夹到 [min_frames, max_frames]

    例子（60 秒、30fps 的视频，共 1800 帧，默认 fps=2）：
      nframes = 1800 / 30 * 2 = 120
      夹到 [4, min(768, 1800)] -> 120
      对齐到 2 的倍数 -> 120 帧

    例子（1 秒、30fps 的短视频，共 30 帧）：
      nframes = 30/30*2 = 2，但 min_frames=4 -> 提到 4 帧
    ------------------------------------------------------------------
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        # 模式一：用户直接指定帧数
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        # 模式二：按帧率重采样
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        # 时长(秒) * 目标帧率 = 应抽帧数
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        # 依次夹到 [min_frames, max_frames] 和 total_frames
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
) -> torch.Tensor:
    """视频预处理：抽帧 -> 转 TCHW -> 按「总 token 预算 / 帧数」动态定分辨率 -> 缩放。

    关键权衡：帧数越多，每帧能分到的像素预算就越少。
      max_pixels = clamp(总预算/帧数*2, 下限=min_pixels*1.05, 上限=768 token)

    例子（120 帧，总预算 VIDEO_TOTAL_PIXELS ≈ 115200 token）：
      每帧预算 = 115200/120*2 = 1920 token，但单帧上限 768 -> 取 768
      再经 smart_resize 对齐到 28 倍数，比如 640x360 -> (364, 644) = 13*23 = 299 token
      总计 120 * 299 ≈ 35880 token

    返回 (缩放后的视频 tensor [T,C,H,W], 元数据 dict)。
    """
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
        # 已经是预处理好的数据（比如上游直接传了 tensor/dict），原样返回
        return vr, None
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.avg_fps

    # 第一步：决定抽多少帧
    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    # 在整个视频上均匀取 nframes 个下标（unique 防止短视频出现重复下标）
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    video = vr.get_frames_as_tensor(idx.tolist())

    video = video.permute(0, 3, 1, 2)  # NHWC -> TCHW

    nframes, _, height, width = video.shape
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = video_config.get("total_pixels", VIDEO_TOTAL_PIXELS)
    # 第二步：单帧像素预算 = min(单帧上限, 总预算摊到每帧)，再兜一个下限
    max_pixels = max(
        min(
            video_config.get("max_pixels", VIDEO_MAX_PIXELS),
            total_pixels / nframes * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )

    get_batch_time = time.perf_counter()

    max_pixels_supposed = video_config.get("max_pixels", max_pixels)

    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in video_config and "resized_width" in video_config:
        # 用户显式指定了目标尺寸，只做 28 对齐，不再受像素预算约束
        resized_height, resized_width = smart_resize(
            video_config["resized_height"],
            video_config["resized_width"],
            factor=image_factor,
        )
    else:
        # 第三步：按预算算出实际缩放尺寸
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    smart_resize_time = time.perf_counter()
    # 整批帧一次性 resize（[T,C,H,W] 上直接做，比逐帧快）
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
    # 锁页内存，后续 H2D 拷贝可以走异步 DMA
    video = video.pin_memory()
    # 元数据给下游算时间戳/M-RoPE 的时间维用
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }
    torchvision_resize_time = time.perf_counter()
    logger.debug(
        f"[preprocess_video Perf], "
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "
        f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "
        f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "
        f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"
    )
    return video, video_metadata


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):
    """Qwen 全系视觉/多模态处理器，覆盖 Qwen2-VL ~ Qwen3.5、Qwen3-Omni、InternS2。

    职责有两块：
      1) 图/视频/音频的预处理（大部分复用 HF 的 processor，本类只做编排）
      2) **M-RoPE 位置计算** —— 本文件最复杂的部分

    关于 M-RoPE（3 维位置 t/h/w）：
      文本 token：t=h=w，同步递增，等价于普通 RoPE
      图像 token：t 固定不变，h/w 铺成二维网格
      视频 token：t 随帧递增，h/w 同图像
    位置编码要和 input_ids 严格对齐，所以本类有 3 条计算路径（见 __init__ 下方各方法）。
    """

    supports_transformers_backend = True
    # 一个 processor 服务这么多模型，因为它们共享同一套视觉编码方案
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ForCausalLMMTP,
        InternS2PreviewForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        if hf_config.model_type == "qwen3_omni_moe":
            # Omni 是「thinker + talker」双塔结构，视觉相关配置都在 thinker 里
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # 视觉区间的边界标记：<|vision_start|> ... <|vision_end|>
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        # 音频相关（只有 Omni 系列才有，其余模型为 None）
        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        # 2x2 的 patch 合并系数：ViT 输出的 4 个 patch 合成 1 个 LLM token
        self._spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        # 视频时间维每秒对应多少个位置 id（M-RoPE 的 t 维用）
        self._tokens_per_second = getattr(
            self.hf_config.vision_config, "tokens_per_second", None
        )

        # 告诉通用流水线怎么在 prompt 里定位图片占位符。
        # 注意 image_token 是三段式：<|vision_start|><|image_pad|><|vision_end|>；
        # 而 regex 匹配的是**已展开**的形态（中间的 image_pad 会重复 N 次）。
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    @property
    def spatial_merge_size(self):
        return self._spatial_merge_size

    def build_input_ids_with_timestamps(
        self, prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
    ):
        """
        Build input_ids with timestamps for qwen3_vl models.

        ------------------------------------------------------------------
        中文详解：qwen3-vl 的视频输入会在**每一帧前面插入时间戳文本**，
        让模型知道这帧发生在第几秒。

        单帧展开后的形态：
            "<2.5 seconds>" <|vision_start|> {frame_seqlen 个 video_pad} <|vision_end|>

        因此原始 prompt 里的 1 个视频占位符，会被展开成 num_frames 组上述结构；
        每一帧被当作**独立的一个 mm_item**（modality_list 会 append 多次）。

        图像分支则简单得多：直接把占位符替换成 mm_token_num 个 image_pad。
          mm_token_num = t*h*w / merge^2

        返回 (展开后的 input_ids, 每段视觉 token 的 [start,end] 偏移, 模态列表)。
        offsets 后续用来把预计算好的视觉 embedding 切片对号入座。
        ------------------------------------------------------------------
        """
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        spatial_merge_size = self.spatial_merge_size
        vision_start_token_id = getattr(self, "vision_start_token_id", None)
        vision_end_token_id = getattr(self, "vision_end_token_id", None)

        input_ids = []
        offsets = []
        modality_list = []
        cur_idx = 0  # 原 prompt 中已经拷贝到 input_ids 的位置

        # 扫描 prompt，找出所有 <|vision_start|> 的下标
        # （判据是「下一个 token 是 image_pad 或 video_pad」）
        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))

        img_idx = 0
        video_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            modality_list.append(modality)
            video_tokens = None
            if modality == Modality.IMAGE:
                # 图像：token 数 = t*h*w / merge^2
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                curr_timestamps = video_timestamps[video_idx]
                num_frames = video_grid_thw[video_idx][0]
                # 单帧 token 数 = h*w / merge^2
                frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (
                    spatial_merge_size**2
                )
                video_tokens = []
                # _current_offset 追踪「当前帧的视觉 token 在最终 input_ids 里的绝对位置」
                _current_offset = len(input_ids) + mm_start_idx + 1 - cur_idx
                # take single frame as one mm_item
                for frame_idx in range(num_frames):
                    if frame_idx > 0:
                        # 除第一帧外，其余帧各自再登记一次模态（一帧 = 一个 mm_item）
                        modality_list.append(Modality.VIDEO)
                    curr_time = curr_timestamps[frame_idx]
                    # 时间戳文本，例如 "<2.5 seconds>"
                    timestamp_text = f"<{curr_time:.1f} seconds>"
                    timestamp_tokens = self._processor.tokenizer.encode(
                        timestamp_text, add_special_tokens=False
                    )
                    video_tokens.extend(timestamp_tokens)
                    _current_offset += len(timestamp_tokens)
                    if vision_start_token_id is not None:
                        video_tokens.append(vision_start_token_id)
                        _current_offset += 1
                    # 这一帧的视觉占位 token
                    video_tokens.extend([video_token_id] * frame_seqlen)
                    if vision_end_token_id is not None:
                        video_tokens.append(vision_end_token_id)
                    # 记录本帧视觉 token 的闭区间 [start, end]
                    offsets.append(
                        (_current_offset, _current_offset + frame_seqlen - 1)
                    )
                    _current_offset += (
                        frame_seqlen + 1
                        if vision_end_token_id is not None
                        else frame_seqlen
                    )  # for vision_end_token_id
                mm_token_num = len(video_tokens)
                mm_token_id = None
                video_idx += 1
            else:
                logger.warning(
                    f"{modality} modality is not supported for qwen3_vl models with timestamps."
                )
                continue
            assert cur_idx <= mm_start_idx
            # 先把占位符之前的普通文本原样拷过来（含 <|vision_start|> 本身）
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            if modality == Modality.VIDEO:
                input_ids.extend(video_tokens)
            else:
                mm_offset_start = len(input_ids)
                input_ids.extend([mm_token_id] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
            cur_idx = mm_start_idx + 2  # jump to vision_end_id
        else:
            # for-else：循环正常结束后把剩余的尾部文本补上
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def compute_mrope_positions(self, input_ids, mm_items):
        """标准路径：把各 mm_item 的 grid_thw 汇总后，交给 MRotaryEmbedding 统一算 M-RoPE。"""
        image_grid_thw = self._concat_mm_item_grid(
            mm_items, "image_grid_thw", Modality.IMAGE
        )
        video_grid_thw = self._concat_mm_item_grid(
            mm_items, "video_grid_thw", Modality.VIDEO
        )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=self._tokens_per_second,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    @staticmethod
    def _get_processor_output_value(ret, key):
        """兼容取值：HF processor 的返回可能是 dict-like，也可能是普通对象。"""
        if ret is None:
            return None
        return ret.get(key) if hasattr(ret, "get") else getattr(ret, key, None)

    def _get_precomputed_mrope_from_output(self, ret):
        """最快路径：如果 HF processor 已经算好了 M-RoPE，直接拿来用，省一次计算。

        同时做严格的形状校验，任一项不符就返回 None 回退到自己算：
          mrope_positions 必须能规约成 [3, seq_len]（3 = t/h/w 三个维度）
        """
        mrope_positions = self._get_processor_output_value(ret, "mrope_positions")
        mrope_position_delta = self._get_processor_output_value(
            ret, "mrope_position_delta"
        )
        if mrope_positions is None or mrope_position_delta is None:
            return None

        mrope_positions = torch.as_tensor(mrope_positions)
        if mrope_positions.ndim == 3:
            # [3, batch, seq] -> 只接受 batch==1，squeeze 掉
            if mrope_positions.shape[1] != 1:
                return None
            mrope_positions = mrope_positions.squeeze(1)
        if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
            return None

        mrope_position_delta = torch.as_tensor(mrope_position_delta)
        if mrope_position_delta.ndim <= 1:
            mrope_position_delta = mrope_position_delta.reshape(-1, 1)
        return mrope_positions, mrope_position_delta

    @staticmethod
    def _as_grid_batch(value):
        """把 grid_thw 规整成 [N, 3] 的 batch 形式（单个 [3] 会补上 batch 维）。"""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0) if value.ndim == 1 else value
        tensor = torch.as_tensor(value, dtype=torch.long)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def _compute_image_only_mrope_positions_from_offsets(
        self,
        input_len: int,
        mm_items: List[MultimodalDataItem],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """instead of calling get_rope_index, build mrope position from mm_items.offsets and image_grid_thw of each image
        basically a simplified version of get_rope_index for image-only reqs

        ------------------------------------------------------------------
        中文详解：**纯图像请求的快速路径**。

        通用的 get_rope_index 要扫一遍 input_ids 找特殊 token，开销不小。
        而纯图场景下，每张图的位置区间 (offsets) 和网格 (grid_thw) 都是已知的，
        可以直接拼出位置张量，不用扫描。

        构造方式：把序列切成「文本段 / 图像段」交替，逐段生成位置：
          文本段：t=h=w 同步递增（退化成普通 RoPE）
          图像段：t 恒为 0 偏移，h 按行号、w 按列号铺开

        关键细节 —— 段与段之间的位置衔接：
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
          图像占了 h*w 个 token，但位置只前进 max(t,h,w)。
          因为图内 h/w 是并行展开的（同一行的 token 共享 h），
          所以图像整体只「消耗」了 max 这么多个位置刻度。

        例子（文本 5 token + 一张 2x3 网格的图 + 文本 4 token）：
          文本段:  t/h/w = 0,1,2,3,4          -> next_pos = 5
          图像段:  t = [0,0,0,0,0,0] + 5
                   h = [0,0,0,1,1,1] + 5
                   w = [0,1,2,0,1,2] + 5      -> next_pos += max(1,2,3)=3 -> 8
          文本段:  t/h/w = 8,9,10,11

        任何一步校验不过（有非图 item、offsets 不唯一、token 数对不上）
        就返回 None，让调用方回退到通用实现。
        ------------------------------------------------------------------
        """
        # 只有这几个模型的 rope 规则与本简化实现一致
        if self.model_type not in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
        ):
            return None

        # 必须「全是图像」才能走快速路径；混了视频/音频就放弃
        image_items = [item for item in mm_items if item.is_image()]
        if not image_items or len(image_items) != len(mm_items):
            return None

        spatial_merge_size = self._spatial_merge_size
        # 按在序列中出现的先后排序
        sorted_items = sorted(image_items, key=lambda item: item.offsets[0][0])
        position_segments = []
        st = 0  # 已处理到的序列位置
        next_pos = 0  # 下一段应该从哪个位置刻度开始

        for item in sorted_items:
            if item.offsets is None or len(item.offsets) != 1:
                return None

            start, end = item.offsets[0]
            if start < st or end >= input_len:
                return None

            # --- 图像之前的文本段：t/h/w 同步递增 ---
            text_len = start - st
            if text_len > 0:
                position_segments.append(
                    torch.arange(text_len, dtype=dtype, device=device)
                    .view(1, -1)
                    .expand(3, -1)
                    + next_pos
                )
                next_pos += text_len

            grid = self._as_grid_batch(item.model_specific_data.get("image_grid_thw"))
            if grid is None or grid.shape[0] != 1:
                return None
            t, h, w = [int(x) for x in grid[0].tolist()]
            # grid 记录的是 ViT patch 数，除以 merge 才是 LLM token 数
            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            # 一致性校验：算出的 token 数必须和 offsets 区间长度吻合
            if num_image_tokens != end - start + 1:
                return None

            # --- 图像段：t/h/w 三个维度各自铺开 ---
            # t: 每帧内所有 token 相同 -> [0]*hw, [1]*hw, ...
            t_index = (
                torch.arange(llm_grid_t, dtype=dtype, device=device)
                .view(-1, 1)
                .expand(llm_grid_t, llm_grid_h * llm_grid_w)
                .reshape(-1)
            )
            # h: 同一行内相同 -> [0]*w, [1]*w, ...
            h_index = (
                torch.arange(llm_grid_h, dtype=dtype, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            # w: 每行内 0..w-1 循环
            w_index = (
                torch.arange(llm_grid_w, dtype=dtype, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            position_segments.append(
                torch.stack([t_index, h_index, w_index]) + next_pos
            )
            # 图像整体只前进 max(t,h,w) 个位置刻度（三个维度是并行的，不是串行累加）
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
            st = end + 1

        # --- 最后一张图之后的尾部文本 ---
        if st < input_len:
            text_len = input_len - st
            position_segments.append(
                torch.arange(text_len, dtype=dtype, device=device)
                .view(1, -1)
                .expand(3, -1)
                + next_pos
            )

        mrope_positions = torch.cat(position_segments, dim=1).unsqueeze(1)
        # delta = 生成阶段续写时，位置 id 相对 token 下标的偏移量
        mrope_position_delta = (mrope_positions.max() + 1 - input_len).reshape(1, 1)
        return mrope_positions, mrope_position_delta

    @classmethod
    def _concat_mm_item_grid(cls, mm_items: list[MultimodalDataItem], key, modality):
        """把同一模态下所有 item 的 grid_thw 沿 batch 维拼成 [N, 3]。"""
        grids = []
        for item in mm_items:
            if not item.is_modality(modality):
                continue
            grid = cls._as_grid_batch(item.model_specific_data.get(key))
            if grid is not None:
                grids.append(grid)
        if not grids:
            return None
        if len(grids) == 1:
            return grids[0]
        return torch.cat(grids, dim=0)

    @classmethod
    def _get_grid_from_output_or_items(
        cls, ret, mm_items, key, modality, input_data=None
    ):
        """三级回退取 grid_thw：processor 输出 -> mm_items 汇总 -> 原始输入 dict。"""
        grid = cls._get_processor_output_value(ret, key)
        if grid is None:
            grid = cls._concat_mm_item_grid(mm_items, key, modality)
        if grid is None and input_data and isinstance(input_data[0], dict):
            grid = input_data[0].get(key)
        return grid

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """**预计算 embedding 路径**：外部已经把视觉特征算好了，这里只负责
        展开 input_ids、算 M-RoPE、并把 embedding 按 offsets 切片分配给各 mm_item。

        与 process_mm_data_async 的区别：
          process_mm_data_async  从原始图片/视频出发，走完整预处理
          get_mm_data            从已有的 embeddings 出发（例如 PD 分离、跨进程复用场景）
        """
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)
        video_timestamps = kwargs.get("video_timestamps", None)
        second_per_grid_ts = kwargs.get("second_per_grid_ts", None)

        audio_seq_lens = None
        if audio_feature_lens is not None:
            # 音频经过多层卷积下采样后的实际序列长度，各模型公式不同
            if self.model_type == "qwen3_omni_moe":
                # apply _get_feat_extract_lengths to get seq_lens
                input_lengths_leave = audio_feature_lens % 100
                feat_lengths = (input_lengths_leave - 1) // 2 + 1
                audio_seq_lens = (
                    ((feat_lengths - 1) // 2 + 1 - 1) // 2
                    + 1
                    + (audio_feature_lens // 100) * 13
                )
            elif self.model_type == "qwen2_5_omni":
                # 两次 stride=2 的下采样
                audio_seq_lens = (audio_feature_lens - 1) // 2 + 1
                audio_seq_lens = (audio_seq_lens - 2) // 2 + 1

        # qwen3-vl 系列 + 有视频时间戳 -> 走带时间戳的展开逻辑
        if (
            self.model_type
            in [
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
                "intern_s2_preview",
            ]
            and video_timestamps is not None
        ):
            input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(
                prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
            )
        else:
            input_ids, offsets, modality_list = self.build_input_ids(
                prompt, img_grid_thw, video_grid_thw, audio_seq_lens=audio_seq_lens
            )
        assert all(isinstance(modality, Modality) for modality in modality_list)

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=(
                audio_feature_lens if self.model_type == "qwen3_omni_moe" else None
            ),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
            tokens_per_second=self._tokens_per_second,
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_items = []
        # 记录每种模态的 embedding 已经被消费到哪个位置
        consumed_per_modality = {}

        # 按 offsets 顺序，把大块 embedding 切成一段段分给各 mm_item
        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        """SGLang 的标准异步入口，完整流程如下：

          1. load_mm_data      异步下载/解码图片、视频、音频
          2. preprocess_video  视频抽帧 + 缩放（qwen3-vl 在此完成，故后面 do_sample_frames=False）
          3. process_and_combine_mm_data  调 HF processor 做归一化，展开 input_ids
          4. 算 M-RoPE         三级回退：processor 已算好 -> 纯图快速路径 -> 通用 get_rope_index
          5. 打包 MultimodalProcessorOutput

        全程用 time.perf_counter() 打点，logger.debug 输出各阶段耗时。
        """
        entry_time = time.perf_counter()
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        load_time = time.perf_counter()
        rid = getattr(request_obj, "rid", "anonymous_rid")

        video_metadata = None
        # 只有拿到的是原始视频对象（不是已预处理的 dict）才需要抽帧
        if base_output.videos and not isinstance(base_output.videos[0], dict):
            videos_processed = [
                await preprocess_video(video, video_config=self.video_config)
                for video in base_output.videos
            ]
            # [(v1, m1), (v2, m2)] -> ([v1, v2], [m1, m2])
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        preprocess_time = time.perf_counter()

        # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
        if self.hf_config.model_type in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
        ):
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                video_metadata=video_metadata,
                do_sample_frames=False,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

        audio_feature_lengths = None

        if self.model_type == "qwen3_omni_moe":
            # Omni 才有音频；feature_attention_mask 求和得到每条音频的真实长度
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1
                )

        # 视频每个时间网格对应多少秒（M-RoPE 时间维要用），不同版本字段名不同
        second_per_grid_ts = self._get_processor_output_value(ret, "second_per_grid_ts")
        if second_per_grid_ts is None:
            second_per_grid_ts = self._get_processor_output_value(
                ret, "video_second_per_grid"
            )

        process_time = time.perf_counter()

        input_ids = input_ids.flatten()
        base_input_ids = getattr(base_output, "input_ids", None)
        if (
            isinstance(base_input_ids, list)
            and len(base_input_ids) == input_ids.numel()
        ):
            # reuse preprocess input if it already carries list of input_ids
            # 复用现成的 list，省一次 tensor->list 转换
            input_ids_list = base_input_ids
        else:
            input_ids_list = input_ids.tolist()

        # look for if padded_input_ids already exists before computing
        # padded_input_ids：把视觉占位 token 替换成特殊哨兵值的版本，
        # 用于 radix cache 前缀匹配时区分「同样的占位符但内容不同的图」
        padded_input_ids = self._get_processor_output_value(ret, "padded_input_ids")
        if padded_input_ids is None:
            padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            )
        elif isinstance(padded_input_ids, torch.Tensor):
            # reuse existing padded_input_ids
            padded_input_ids = padded_input_ids.flatten().tolist()
        else:
            padded_input_ids = list(padded_input_ids)

        image_grid_thw = self._get_grid_from_output_or_items(
            ret, mm_items, "image_grid_thw", Modality.IMAGE, image_data
        )
        video_grid_thw = self._get_grid_from_output_or_items(
            ret,
            mm_items,
            "video_grid_thw",
            Modality.VIDEO,
            request_obj.video_data,
        )

        mrope_result = self._get_precomputed_mrope_from_output(ret)
        if mrope_result is None:
            # 第二级：纯图像请求（无视频、无音频）走快速路径
            if (
                video_grid_thw is None
                and second_per_grid_ts is None
                and audio_feature_lengths is None
            ):
                mrope_result = self._compute_image_only_mrope_positions_from_offsets(
                    input_len=input_ids.numel(),
                    mm_items=mm_items,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
        if mrope_result is None:
            # 第三级：通用实现，扫描 input_ids 逐 token 判定模态，最慢但最全
            mrope_result = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self._spatial_merge_size,
                image_token_id=self.mm_tokens.image_token_id,
                video_token_id=self.mm_tokens.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                tokens_per_second=self._tokens_per_second,
                # use the expanded token ids
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_audio_in_video=False,
                audio_seqlens=audio_feature_lengths,
                audio_token_id=getattr(self.hf_config, "audio_token_id", None),
                audio_start_token_id=self.audio_start_token_id,
                position_id_per_seconds=getattr(
                    self.hf_config, "position_id_per_seconds", None
                ),
            )

        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)
        get_rope_index_time = time.perf_counter()
        logger.debug(
            f"[QwenVLProcessor Perf] {rid=}, "
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "
            f"preprocess_time: {(preprocess_time - load_time) * 1000:.2f} ms, "
            f"process_time: {(process_time - preprocess_time) * 1000:.2f} ms, "
            f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "
            f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=padded_input_ids,
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
