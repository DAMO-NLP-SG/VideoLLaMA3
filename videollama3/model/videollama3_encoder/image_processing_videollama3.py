# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py.
# Below is the original copyright:
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for VideoLLaMA3."""

import math
from typing import Dict, List, Optional, Union

import numpy as np

import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
)
from transformers.video_utils import VideoInput
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def is_valid_video(video) -> bool:
    if isinstance(video, (list, tuple)):
        return all(is_valid_image(frame) for frame in video)
    elif isinstance(video, np.ndarray):
        return video.ndim == 4
    elif isinstance(video, torch.Tensor):
        return video.ndim == 4
    return False


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)):
        # list of images/videos
        if not all(is_valid_video(image) or is_valid_image(image) for image in images):
            raise ValueError(f"Could not make batched images from {images}")
        return images
    elif is_valid_video(images) or is_valid_image(images):
        # single image/video
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


def simple_batched_resize(
    images, factor: int = 28, min_tokens: int = 4 * 4, max_tokens: int = 16384, input_data_format: str = None
):
    min_pixels = min_tokens * factor * factor
    max_pixels = max_tokens * factor * factor

    num_images = 0
    for image in images:
        if is_valid_video(image):
            num_images += len(image)
        else:
            num_images += 1

    image_sizes = []
    for image in images:
        if is_valid_video(image):
            image = image[0]
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = get_image_size(image, channel_dim=input_data_format)
        image_sizes.append([height, width])

    tmp_image_sizes = []
    for height, width in image_sizes:
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > (max_pixels // num_images):
            beta = math.sqrt((height * width) / (max_pixels // num_images))
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        # per image min_pixels
        if h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        tmp_image_sizes.append((h_bar, w_bar))
    image_sizes = tmp_image_sizes
    return image_sizes


def batched_resize(
    images, factors: List[int], min_tokens: int = 4 * 4, max_tokens: int = 16384, input_data_format: str = None
):
    image_sizes = []
    for image in images:
        if is_valid_video(image):
            num_frame = len(image)
            image = image[0]
        else:
            num_frame = 1
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = get_image_size(image, channel_dim=input_data_format)
        image_sizes.append([num_frame, height, width])

    # global max_pixels
    smart_scale_factors = 1.0
    total_tokens = 0
    for (num_frame, height, width), factor in zip(image_sizes, factors):
        total_tokens += num_frame * math.ceil(height / factor) * math.ceil(width / factor)

    # TODO: add min_pixels
    if total_tokens > max_tokens:
        beta = math.sqrt(total_tokens / max_tokens)
        tmp_image_sizes = []
        for (_, height, width), factor in zip(image_sizes, factors):
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
            tmp_image_sizes.append((h_bar, w_bar))
        image_sizes = tmp_image_sizes
    else:
        tmp_image_sizes = []
        for (_, height, width), factor in zip(image_sizes, factors):
            height = round(height / factor) * factor
            width = round(width / factor) * factor
            tmp_image_sizes.append((height, width))
        image_sizes = tmp_image_sizes

    return image_sizes


class Videollama3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DAMOVL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
    """

    model_input_names = ["pixel_values", "grid_sizes", "merge_sizes"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_tokens: int = 4 * 4,
        max_tokens: int = 16384,
        patch_size: int = 14,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.patch_size = patch_size
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        target_size: List[int],
        merge_size: int = 1,
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            target_size (`List[int]`):
                The target size to resize the image to. Should be a list of two integers: [target_height, target_width].
            merge_size (`int`, *optional*, defaults to `1`):
                The merge size after the vision encoder.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = target_size
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        t = patches.shape[0]
        channel = patches.shape[1]
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            t,
            channel,
            grid_h // merge_size,
            merge_size,
            self.patch_size,
            grid_w // merge_size,
            merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = patches.reshape(
            t * grid_h * grid_w, channel * self.patch_size * self.patch_size
        )

        return flatten_patches, (t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        merge_size: Optional[Union[int, List[int]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_batched_images(images)

        if isinstance(merge_size, (list, tuple)):
            assert len(merge_size) == len(images), "Merge size must be the same length as images."
            merge_sizes = merge_size
        else:
            merge_sizes = [merge_size for _ in images]

        if all(merge_size == merge_sizes[0] for merge_size in merge_sizes):
            target_sizes = simple_batched_resize(
                images,
                factor=self.patch_size * merge_sizes[0],
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                input_data_format=input_data_format,
            )
        else:
            target_sizes = batched_resize(
                images,
                factors=[self.patch_size * merge_size for merge_size in merge_sizes],
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                input_data_format=input_data_format,
            )

        pixel_values, grid_sizes = [], []
        for image, merge_size, target_size in zip(images, merge_sizes, target_sizes):
            patches, grid_size = self._preprocess(
                image,
                target_size=target_size,
                merge_size=merge_size,
                do_resize=do_resize,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
            )
            pixel_values.append(patches)
            grid_sizes.append(grid_size)

        pixel_values = np.concatenate(pixel_values, axis=0)
        grid_sizes = np.array(grid_sizes)
        merge_sizes = np.array(merge_sizes)

        data = {
            "pixel_values": pixel_values,
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)
