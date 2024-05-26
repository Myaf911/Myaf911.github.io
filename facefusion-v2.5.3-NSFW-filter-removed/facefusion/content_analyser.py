from typing import Any
from functools import lru_cache
from time import sleep
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import facefusion.globals
facefusion.globals.skip_download = True  # 1 Отключение загрузки модели

from facefusion import process_manager, wording
from facefusion.thread_helper import thread_lock, conditional_thread_semaphore
from facefusion.typing import VisionFrame, Fps
from facefusion.execution import apply_execution_provider_options
from facefusion.vision import get_video_frame, count_video_frame_total, read_image, detect_video_fps
from facefusion.filesystem import resolve_relative_path, is_file
from facefusion.download import conditional_download

CONTENT_ANALYSER = None
# MODELS: ModelSet = None  # 2 Comment out or delete

#{      # 3 This piece I deleted differently didn't work
#	'open_nsfw':
#	{
#		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
#		'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
#	}
#}



PROBABILITY_LIMIT = 0.80 
RATE_LIMIT = 10
STREAM_COUNTER = 0


def get_content_analyser() -> Any:
    global CONTENT_ANALYSER

    with thread_lock():
        while process_manager.is_checking():
            sleep(0.5)
        if CONTENT_ANALYSER is None:
# 4     # model_path = MODELS.get('open_nsfw').get('path')  # Comment out or delete
            CONTENT_ANALYSER = None  # 5 Initialize with empty value
    return CONTENT_ANALYSER


def clear_content_analyser() -> None:
    global CONTENT_ANALYSER

    CONTENT_ANALYSER = None


def pre_check() -> bool:
# 6 # download_directory_path = resolve_relative_path('../.assets/models')  # Comment out or delete
    # model_url = MODELS.get('open_nsfw').get('url')  # Comment out or delete
    # model_path = MODELS.get('open_nsfw').get('path')  # Comment out or delete
    
    # if not facefusion.globals.skip_download:  # Comment out or delete
    #     process_manager.check()  # Comment out or delete
    #     conditional_download(download_directory_path, [model_url])  # Comment out or delete
    #     process_manager.end()  # Comment out or delete
    # return is_file(model_path)  # Comment out or delete
    return True  # Return True to skip the check


def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    global STREAM_COUNTER

    STREAM_COUNTER = STREAM_COUNTER + 1
    if STREAM_COUNTER % int(video_fps) == 0:
        return False  # 7 Instead of analyse_frame(vision_frame)
    return False

     # 9 The code below this line has been heavily modified. 
     #(well, maybe not heavily, but I spent a long time figuring out to get everything to work in the end :( )

def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    vision_frame = cv2.resize(vision_frame, (224, 224)).astype(numpy.float32)
    vision_frame -= numpy.array([104, 117, 123]).astype(numpy.float32)
    vision_frame = numpy.expand_dims(vision_frame, axis=0)
    return vision_frame


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    frame = read_image(image_path)
    return False  # 8 Instead of analyse_frame(frame)


@lru_cache(maxsize = None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    video_frame_total = count_video_frame_total(video_path)
    video_fps = detect_video_fps(video_path)
    frame_range = range(start_frame or 0, end_frame or video_frame_total)
    rate = 0.0
    counter = 0

    with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame', ascii=' =',
              disable=facefusion.globals.log_level in ['warn', 'error']) as progress:
        for frame_number in frame_range:
            if frame_number % int(video_fps) == 0:
                frame = get_video_frame(video_path, frame_number)
            rate = counter * int(video_fps) / len(frame_range) * 100
            progress.update()
            progress.set_postfix(rate=rate)
    return rate > RATE_LIMIT