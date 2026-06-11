import os

# Change this one path for your machine / cluster account.
ROOT = '/home/hk-project-pai00116/id_glh1237'

def _p(*parts):
    return os.path.join(ROOT, *parts)

# Path to the downloaded CLIP official weights.
# See: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30
CLIP_VIT_B16_PATH = _p('Frame2Freq', 'ViT-B-16.pt')
CLIP_VIT_L14_PATH = _p('Frame2Freq', 'ViT-L-14.pt')

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
DWCONV3D_DISABLE_CUDNN = True

# Configuration of datasets.
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT=_p('data', 'ssv2', 'videos'),
        VAL_ROOT=_p('data', 'ssv2', 'videos'),
        TRAIN_LIST=_p('data', 'ssv2', 'labels', 'train_annotations.txt'),
        VAL_LIST=_p('data', 'ssv2', 'labels', 'val_annotations.txt'),
        NUM_CLASSES=174,
    ),
    'diving48': dict(
        TRAIN_ROOT=_p('data', 'diving48', 'rgb'),
        VAL_ROOT=_p('data', 'diving48', 'rgb'),
        TRAIN_LIST=_p('data', 'diving48', 'annotations', 'train_annotations.txt'),
        VAL_LIST=_p('data', 'diving48', 'annotations', 'test_annotations.txt'),
        NUM_CLASSES=48,
    ),
}