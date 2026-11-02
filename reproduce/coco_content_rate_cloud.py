"""Immutable content-rate codec worker binder."""
import json
from contextlib import contextmanager
from pathlib import Path
from . import tri_task_cloud as cloud
from .coco_content_rate_study import config

@contextmanager
def binding(cfg):
    if cfg!=config(cfg['task'],'control',cfg['codecs'][0]):raise ValueError('content codec configuration differs')
    old_config,old_command=cloud.config,cloud.COMMAND
    cloud.config=lambda task,arm:config(task,arm,cfg['codecs'][0])
    cloud.COMMAND=old_command.replace('reproduce.tri_task_study','reproduce.coco_content_rate_study')
    try:yield
    finally:cloud.config,cloud.COMMAND=old_config,old_command

def prepare(root,directory,account,slug,cfg):
    with binding(cfg):return cloud.prepare(root,directory,account,slug,cfg)

def verify(directory,root=None):
    cfg=json.loads((Path(directory)/'job.json').read_text())['config']
    with binding(cfg):return cloud.verify(directory,root)
