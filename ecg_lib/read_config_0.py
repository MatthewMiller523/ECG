# -*- coding: utf-8 -*-
#=============================
import tomllib
from pathlib import Path

#get settings.toml information




class Config:
    def __init__(self, filename='settings.toml'):
        base_dir = Path(__file__).resolve().parent.parent
        settings_path = base_dir / filename
        
        with open(settings_path, 'rb') as f:
            config_dict = tomllib.load(f)
        
        self._content = dict(config_dict)

    @property
    def all(self):
        return self._content.copy()

    @property
    def general(self):
        return self._content['general'].copy()
    @property
    def model(self):
        return self._content['model'].copy()
    @property
    def data(self):
        return self._content['data'].copy()
    @property
    def meta(self):
        return self._content['meta'].copy()

