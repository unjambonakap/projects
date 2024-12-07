#!/usr/bin/env python

from pydantic import Field

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
import chdrft.utils.misc as cmisc


class ORMSettings(BaseSettings):
  url: str = ''

class FastApiSettings(BaseSettings):
  pass

env_path = os.environ.get('ENV_PATH', cmisc.path_here('../env/main.env'))

class RootSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_path,
      env_nested_delimiter='.',
    )

    orm : ORMSettings = ORMSettings()
    fastapi: FastApiSettings = FastApiSettings()


settings = RootSettings()
