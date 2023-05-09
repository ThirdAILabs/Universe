import os
import re
import warnings


def check_if_experimental_mode():
    def str_to_bool(s):
        return s.lower() in ["true", "1", "t", "y", "yes"]

    experimental_autotune = os.environ.get("experimental_autotune", "false")
    experimental_autotune = str_to_bool(experimental_autotune)
    return experimental_autotune


def get_configs(config_regex, runner):
    config_re = re.compile(config_regex)

    configs = [
        config
        for config in runner.config_type.__subclasses__()
        if config.config_name is not None and config_re.match(config.config_name)
    ]
    if len(configs) == 0:
        warnings.warn(
            f"Couldn't match regular expression '{config_regex}' to any configs for {runner}."
        )
    return configs
