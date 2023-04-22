import re


def get_configs(config_regex, runner):
    config_re = re.compile(config_regex)
    configs = [
        config
        for config in runner.config_type.__subclasses__()
        if config.config_name is not None and config_re.match(config.config_name)
    ]
    if len(configs) == 0:
        raise ValueError(
            f"Couldn't match regular expression '{config_regex}' to any configs."
        )
    return configs
