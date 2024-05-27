import re
import warnings


def get_configs(config_regex, runner, config_type=None):
    configs = [config for config in runner.config_type.__subclasses__()]

    config_re = re.compile(config_regex)
    configs = list(
        filter(
            lambda config: config.config_name is not None
            and config_re.match(config.config_name),
            configs,
        )
    )
    if len(configs) == 0:
        warnings.warn(
            f"Couldn't match regular expression '{config_regex}' to any configs for {runner}."
        )

    if config_type:
        configs = list(
            filter(lambda config: config.config_type == config_type, configs)
        )
        if len(configs) == 0:
            warnings.warn(
                f"Couldn't find configs with config_type '{config_type}' in {runner}."
            )

    return configs
