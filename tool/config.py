import yaml
from tool.utils import download_config

url_config = {
    "vgg_transformer": "vgg-transformer.yml",
    "base": "base.yml",
}

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        base_config = download_config(url_config['base'])
        with open(fname, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config["base"])
        config = download_config(url_config[name])

        base_config.update(config)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, "w", encoding="utf-8") as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)