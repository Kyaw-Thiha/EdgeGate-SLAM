import hydra
from omegaconf import DictConfig
from edgegate.training.trainer import train


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
