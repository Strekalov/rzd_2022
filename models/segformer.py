import transformers

from data import dataset_config
from schemas import Config


def get_model(config: Config):
    pretrain_name = config.model.pretrain_name
    print(f"UPLOAD {pretrain_name}")
    return transformers.SegformerForSemanticSegmentation.from_pretrained(
        pretrain_name,
        num_labels=dataset_config.NUM_CLASSES,
        id2label=dataset_config.ID_TO_LABEL,
        label2id=dataset_config.LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )


def get_feature_extractor(config: Config) -> transformers.SegformerFeatureExtractor:
    pretrain_name = config.model.pretrain_name
    size = config.dataset.input_size

    return transformers.SegformerFeatureExtractor().from_pretrained(
        pretrain_name,
        size=size,
        # size=(1305, 768),
        num_labels=dataset_config.NUM_CLASSES,
        id2label=dataset_config.ID_TO_LABEL,
        label2id=dataset_config.LABEL_TO_ID,
        # reduce_labels=True
    )
