

from collections import OrderedDict


LEGACY_KEY_RENAMES = {
    "mask_decoder.hq_token": "mask_decoder.expert_tokens",
    "mask_decoder.hq_mlp": "mask_decoder.expert_mlp",
    "mask_decoder.output_hypernetworks_mlps_hq": "mask_decoder.expert_output_mlps",
    "mask_decoder.embedding_encoder": "mask_decoder.expert_embedding_encoder",
    "mask_decoder.embedding_maskfeature": "mask_decoder.expert_feature_fusion",
}


def upgrade_legacy_state_dict(state_dict):

    upgraded = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for old, new in LEGACY_KEY_RENAMES.items():
            new_key = new_key.replace(old, new)
        upgraded[new_key] = value
    return upgraded
