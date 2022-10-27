from typing import TypedDict, Any


class Params(TypedDict):
    output_dir: str
    spm_file: str
    tsv_file: str
    rom_file_path: str
    batch_size: int
    lr: float
    gamma: float
    embedding_size: int
    hidden_size: int
    padding_idx: int
    gat_emb_size: int
    dropout_ratio: float
    preload_weights: str
    seed: int
    steps: int
    reset_steps: int
    stuck_steps: int
    trial: str
    loss: str
    graph_dropout: float
    k_object: int
    g_val: bool
    entropy_coeff: float
    clip: int
    bptt: int
    value_coeff: float
    template_coeff: float
    object_coeff: float
    recurrent: bool
    checkpoint_interval: int
    gat: bool
    masking: str
