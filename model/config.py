from transformers.configuration_utils import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "qwen3"

    def __init__(
        self,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        head_dim:int = 128,
        hidden_act: str = "silu",
        hidden_size: int = 1024,
        initializer_range: float = 0.02,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 40960,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 28,
        num_key_value_heads: int = 8,
        vocab_size: int = 151936,
        rms_norm_eps: float = 1e-06,
        rope_theta: int = 1000000,
        flash_attn: bool = True,
        torch_dtype: str = "bfloat16",
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = attention_dropout
        self.attention_bias = attention_bias
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        self.torch_dtype = torch_dtype
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

