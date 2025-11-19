# gemma3_biencoder.py
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextModel,
)

class Gemma3EncoderForMaskedLM(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig
    base_model_prefix = "encoder"  # must match attribute name below
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig):
        # work on a private copy to avoid mutating the caller's config in-place
        cfg = copy.deepcopy(config)

        # 1) Make text attention bidirectional (official switch in HF)
        #    This disables the triangular causal mask for TEXT tokens.
        #    (Vision-token behavior is unchanged.)
        if hasattr(cfg, "use_bidirectional_attention"):
            cfg.use_bidirectional_attention = True  # <- key change
        # 2) Encoders don't use KV cache
        cfg.use_cache = False

        super().__init__(cfg)
        self.encoder = Gemma3TextModel(cfg)
        self.vocab_size = cfg.vocab_size
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, S] (1=keep, 0=pad)
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:

        # Run the text backbone in non-causal mode and without cache.
        # `is_causal` is passed defensively; the config flag is what actually
        # disables the triangular mask in Gemma3.
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,      # 2D pad mask
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            is_causal=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            out = (logits, hidden_states)
            if output_hidden_states:
                out += (outputs.hidden_states,)
            if output_attentions:
                out += (outputs.attentions,)
            if loss is not None:
                out = (loss,) + out
            return out

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )