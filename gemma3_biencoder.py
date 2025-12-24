# gemma3_biencoder.py
from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextModel,
)

class Gemma3EncoderForMaskedLM(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig
    base_model_prefix = "encoder"
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, config: Gemma3TextConfig):
        cfg = copy.deepcopy(config)
        if hasattr(cfg, "use_bidirectional_attention"):
            cfg.use_bidirectional_attention = True
        cfg.use_cache = False
        super().__init__(cfg)

        self.encoder = Gemma3TextModel(cfg)
        self.vocab_size = cfg.vocab_size
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.post_init()  # calls tie_weights()

    # Embeddings / head
    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_head: nn.Module):
        self.lm_head = new_head

    # Keep vocab_size in sync; ensure pointer-tying
    def tie_weights(self):
        if hasattr(self.config, "vocab_size"):
            self.config.vocab_size = self.get_input_embeddings().num_embeddings
            self.vocab_size = self.config.vocab_size
        if getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())

    # Ensure 'lm_head.weight' exists when saving (avoids resume warnings)
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if "lm_head.weight" not in sd and getattr(self.config, "tie_word_embeddings", True):
            emb_key = f"{self.base_model_prefix}.embed_tokens.weight"
            if emb_key in sd:
                sd["lm_head.weight"] = sd[emb_key]
        return sd

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
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


class Gemma3EncoderForSequenceClassification(Gemma3PreTrainedModel):
    """Gemma3 Encoder with a sequence classification head (mean pooling + linear)."""
    config_class = Gemma3TextConfig
    base_model_prefix = "encoder"

    def __init__(self, config: Gemma3TextConfig):
        cfg = copy.deepcopy(config)
        if hasattr(cfg, "use_bidirectional_attention"):
            cfg.use_bidirectional_attention = True
        cfg.use_cache = False
        super().__init__(cfg)

        self.num_labels = getattr(cfg, "num_labels", 2)
        self.encoder = Gemma3TextModel(cfg)
        
        classifier_dropout = getattr(cfg, "classifier_dropout", 0.0)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(cfg.hidden_size, self.num_labels)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            is_causal=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Gemma3EncoderForTokenClassification(Gemma3PreTrainedModel):
    """Gemma3 Encoder with a token classification head for NER/POS tagging."""
    config_class = Gemma3TextConfig
    base_model_prefix = "encoder"

    def __init__(self, config: Gemma3TextConfig):
        cfg = copy.deepcopy(config)
        if hasattr(cfg, "use_bidirectional_attention"):
            cfg.use_bidirectional_attention = True
        cfg.use_cache = False
        super().__init__(cfg)

        self.num_labels = getattr(cfg, "num_labels", 2)
        self.encoder = Gemma3TextModel(cfg)
        
        classifier_dropout = getattr(cfg, "classifier_dropout", 0.0)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(cfg.hidden_size, self.num_labels)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            is_causal=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )