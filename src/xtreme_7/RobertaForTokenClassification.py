import os

import torch
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers.modeling_outputs import TokenClassifierOutput
import sys
sys.path.append("/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/")
from model import StructTransformer

import logging
logger = logging.getLogger(__name__)

class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, args, config_class, model_class):
        super().__init__(config, args, config_class, model_class)
        self.num_labels = config.num_labels

        self.encoder = StructTransformer(args, config_class, model_class)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        #self.encoder_init(args)

    def encoder_init(self, config):
        output_dir = os.path.join(config.critic_output_dir, "checkpoint-best")
        logger.info("Training/evaluation parameters %s", config)
        self.encoder.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))

    def get_encoder(self, text, mask):
        return self.encoder.encoder(text, attention_mask=mask, struct_action_probs=None)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        action: Optional[torch.LongTensor] = None,
        get_embedding=False,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # self.encoder.eval()
        if get_embedding:
            return self.encoder.encoder(input_ids, attention_mask=attention_mask, struct_action_probs=None)
        outputs = self.encoder.encoder(
            input_ids,
            attention_mask=attention_mask,
            struct_action_probs=action,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if output_hidden_states:
            sequence_output = outputs['hidden_states'][15]
        else:
            sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
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