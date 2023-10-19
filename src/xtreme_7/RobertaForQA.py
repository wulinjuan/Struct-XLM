import os

import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, RobertaConfig
import torch.nn as nn
import sys
sys.path.append("/data/home10b/wlj/code/Struct-XLM/src/Struct_XLM/")
from model import StructTransformer

import logging

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaForQuestionAnswering(BertPreTrainedModel):
    r"""
      **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
          Labels for position (index) of the start of the labelled span for computing the token classification loss.
          Positions are clamped to the length of the sequence (`sequence_length`).
          Position outside of the sequence are not taken into account for computing the loss.
      **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
          Labels for position (index) of the end of the labelled span for computing the token classification loss.
          Positions are clamped to the length of the sequence (`sequence_length`).
          Position outside of the sequence are not taken into account for computing the loss.
  Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
      **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
          Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
      **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
          Span-start scores (before SoftMax).
      **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
          Span-end scores (before SoftMax).
      **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
          list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
          of shape ``(batch_size, sequence_length, hidden_size)``:
          Hidden-states of the model at the output of each layer plus the initial embedding outputs.
      **attentions**: (`optional`, returned when ``config.output_attentions=True``)
          list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
          Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
  Examples::
      tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
      model = RobertaForQuestionAnswering.from_pretrained('roberta-large')
      question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
      input_ids = tokenizer.encode(question, text)
      start_scores, end_scores = model(torch.tensor([input_ids]))
      all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
      answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
  """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, args, config_class, model_class):
        super(RobertaForQuestionAnswering, self).__init__(config, args, config_class, model_class)
        self.num_labels = config.num_labels

        self.encoder = StructTransformer(args, config_class, model_class)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.args = args
        #self.encoder_init()

    def encoder_init(self):
        output_dir = os.path.join(self.args.critic_output_dir, "checkpoint-best")
        logger.info("Training/evaluation parameters %s", self.args)
        self.encoder.load_state_dict(torch.load(os.path.join(output_dir, "target_model_critic.bin")))

    def get_encoder(self, text, mask):
        return self.encoder.encoder(text, attention_mask=mask, struct_action_probs=None)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            start_positions=None,
            end_positions=None,
            action=None,
            get_embedding=False,
    ):
        if get_embedding:
            return self.encoder.encoder(input_ids, attention_mask=attention_mask, struct_action_probs=None)
        outputs = self.encoder.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            struct_action_probs=action,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
