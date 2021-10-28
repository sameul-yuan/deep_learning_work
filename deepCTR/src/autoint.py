
import copy
import os
import sys
import itertools
import yaml
import logging
import warnings
import json 
import copy
import math 
from typing import List,Tuple,Dict

import torch
import torch.utils.checkpoint
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss,BCEWithLogitsLoss
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

from torch import Tensor, device, dtype



class PretrainedConfig(object):
    r""" Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.

        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.

        Class attributes (overridden by derived classes):
            - ``model_type``: a string that identifies the model type, that we serialize into the JSON file, and that we use to recreate the correct object in :class:`~transformers.AutoConfig`.

        Args:
            finetuning_task (:obj:`string` or :obj:`None`, `optional`, defaults to :obj:`None`):
                Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            num_labels (:obj:`int`, `optional`, defaults to `2`):
                Number of classes to use when the model is a classification model (sequences/tokens)
            output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Should the model returns all hidden-states.
            output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Should the model returns all attentions.
            torchscript (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Is the model used with Torchscript (for PyTorch models).
    """
    model_type: str = ""

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_cache = kwargs.pop("use_cache", True)  # Not used by all models
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        # self.do_sample = kwargs.pop("do_sample", False)
        # self.early_stopping = kwargs.pop("early_stopping", False)
        # self.num_beams = kwargs.pop("num_beams", 1)
        # self.temperature = kwargs.pop("temperature", 1.0)
        # self.top_k = kwargs.pop("top_k", 50)
        # self.top_p = kwargs.pop("top_p", 1.0)
        # self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        # self.length_penalty = kwargs.pop("length_penalty", 1.0)
        # self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        # self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        # self.num_return_sequences = kwargs.pop("num_return_sequences", 1)

        # Fine-tuning task arguments
        # self.architectures = kwargs.pop("architectures", None)
        # self.finetuning_task = kwargs.pop("finetuning_task", None)
        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.prefix = kwargs.pop("prefix", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs):
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logging.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: str):
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: An instance of a configuration object

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_diff_dict(self):
        """
        Removes all attributes from config which correspond to the default
        config attributes for better readability and serializes to a Python
        dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if key not in default_config_dict or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self, use_diff=True):
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path, use_diff=True):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`):
                If set to True, only the difference between the config instance and the default PretrainedConfig() is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict):
        """
        Updates attributes of this class
        with attributes from `config_dict`.

        Args:
            :obj:`Dict[str, any]`: Dictionary of attributes that shall be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

class BertAttention(nn.Module):
    """
    multli-head attention 
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.input_project = nn.Linear(config.hidden_size, self.all_head_size,bias=False)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  #(bsz,seq, heads,head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  #(bsz,heads,seq,hz)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)  # (bsz,seq,h)
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) #  #(bsz, heads, seq, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #batch-wise matrix multiply  (bsz, heads, seq, seq)  
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # to avoid saturation 
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask  # (-1e9)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  #(bsz,heads,seq,seq)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) #(bsz,heads,seq,head_size)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  #(bsz,seq,heads,head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #(bsz,seq, hidden_size)
        context_layer = context_layer.view(*new_context_layer_shape)

        res_input = self.input_project(hidden_states)
        # res_input = hidden_states
        context_layer = context_layer+res_input
        
        # ouput
        context_layer = nn.functional.relu(context_layer)
        context_layer = self.layer_norm(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        # self.intermediate = BertIntermediate(config)
        # self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,   #(bsz,seq)
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,   #(bsz,seq)
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]   # (bsz, seq, hidden_size)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # intermediate_output = self.intermediate(attention_output)
        # layer_output = self.output(intermediate_output, attention_output)
        # outputs = (layer_output,) + outputs  # outputs, self_atten_out, cross_atten_out

        outputs = (attention_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config
        

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights if needed
        self.tie_weights()

    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of whether we are using TorchScript or not
        """
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self):
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`:
                A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings

        Args:
            value (:obj:`nn.Module`):
                A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        """
        Returns the model's output embeddings.

        Returns:
            :obj:`nn.Module`:
                A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings
    
    @property
    def device(self):
        """
        Get torch.device from module, assuming that the whole module has one device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @property
    def dtype(self):
        """
        Get torch.dtype from module, assuming that the whole module has one dtype.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def invert_attention_mask(self, encoder_attention_mask: Tensor):
        """type: torch.Tensor -> torch.Tensor"""
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                "{} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`".format(
                    self.dtype
                )
            )

        return encoder_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple, device: device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2: 
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class CustomBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config   # PretrainedConfig
        # self.embeddings = CustomBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,  # for encoder self-attention or decoder self-attention
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None, # for decoder cross attention 
        encoder_attention_mask=None, #used for decoder cross attention
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:          
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  #(batch, seq)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads. [batch_size, num_heads, seq_length, seq_length]
        # pad mask for encoder self, pad & causl mask for decoder self-attention and pad mask for decoder cross_attention
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:   
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = [None]* self.config.num_hidden_layers

        # embedding_output = self.embeddings(
        #     input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        # )
        embedding_output = inputs_embeds
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)
        pooled_output = None
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The totale number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class AutoInt(nn.Module):
    def __init__(self, pretrained_config, params_config, field_size=None, feature_size=None):

        super().__init__()
        assert isinstance(pretrained_config, PretrainedConfig)
        assert pretrained_config.hidden_size == params_config['embedding_size'], 'embedding_size not equals'

        self.pretrained_config = pretrained_config
        self.field_size = field_size
        self.feature_size = feature_size
        self.emb_size = params_config.pop('embedding_size')
        self.param_cfg = params_config
 
    
        self.encoder = CustomBertModel(self.pretrained_config)


        self.high_order_embedding = torch.nn.Embedding(self.feature_size, self.emb_size) # F*K
        self.high_order_project = torch.nn.Linear(self.emb_size*self.field_size,1)
        # self.high_order_project = torch.nn.Linear(self.emb_size*self.field_size+self.field_size+self.emb_size,1)  #fully connect 


        #--init---
        self.high_order_embedding.apply(self.__init_weight)
        self.high_order_project.weight.data.normal_(std=self.param_cfg['init_norm_var'])
        self.high_order_project.bias.data.fill_(0.0)

    def __init_weight(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(0, self.param_cfg['init_norm_var']) #  nn.init.normal_(module.weight.data,0,0.02),module.weight.data.fill_(0)/zero_()
            
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            module.bias.data.normal_(std=self.param_cfg['init_norm_var'])
        


    def forward(self, xi,xv):
        """
        xi: (bsz,F)
        xv: (bsz,F)

        """
        # size = x.shape[1]
        # xi = x[:,:size//2]
        # xv = x[:,size//2:]
        # second_part
        embs = self.high_order_embedding(xi.long())  # (bsz,F,K)
        # print(embs.dtype, xv.dtype)
        embs = torch.mul(embs, xv.unsqueeze(-1).float())
    
        # self-attention
        encoder_outputs = self.encoder(inputs_embeds=embs, output_hidden_states=None, output_attentions=None)  #
        encoder_seq_outputs, pooled_output = encoder_outputs[0], encoder_outputs[1]  #(batch,F,emb_dim), (batch, emb_dim)
        
        dnn_input = encoder_seq_outputs.view(-1, self.field_size*self.emb_size)
        dnn_out = self.high_order_project(dnn_input)

        return dnn_out



if __name__ =='__main__':
    import pandas as pd 
    import json
    import yaml
    from torch.utils.data import SequentialSampler, DataLoader
    from dataset import DeepFmDataset
    from self_attention.config import PretrainedConfig
    # from torchviz import make_dot

    data_path = '../data/reno4_lookalike_autodis_log.pkl'
    feature_cfg  = json.load(open('../server/reno4_lookalike_autodis_log.dict','r'))

    all_cfg = yaml.load(open('../config.yaml','r'))
    param_cfg = all_cfg['FM']
    pretrained_cfg  = PretrainedConfig(**all_cfg['self_attention'])
    pretrained_cfg.initializer_range = param_cfg['init_norm_var']
    pretrained_cfg.hidden_size = param_cfg['embedding_size']


    df = pd.read_pickle(data_path)
    df = df.sample(frac=0.1, replace=False)
    drop_cols = ['label']
    df = df.drop(columns=drop_cols)
    df = df.rename(columns={'label_2m':'label'})
    print(df.shape)
    

    dataset = DeepFmDataset(df,feature_cfg)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=512)

    model = AutoInt(pretrained_cfg, param_cfg,field_size=feature_cfg['field_dim'],feature_size=feature_cfg['feature_dim'])
    model_path  = '/home/notebook/code/personal/smart_rec/deepFM/server/model/20210202_reno4_lookalike_autodis_log2_0.8002/checkpoint_last/model.dict'
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))  #map_location=torch.device('cpu')
    model.eval()

    for idx, batch in enumerate(dataloader):
        xi,xv,label=batch
        model.eval()
        logits = model(xi,xv)
        if idx==0:
            print(logits.shape)
            break


    # g = make_dot(logits, params=dict(list(model.named_parameters()) + [('xi', xi),('xv',xv)]))
    # g.render('../autoindo.pdf', view=False) 

