import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import optuna
import csv
import os
import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttention

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)#(768,768)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp): 
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type): 
        super().__init__()
        self.pooler_type = pooler_type 
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state 
        pooler_output = outputs.pooler_output #None
        hidden_states = outputs.hidden_states #None

        if self.pooler_type in ['cls_before_pooler', 'cls']: 
            return last_hidden[:, 0] 
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type #cls
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls": #True
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def analyze_tensor(tensor,batch_size)
    max_values, max_indices = torch.max(tensor, dim=1)
    sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
    diagonal_elements = torch.diagonal(tensor)
    
    difference = torch.zeros(batch_size)
    difference2 = torch.zeros(batch_size)  

    for i in range(batch_size):
        positive_example = diagonal_elements[i]
        negative_examples = tensor[i]
        negative_examples = torch.cat((negative_examples[:i], negative_examples[i+1:]))
        difference[i] = positive_example - torch.mean(negative_examples)
        
        if max_values[i] == positive_example:  
            difference2[i] = positive_example - sorted_tensor[i][1]  
        else:  
            difference2[i] = positive_example - max_values[i]

    diagonal_rank = torch.zeros(batch_size)

    for i in range(batch_size):
        row_sorted, _ = torch.sort(tensor[i], descending=True)
        nonzero_indices = (row_sorted == diagonal_elements[i]).nonzero(as_tuple=True)[0].tolist()


        if len(nonzero_indices) != 0:
            diagonal_rank[i] = nonzero_indices[0] + 1
        else:
            print('There was a problem when checking the ranking of diagonal values returned by "cos_sim"')
            pass
    return max_values, max_indices, difference, difference2, diagonal_rank


def find_max_except_diagonal(cos_sim_bak):
    cos_sim_bak = cos_sim_bak.cuda()
    diagonal_mask = torch.eye(cos_sim_bak.size(0), device='cuda').bool()
    cos_sim_bak = cos_sim_bak.masked_fill(diagonal_mask, float('-inf'))
    row_max_values, row_max_indices = torch.max(cos_sim_bak, dim=1)
    return row_max_values, row_max_indices

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict #True
    ori_input_ids = input_ids #input_ids.shape:[128,2,32]
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len) --> [256,32]
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent,len) --->[256,32]
    if token_type_ids is not None: #true
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)--->[256,32]

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids, #None
        head_mask=head_mask, #None
        inputs_embeds=inputs_embeds, #None
        output_attentions=output_attentions, #None
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False, #False
        return_dict=True,
    )


    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs) #pooler_output.shape:[256,768]
    # (bs, num_sent, hidden) (128,2,768)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) #shape:[128,2,768]

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls": #cls
        pooler_output = cls.mlp(pooler_output) #shape:[128,2,768]


    # Separate representation 
    z1, z2 = pooler_output[:,0], pooler_output[:,1]
    #z3 = torch.zeros(z2.shape[0], z2.shape[1]).to(cls.device)
    z3 = torch.zeros(z2.shape[0], z2.shape[1]).to(cls.device)

    cos_sim_bak2 = cos_sim_bak = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) 
    
    row_max_values, row_max_indices=find_max_except_diagonal(cos_sim_bak2)
    
    def objective(trial):
        weight1 = trial.suggest_float("weight1", 0, 2)
        weight2 = trial.suggest_float("weight2", 0, 2)

        for i in range(z2.shape[0]):
            idx = row_max_indices[i]
            if cos_sim_bak[i, i] > row_max_values[i]: 
                if (cos_sim_bak[i, i] - row_max_values[i]) / cos_sim_bak[i, i] <= 0.1:
                    z3[i] = weight1 * z2[i] + weight2 * z2[idx]      
                else:  
                    z3[i] = z2[i]
            else:      
                z3[i] = cos_sim_bak[i, i] / (cos_sim_bak[i, i] + row_max_values[i]) * z2[i] + row_max_values[i] / (cos_sim_bak[i, i] + row_max_values[i]) * z2[idx]

        cos_sim = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        max_values, max_indices, difference, difference2, diagonal_rank = analyze_tensor(cos_sim,batch_size)
       
        return difference.mean().item() + diagonal_rank.mean().item()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2000)  

    best_weight1 = study.best_params["weight1"]
    best_weight2 = study.best_params["weight2"]      

    cos_sim = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0)) 

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device) 
 
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)
 
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm: #False
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        #一个eval_steps==125之后，sent_emb==True
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
