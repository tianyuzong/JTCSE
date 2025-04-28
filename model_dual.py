import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class teacher_model(nn.Module):
    def __init__(self, pretrained_model_1, pretrained_model_2):
        super(teacher_model, self).__init__()
        self.teacher_model_1 = AutoModel.from_pretrained(pretrained_model_1)
        self.teacher_model_2 = AutoModel.from_pretrained(pretrained_model_2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out_1 = self.teacher_model_1(input_ids, attention_mask, token_type_ids, output_hidden_states=True,
                                     return_dict=True)
        out_2 = self.teacher_model_2(input_ids, attention_mask, token_type_ids, output_hidden_states=True,
                                     return_dict=True)
        return out_1.last_hidden_state[:, 0] + out_2.last_hidden_state[:, 0]


def unsup_infonce_loss(y_pred, device, temp=0.05):
    # label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1

    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # print(sim.shape,torch.eye(y_pred.shape[0]).shape)
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12

    sim = sim / temp

    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def icnce(y_pred_1, y_pred_2, device, temp=0.05):
    # print(y_pred_1.shape, y_pred_2.shape)
    y_true = torch.arange(y_pred_1.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1

    sim = F.cosine_similarity(y_pred_1.unsqueeze(1), y_pred_2.unsqueeze(0), dim=-1)
    # print(sim.shape, torch.eye(y_pred_1.shape[0]).shape)
    sim = sim - torch.eye(y_pred_1.shape[0], device=device) * 1e12

    sim = sim / temp

    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def LTN_loss(y_pred_L, y_pred_a_L, y_pred_P, y_pred_a_P, device="cuda"):
    y_true = torch.arange(y_pred_L.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1

    sim = F.cosine_similarity(y_pred_L.unsqueeze(1), y_pred_a_L.unsqueeze(0), dim=-1)
    loss_temp = 0.0
    for i in range(y_pred_L.shape[0]):
        if i % 2 == 0:
            sim_temp = sim[i][y_true[i]]
            if sim_temp < 0.0:
                sim_temp = torch.tensor(0.0001, device=device)
            sim_temp = -1.0 * torch.log(sim_temp)
            M_long = torch.norm(y_pred_P[i] - y_pred_a_P[i + 1], p=2) / (
                    torch.norm(y_pred_P[i], p=2) + torch.norm(y_pred_a_P[i + 1], p=2)) / 2.0
            loss_temp += sim_temp * M_long
    loss_temp = loss_temp / (y_pred_L.shape[0] / 2)
    return loss_temp
