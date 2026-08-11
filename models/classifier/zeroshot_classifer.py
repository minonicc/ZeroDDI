import numpy as np
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn
import os
from torch.nn import CrossEntropyLoss, MSELoss, BatchNorm1d, Parameter
import torch.nn.init as init
from ..builder import build_left, build_right, CLASSIFIERS
from .candidate_matching import ReverseAttentionCandidateMatcher

import time
from collections import defaultdict
import math

    
@CLASSIFIERS.register_module()
class classifier(nn.Module):
    def __init__(self, leftmodel, rightmodel,temperature=0.9,
                 seen_labels=None, device=None,separability=False,UniformityLoss=True,
                 train_rightinput=None,val_zsl_rightinput=None,val_gzsl_rightinput=None, 
                 uni_lambda=0.7,use_w = False,
                instanse_uni_loss=True, class_uni_loss = True,
                 use_attention=True,
                 matching_mode='zeroddi',
                 matching_hidden_dim=256,
                 matching_dropout=0.1,
                 matching_use_null_evidence=True,
                 matching_use_substructure_evidence=True,
                 matching_use_evidence_gate=False,
                 matching_use_kg_evidence=False,
                 matching_kg_evidence_dim=300,
                 matching_kg_hidden_dim=None,
                 matching_kg_feature_vocab_sizes=None,
                 matching_use_pharmacophore_evidence=False,
                 matching_pharmacophore_evidence_dim=300,
                 matching_pharmacophore_hidden_dim=None,
                 matching_use_pharmacophore_gate=False,
                 matching_pharmacophore_top_k=None,
                 semantic_aux_lambda=0.0,
                 use_sign_cls = False,
                 attributlabel=None,
                zsl_labels=None,
        gzsl_labels=None):
        
        super(classifier, self).__init__()
        # this is for bert/gpt token attention
        self.Leftmodel = build_left(leftmodel)
        self.Rightmodel = build_right(rightmodel)
        self.temperature =temperature
        self.device = device
        self.seen_labels = seen_labels
        self.zsl_labels = zsl_labels
        self.gzsl_labels = gzsl_labels
        self.train_rightinput =train_rightinput 
        self.val_zsl_rightinput = val_zsl_rightinput
        self.val_gzsl_rightinput=val_gzsl_rightinput
        self.uni_lambda=uni_lambda
        self.use_w = use_w
        self.instanse_uni_loss=instanse_uni_loss
        self.class_uni_loss=class_uni_loss
        self.use_attention = use_attention
        self.matching_mode = matching_mode
        self.matching_use_kg_evidence = matching_use_kg_evidence
        self.matching_use_pharmacophore_evidence = matching_use_pharmacophore_evidence
        self.semantic_aux_lambda = semantic_aux_lambda
        self.use_sign_cls = use_sign_cls
        self.attributlabel = attributlabel

        if self.use_sign_cls:
            self.sign_w = Parameter(torch.Tensor(self.Rightmodel.output_dim, self.Rightmodel.output_dim))
            self.sign_cls =  nn.Sequential(
            nn.Linear(self.Rightmodel.output_dim, 1),
            nn.LeakyReLU())
            self.sign_loss = nn.CrossEntropyLoss()
       
        print("number of seen_labels:", len(self.seen_labels))
        if self.use_w:
            self.W = Parameter(torch.Tensor(256, 119))
            init.xavier_normal_(self.W)

        #global
        self.UniformityLoss=UniformityLoss
        self.separability=separability

        #local
        if self.use_attention:
            self.W_q = Parameter(torch.Tensor(300, 256))
            self.W_k = Parameter(torch.Tensor(self.Rightmodel.output_dim, self.Rightmodel.output_dim))
            self.W_v = Parameter(torch.Tensor(self.Rightmodel.output_dim, self.Rightmodel.output_dim))
            init.xavier_normal_(self.W_q)
            init.xavier_normal_(self.W_k)
            init.xavier_normal_(self.W_v)
        if self.matching_mode == 'reverse':
            self.ReverseMatcher = ReverseAttentionCandidateMatcher(
                pair_dim=256,
                evidence_dim=300,
                event_dim=self.Rightmodel.output_dim,
                hidden_dim=matching_hidden_dim,
                dropout=matching_dropout,
                use_null_evidence=matching_use_null_evidence,
                use_substructure_evidence=matching_use_substructure_evidence,
                use_evidence_gate=matching_use_evidence_gate,
                use_kg_evidence=matching_use_kg_evidence,
                kg_evidence_dim=matching_kg_evidence_dim,
                kg_hidden_dim=matching_kg_hidden_dim,
                kg_feature_vocab_sizes=matching_kg_feature_vocab_sizes,
                use_pharmacophore_evidence=matching_use_pharmacophore_evidence,
                pharmacophore_evidence_dim=matching_pharmacophore_evidence_dim,
                pharmacophore_hidden_dim=matching_pharmacophore_hidden_dim,
                use_pharmacophore_gate=matching_use_pharmacophore_gate,
                pharmacophore_top_k=matching_pharmacophore_top_k,
            )
        elif self.matching_mode != 'zeroddi':
            raise ValueError(f"Unsupported matching_mode: {self.matching_mode}")
        self.loss = nn.CrossEntropyLoss()
      
      
        
    
    def UniformityLoss_twoemb(self, drugembeddings, proto_embedding,device):
        
        center_proto_embedding = torch.mean(proto_embedding, dim=1)
        
        normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding.unsqueeze(1))
        #print("normalize_proto_embedding",normalize_proto_embedding.shape)#[128,107,256]
        cos_dist_matrix = torch.matmul(normalize_proto_embedding, normalize_proto_embedding.transpose(1, 2))
        unit_matrix = torch.eye(cos_dist_matrix.shape[1]).to(device)
        cos_dist_matrix = cos_dist_matrix - unit_matrix
        loss1 = torch.mean(torch.mean(torch.max(cos_dist_matrix, 2).values))
        
        
        
        n_d = F.normalize(drugembeddings - center_proto_embedding)#[256,256]
        #print("n_d",n_d.shape)
        d_cos_dist_matrix = torch.matmul(n_d, n_d.transpose(1, 0))
        d_unit_matrix = torch.eye(d_cos_dist_matrix.shape[0]).to(device)
        #print("d_unit_matrix",d_unit_matrix.shape)
        d_cos_dist_matrix = d_cos_dist_matrix - d_unit_matrix
        loss2 =torch.mean(torch.max(d_cos_dist_matrix, 1).values)
        
        if self.class_uni_loss ==True and self.instanse_uni_loss==False:
            return loss1
        elif self.class_uni_loss ==False and self.instanse_uni_loss==True:
            return loss2
        elif self.class_uni_loss ==True and self.instanse_uni_loss==True:
            return loss1+loss2

       
    def single_UniformityLoss_twoemb(self, drugembeddings, proto_embedding,device):
        
        
        center_proto_embedding = torch.mean(proto_embedding, dim=0)
        normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding.unsqueeze(0))
   
        cos_dist_matrix = torch.matmul(normalize_proto_embedding, normalize_proto_embedding.transpose(0, 1))
       
        unit_matrix = torch.eye(cos_dist_matrix.shape[0]).to(device)
        cos_dist_matrix = cos_dist_matrix - unit_matrix
        
        loss1 = torch.mean(torch.max(cos_dist_matrix, 1).values)
        
       

        n_d = F.normalize(drugembeddings - center_proto_embedding.unsqueeze(0))#[256,256]
      
        d_cos_dist_matrix = torch.matmul(n_d, n_d.transpose(1, 0))
        d_unit_matrix = torch.eye(d_cos_dist_matrix.shape[0]).to(device)
        #print("d_unit_matrix",d_unit_matrix.shape)
        d_cos_dist_matrix = d_cos_dist_matrix - d_unit_matrix
        loss2 =torch.mean(torch.max(d_cos_dist_matrix, 1).values)

        return loss1+loss2  
    


    def Local(self,left_output, drugemb, semanticemb,emb_ids, add_uniformity=True):
        if self.use_attention:
            d_k = drugemb.size(-1)
            Q = torch.matmul(drugemb, self.W_q).expand((semanticemb.shape[0], drugemb.shape[0],
                                                        drugemb.shape[1],256)).transpose(0, 1) # [batch,102,16, 256]
            K = torch.matmul(semanticemb, self.W_k).expand((drugemb.shape[0],semanticemb.shape[0],
                                                        semanticemb.shape[1],semanticemb.shape[2])) #[batch, 102, 35, 256]
            V = torch.matmul(semanticemb, self.W_v).expand((drugemb.shape[0],semanticemb.shape[0],
                                                        semanticemb.shape[1],semanticemb.shape[2])) ##[batch, 102, 35, 256]
            a = F.softmax(torch.matmul(Q,K.transpose(2, 3))/ math.sqrt(d_k),dim=-1) # [batch, 102, 16, 35]
            drug_atten = torch.matmul(a,V)# [batch, class, sub, dim]
            drug_atten_ = torch.mean(drug_atten,dim=2)# [batch, class, dim]
            if self.use_w: 
                print("use w")
                left_output = torch.matmul(left_output,self.W) 
            logits = torch.matmul(left_output.unsqueeze(dim=1),drug_atten_.transpose(1, 2)).squeeze(1)/ self.temperature
            loss = self.loss(logits,torch.tensor(emb_ids, dtype=torch.long).to(self.device))
            if self.UniformityLoss and add_uniformity:
                loss_uni= self.UniformityLoss_twoemb(left_output,drug_atten_, self.device)
                loss = loss+self.uni_lambda*loss_uni
        else:
            if len(semanticemb.shape)==3:
                drug_atten_ = torch.mean(semanticemb,1)
            else:
                drug_atten_ = semanticemb
                
            if self.use_w: 
                #print("use w")
                left_output = torch.matmul(left_output,self.W)
                #print("llllllleft_output",left_output.shape)
            #print("drug_atten_",drug_atten_.shape)#[107,256]
            #print("left_output",left_output.shape)#[batch,256]
            a = 0 
           
            logits = torch.matmul(left_output,drug_atten_.transpose(0, 1))/ self.temperature 
        
            loss = self.loss(logits,torch.tensor(emb_ids, dtype=torch.long).to(self.device))
            if self.UniformityLoss and add_uniformity:
                loss_uni= self.single_UniformityLoss_twoemb(left_output,drug_atten_, self.device)
                loss = loss+self.uni_lambda*loss_uni
        
        return logits,loss,a,drug_atten_

    def ReverseLocal(
        self,
        left_output,
        drugemb,
        semanticemb,
        emb_ids,
        kg_evidence=None,
        pharmacophore_evidence=None,
    ):
        kg_evidence_tokens = None
        kg_evidence_mask = None
        pharmacophore_evidence_tokens = None
        pharmacophore_evidence_mask = None
        if kg_evidence is not None:
            if isinstance(kg_evidence, dict):
                kg_evidence_tokens = kg_evidence.get("tokens")
                kg_evidence_mask = kg_evidence.get("mask")
            elif isinstance(kg_evidence, (list, tuple)):
                kg_evidence_tokens = kg_evidence[0]
                if len(kg_evidence) > 1:
                    kg_evidence_mask = kg_evidence[1]
            else:
                kg_evidence_tokens = kg_evidence
        if kg_evidence_tokens is not None:
            kg_evidence_tokens = kg_evidence_tokens.to(self.device)
        if kg_evidence_mask is not None:
            kg_evidence_mask = kg_evidence_mask.to(self.device)
        if pharmacophore_evidence is not None:
            if isinstance(pharmacophore_evidence, dict):
                pharmacophore_evidence_tokens = pharmacophore_evidence.get("tokens")
                pharmacophore_evidence_mask = pharmacophore_evidence.get("mask")
            elif isinstance(pharmacophore_evidence, (list, tuple)):
                pharmacophore_evidence_tokens = pharmacophore_evidence[0]
                if len(pharmacophore_evidence) > 1:
                    pharmacophore_evidence_mask = pharmacophore_evidence[1]
            else:
                pharmacophore_evidence_tokens = pharmacophore_evidence
        if pharmacophore_evidence_tokens is not None:
            pharmacophore_evidence_tokens = pharmacophore_evidence_tokens.to(self.device)
        if pharmacophore_evidence_mask is not None:
            pharmacophore_evidence_mask = pharmacophore_evidence_mask.to(self.device)
        labels = torch.as_tensor(emb_ids, dtype=torch.long, device=self.device)
        outputs = self.ReverseMatcher(
            pair_repr=left_output,
            evidence_tokens=drugemb,
            event_tokens=semanticemb,
            labels=labels,
            kg_evidence_tokens=kg_evidence_tokens,
            kg_evidence_mask=kg_evidence_mask,
            pharmacophore_evidence_tokens=pharmacophore_evidence_tokens,
            pharmacophore_evidence_mask=pharmacophore_evidence_mask,
        )
        evidence_diagnostics = {
            "substructure_attention": outputs["attention"],
            "kg_attention": outputs["kg_attention"],
            "pharmacophore_attention": outputs["pharmacophore_attention"],
            "pharmacophore_gate": outputs["pharmacophore_gate"],
            "pharmacophore_selection_indices": outputs[
                "pharmacophore_selection_indices"
            ],
            "pharmacophore_selection_mask": outputs[
                "pharmacophore_selection_mask"
            ],
        }
        return (
            outputs["logits"],
            outputs["loss"],
            outputs["attention"],
            outputs["selected_evidence"],
            evidence_diagnostics,
        )

    def forward(self, input):
        """
        input: self.new_current_dataset[index], self.mode,self.zsl_mode,sign,effect,pattern
        """
        #print("aaa",input[0])
        left_output , sub_structure, d1_att, d2_att = self.Leftmodel(input[0])  # ,[batch, 256]
        #print("left_output",left_output.shape)
        #right_output_all, emb_ids = self.Rightmodel(input)
        ############
        if input[2][0]=="train":
            right_output_all, emb_ids, bertemb = self.Rightmodel(input[0],self.train_rightinput)  # [class_num, 35, 256 ]
            if self.use_sign_cls:
                train_rightinput = torch.matmul(bertemb,self.sign_w)
                train_rightinput = self.sign_cls(train_rightinput)
                sign_loss = self.sign_loss(train_rightinput.squeeze(),torch.tensor(np.array(self.attributlabel[0]), dtype=torch.long).to(self.device))

        elif input[2][0]=="zsl":
            right_output_all, emb_ids,_ = self.Rightmodel(input[0],self.val_zsl_rightinput)  # [class_num, 35, 256 ]
        elif input[2][0] == "gzsl":
            right_output_all, emb_ids,_ = self.Rightmodel(input[0],self.val_gzsl_rightinput)  # [class_num, 35, 256 ]
        ###############
        
        kg_evidence = None
        if self.matching_use_kg_evidence and len(input[0]) > 3:
            kg_evidence = input[0][3]
        pharmacophore_evidence = None
        if self.matching_use_pharmacophore_evidence and isinstance(d2_att, dict):
            pharmacophore_evidence = d2_att.get("pharmacophore")

        #print("right_output_all",right_output_all.shape)
        if self.matching_mode == 'reverse':
            logits, loss_g, cross_att, proto, evidence_diagnostics = self.ReverseLocal(
                left_output,
                sub_structure,
                right_output_all,
                emb_ids,
                kg_evidence=kg_evidence,
                pharmacophore_evidence=pharmacophore_evidence,
            )
            if self.semantic_aux_lambda > 0:
                _, semantic_aux_loss, _, _ = self.Local(
                    left_output,
                    sub_structure,
                    right_output_all,
                    emb_ids,
                    add_uniformity=False,
                )
                loss_g = loss_g + self.semantic_aux_lambda * semantic_aux_loss
        else:
            logits,loss_g,cross_att, proto = self.Local(left_output, sub_structure, right_output_all, emb_ids)
            evidence_diagnostics = None
        if input[2][0]=="train" and self.use_sign_cls:
            loss_g = loss_g+sign_loss
            
        if len(right_output_all.shape)==3:
            eventemb_mean = torch.mean(right_output_all,1) 
        else:
            eventemb_mean=right_output_all
        #print("eventemb_mean",eventemb_mean.shape)
        prototype = eventemb_mean[emb_ids, :]

        return (
            loss_g,
            logits,
            emb_ids,
            left_output,
            prototype,
            cross_att,
            d1_att,
            d2_att,
            evidence_diagnostics,
        )

    

