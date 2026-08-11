# -*- coding: utf-8 -*-
import os
import json
import sqlite3
import zlib
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
from .builder import DATASETS, build_dataset
from transformers import AutoTokenizer, BioGptModel,AutoModel 
from .Mesh_similarity import MeshText

    
@DATASETS.register_module()
class AttriTextBioBERTDataset(Dataset):

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_kg_sqlite_connection"] = None
        return state

    def __init__(self,
                 Allfilename,
                 mode,
                 file_dir,
                 file_name,
                 zsl_mode,
                 output_file=None,
                 output_dim=None,
                 #input_dim = None,
                 device='cuda:0',
                 bert_vision="biobert-base-cased-v1.2",
                 kg_pair_file=None,
                 kg_max_tokens=128,
                 kg_max_nodes=None,
                 kg_max_edges=None,
                 ):
      
        self.Allfilename = Allfilename
        self.mode = mode
        self.file_dir = file_dir
        self.file_name = file_name
        self.output_file = output_file
        self.device = device
        self.output_dim = output_dim
        self.zsl_mode = zsl_mode
        self.kg_pair_file = kg_pair_file
        self.kg_max_tokens = kg_max_tokens
        self.kg_max_nodes = kg_max_nodes or kg_max_tokens
        self.kg_max_edges = kg_max_edges or (self.kg_max_nodes * 4)
        self.kg_pair_tokens, self.kg_feature_vocab_sizes = self._load_kg_pair_file()

        self.bert_vision = bert_vision
        self.mesh_text_biobertemb, self.biobertemb = self._get_all_embeddings()
        
        self.current_all_biogpt_emb, self.current_all_mesh_emb = self.get_current_dataset()
        self.rightinput = (self.current_all_biogpt_emb, self.current_all_mesh_emb)
        self.rightattributelabel = (self.current_sign_id, self.current_mesh_id, self.current_patt_id)
        self.dim = self.output_dim
        self.input_dim = (self.current_all_biogpt_emb.shape[2], self.current_all_mesh_emb.shape[2])

    def _load_kg_pair_file(self):
        if self.kg_pair_file is None:
            return None, None
        if not os.path.exists(self.kg_pair_file):
            raise FileNotFoundError(f"KG pair file not found: {self.kg_pair_file}")

        self._kg_sqlite_connection = None
        if self.kg_pair_file.endswith(".sqlite"):
            connection = sqlite3.connect(self.kg_pair_file)
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
            connection.close()
            data = {key: json.loads(value) for key, value in rows}
            pair_data = True
        else:
            with open(self.kg_pair_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            pair_data = data.get("pairs", {})
        self.kg_format = data.get("format", "flat_tokens_v1")
        self.kg_distance_vocab_size = data.get("distance_vocab_size")
        vocab_sizes = data.get("feature_vocab_sizes")
        if vocab_sizes is not None and self.kg_distance_vocab_size is not None:
            vocab_sizes = dict(vocab_sizes)
            vocab_sizes["distance"] = self.kg_distance_vocab_size
        return pair_data, vocab_sizes

    def _lookup_kg_pair(self, key):
        if self.kg_pair_file.endswith(".sqlite"):
            if self._kg_sqlite_connection is None:
                self._kg_sqlite_connection = sqlite3.connect(self.kg_pair_file)
            row = self._kg_sqlite_connection.execute(
                "SELECT graph FROM pairs WHERE pair_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(zlib.decompress(row[0]).decode("utf-8"))
        return self.kg_pair_tokens.get(key)

    def _pair_key(self, drug1, drug2):
        return f"{drug1}||{drug2}"

    def _get_kg_evidence(self, drug1, drug2):
        if getattr(self, "kg_format", "flat_tokens_v1") == "molecbionet_pair_graph_v2":
            return self._get_kg_graph(drug1, drug2)
        tokens = []
        if self.kg_pair_tokens is not None:
            tokens = self._lookup_kg_pair(self._pair_key(drug1, drug2))
            if tokens is None:
                tokens = self._lookup_kg_pair(self._pair_key(drug2, drug1)) or []
        tokens = tokens[: self.kg_max_tokens]
        mask = [True] * len(tokens)

        pad_len = self.kg_max_tokens - len(tokens)
        if pad_len > 0:
            tokens = tokens + [[0, 0, 0, 0, 0] for _ in range(pad_len)]
            mask = mask + [False] * pad_len

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }

    def _get_kg_graph(self, drug1, drug2):
        graph = self._lookup_kg_pair(self._pair_key(drug1, drug2))
        reversed_pair = graph is None
        if reversed_pair:
            graph = self._lookup_kg_pair(self._pair_key(drug2, drug1))
        if graph is None:
            graph = {
                "node_ids": [], "node_types": [], "distance_to_a": [],
                "distance_to_b": [], "edge_index": [[], []], "edge_relations": [],
            }

        node_ids = list(graph["node_ids"][:self.kg_max_nodes])
        node_types = list(graph["node_types"][:self.kg_max_nodes])
        distance_a = list(graph["distance_to_a"][:self.kg_max_nodes])
        distance_b = list(graph["distance_to_b"][:self.kg_max_nodes])
        if reversed_pair:
            distance_a, distance_b = distance_b, distance_a

        node_count = len(node_ids)
        sources = []
        targets = []
        relations = []
        for source, target, relation in zip(
            graph["edge_index"][0], graph["edge_index"][1], graph["edge_relations"]
        ):
            if source < node_count and target < node_count:
                sources.append(source)
                targets.append(target)
                relations.append(relation)
                if len(relations) >= self.kg_max_edges:
                    break

        node_mask = [True] * node_count
        node_pad = self.kg_max_nodes - node_count
        node_ids.extend([0] * node_pad)
        node_types.extend([0] * node_pad)
        distance_a.extend([0] * node_pad)
        distance_b.extend([0] * node_pad)
        node_mask.extend([False] * node_pad)

        edge_count = len(relations)
        edge_pad = self.kg_max_edges - edge_count
        sources.extend([0] * edge_pad)
        targets.extend([0] * edge_pad)
        relations.extend([0] * edge_pad)
        edge_mask = [True] * edge_count + [False] * edge_pad
        return {
            "node_ids": torch.tensor(node_ids, dtype=torch.long),
            "node_types": torch.tensor(node_types, dtype=torch.long),
            "distance_to_a": torch.tensor(distance_a, dtype=torch.long),
            "distance_to_b": torch.tensor(distance_b, dtype=torch.long),
            "node_mask": torch.tensor(node_mask, dtype=torch.bool),
            "edge_index": torch.tensor([sources, targets], dtype=torch.long),
            "edge_relations": torch.tensor(relations, dtype=torch.long),
            "edge_mask": torch.tensor(edge_mask, dtype=torch.bool),
        }

    def _get_all_embeddings(self):
        # id,drug1,drug2,event_id,MeSH_ID,Sign,Pattern,description,smiles1,smiles2
     
        df_all_dataset = pd.read_csv(self.Allfilename)
        all_dataset = [[id1, id2, ddi_type, a, b, c] for id1, id2, ddi_type, a, b, c in
                       zip(df_all_dataset['drug1'], df_all_dataset['drug2'],
                           df_all_dataset['event_id'],
                           df_all_dataset['MeSH_ID'], df_all_dataset['Sign'],
                           df_all_dataset['Pattern'])]

        print(f"The {self.Allfilename} dataset has {len(all_dataset)} DDIs.")

        self.all_triplet = {}
        for value in all_dataset:
            self.all_triplet[value[2]] = [value[3], value[4].strip(), value[5]]  # event_id to effect, sign, pattern

        self.Meshids = []
        self.Signs = []
        self.Pattern = []
        for key, value in self.all_triplet.items():
            mesh = value[0].split("&")  # because one sample may has more than one mesh id
            sign = value[1].strip().split("&")
            patt = value[2]
            for i in mesh:
                if i not in self.Meshids:
                    self.Meshids.append(i)
            for i in sign:
                if i not in self.Signs:
                    self.Signs.append(i)
            if patt not in self.Pattern:
                self.Pattern.append(patt)

        evenidanddes = [[a, b] for a, b in zip(df_all_dataset['event_id'], df_all_dataset['description'])]
        self.all_descriptions = []
        self.eventid = []
        for item in evenidanddes:
            id, d = item[0], item[1]
            if d not in self.all_descriptions:
                self.all_descriptions.append(d)
                self.eventid.append(id)
        assert len(self.eventid) == len(self.all_descriptions)
        print(f"The {self.Allfilename} dataset has {len(self.all_descriptions)} descriptions")
        if not os.path.exists(self.output_file):
            os.makedirs(self.output_file)
        else:
            print(f"the file'{self.output_file}' exsit")
        file = os.path.join(self.output_file, f"{self.bert_vision}_mesh_text_embedding.pt")
        if os.path.exists(file):
            mesh_text_biobertemb = torch.load(file)
            #print("mesh_text_biobertemb",mesh_text_biobertemb.shape)
        else:
            mesh = MeshText(self.Meshids)
            mesh_text = mesh.get_text()

            tokenizer = AutoTokenizer.from_pretrained(f"data/{self.bert_vision}") #f"dmis-lab/{self.bert_vision}"
            model = AutoModel.from_pretrained(f"data/{self.bert_vision}")
            inputs = tokenizer(mesh_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            mesh_text_biobertemb = outputs.last_hidden_state
            #print("mesh_text_biobertemb", mesh_text_biobertemb.shape)#[114,131,768]
            torch.save(mesh_text_biobertemb, file)
            # df_disease_sim = pd.DataFrame(mesh_smi)
            # df_disease_sim.index = self.Meshids
            # df_disease_sim.to_csv(file, sep=',', )
        #mesh_text_biobertemb = torch.mean(mesh_text_biobertemb,1)
        print("mesh_text_biobertemb",mesh_text_biobertemb.shape,type(mesh_text_biobertemb))

        biobertembfile = os.path.join(self.output_file, f"{self.bert_vision}2.pt")


        if not os.path.exists(biobertembfile):
            tokenizer = AutoTokenizer.from_pretrained(f"data/{self.bert_vision}")
            model = AutoModel.from_pretrained(f"data/{self.bert_vision}")
            inputs = tokenizer(self.all_descriptions, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            biobertemb = outputs.last_hidden_state
            print("biogptemb", biobertemb.shape)
            torch.save(biobertemb, biobertembfile)
        else:
            biobertemb = torch.load(biobertembfile)  # []
        return mesh_text_biobertemb, biobertemb  # attrionehot,biogptemb

    def get_current_dataset(self):
    
        df_dataset = pd.read_csv(os.path.join(self.file_dir, self.file_name))
        current_dataset = [[id1, id2, ddi_type] for id1, id2, ddi_type in
                           zip(df_dataset['drug1'], df_dataset['drug2'],
                               df_dataset['event_id'])]

        print(f"The {self.file_name} dataset has {len(current_dataset)} DDIs.")


        current_dataset_eventids = list(df_dataset['event_id'])

        self.current_dataset_eventid_uni = []
        for i in current_dataset_eventids:
            if i not in self.current_dataset_eventid_uni:
                self.current_dataset_eventid_uni.append(i)
  
        self.eventid2embid = {}
        self.embid2eventid = {}
        count = 0
        mesh_current = []
        self.current_sign_id=[]
        self.current_mesh_id=[]
        self.current_patt_id=[]

        for type in self.current_dataset_eventid_uni:
            item = self.all_triplet[type]
            mesh = item[0].split("&")
            effect_emb = []
            for meshid in mesh:
                id = self.Meshids.index(meshid)
                meshtext = self.mesh_text_biobertemb[id,:].detach().numpy()
                effect_emb.append(meshtext)
                #id = self.Meshids.index(i)
                #meshids.append(id)
            mesh_emb = np.array(effect_emb)
            mesh_emb_mean = np.mean(mesh_emb,0)
            mesh_current.append(mesh_emb_mean)

            sign = item[1].strip().split("&")[0]
            sign_id = self.Signs.index(sign)
            self.current_sign_id.append(sign_id)

            effect = mesh[0]
            effect_id = self.Meshids.index(effect)
            self.current_mesh_id.append(effect_id)

            pattern = item[2]
            pattern_id = self.Pattern.index(pattern)
            self.current_patt_id.append(pattern_id)

            self.eventid2embid[type] = count
            self.embid2eventid[count] = type
            count = count + 1
    
        ind = [self.eventid.index(i) for i in self.current_dataset_eventid_uni]
   
        current_all_biogpt_emb = self.biobertemb[ind, :, :].to(self.device)

        current_all_mesh_emb = torch.tensor(np.array(mesh_current), dtype=torch.float32).to(self.device)

       
        self.new_current_dataset = []
        for item in current_dataset:
            embid = self.eventid2embid[int(item[2])]
            sample = [item[0], item[1], embid]
            self.new_current_dataset.append(sample)
        return current_all_biogpt_emb, current_all_mesh_emb

    def __getitem__(self, index):
        sample = self.new_current_dataset[index]
        if self.kg_pair_tokens is not None:
            sample = list(sample)
            sample.append(self._get_kg_evidence(sample[0], sample[1]))
        return (
        sample, self.mode,self.zsl_mode)

    def __len__(self):
        return len(self.new_current_dataset)

