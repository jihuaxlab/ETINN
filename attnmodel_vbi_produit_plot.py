from typing import List, Dict
import math
import ase
import torch
from torch import nn
import layer
from layer import ShiftedSoftplus
from einops import rearrange
import torch.nn.functional as F

import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os

plot_attn = True
global log_dir_path


def set_log_dir_path(new_path):
    global log_dir_path
    os.makedirs(new_path, exist_ok=True)
    log_dir_path = new_path
    # raise

# classes
class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim,
            dropout=0.
    ):
        """
        Feedforward layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. GELU
        4. Dropout
        5. Linear
        6. Dropout

        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Apply dropout to the input
        3. Apply non-linearity to the input
        4. Apply dropout to the input

        Args:
        -----
        dim: int
            Dimension of input
        hidden_dim: int
            Dimension of hidden layer
        dropout: float
            Dropout rate

        Returns:
        --------
        torch.Tensor
            Output of feedforward layer

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # apply feedforward layer to input tensor
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            id_=0
    ):
        """
        Attention Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. Rearrange
        4. LayerNorm
        5. Linear
        6. Rearrange
        7. Softmax
        8. Dropout
        9. Rearrange
        10. Linear
        11. Dropout

        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Rearrange input tensor
        3. Apply non-linearity to the input
        4. Rearrange input tensor
        5. Apply softmax to the input
        6. Apply dropout to the input
        7. Rearrange input tensor
        8. Apply non-linearity to the input

        """
        super().__init__()
        self.id_ = id_

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # layer norm
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)

        # sftmx
        self.attend = nn.Softmax(dim=-1)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # projections, split from x -> q, k, v
        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False
        )

        # project out
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        global log_dir_path
        # apply layernorm to x
        x = self.norm(x)

        # apply linear layer to x
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # rearrange x to original shape
        q, k, v = map(
            lambda t: rearrange(
                t,
                'b n (h d) -> b h n d',
                h=self.heads
            ), qkv)

        # #normalize key and values, known QK Normalization
        k = self.norm_k(k)
        v = self.norm_v(v)

        if plot_attn:
            # Suppose Q, K, and V are the already calculated tensors, with the shape of [batch, the number of heads, nm tokens, the feature of d length]
            d_k = q.shape[-1]  # shape of heads
       
            # cal attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            attn_probs = F.softmax(attn_scores, dim=-1)  # dim=-1 line avg
            # print(attn_probs.shape) # (batch, head, n_atom, n_atom)
       
            reg_attn = torch.mean(attn_probs, dim=1).cpu()
            # print(reg_attn.shape) torch.Size([1, n_atom, n_atom])
            attn_value = reg_attn[0]
            # print(attn_value.shape) # torch.Size([n_atom])
            attn_value_list = attn_value.tolist()

            attn_value_json = json.dumps(attn_value_list)

            with open(os.path.join(log_dir_path, 'layer{}_vbi_produit_value.json'), 'w') as f:
                json.dump(attn_value_list, f)

            img_size = reg_attn.shape[-1]
            img_data = reg_attn.view(img_size, img_size)
            plt.imshow(img_data.numpy(), cmap='viridis')
            # plt.axis('off')

            plt.savefig(os.path.join(log_dir_path, f'layer{self.id_}_vbi_produit_saliency_map_{time.time()}.png'), bbox_inches='tight', pad_inches=0, dpi=300)  # 保存图像，可调整dpi以改变清晰度

            plt.close()
            plt.figure()

            column_sums = torch.sum(img_data, axis=0)
            total_sum = torch.sum(column_sums)
            column_percentages = column_sums / total_sum
            column_percentages = column_percentages.numpy()
            
            plt.bar(range(img_size), column_percentages)
            plt.xlabel('Atom Index')
            # plt.xticks(range(img_size))  
            # plt.xticks(rotation=90)
            plt.savefig(os.path.join(log_dir_path, f'layer{self.id_}_vbi_produit_w_distribution_{time.time()}.png'), bbox_inches='tight', pad_inches=0, dpi=300)  
            plt.close()

            np.save(os.path.join(log_dir_path, f'layer{self.id_}_vbi_produit_w_data_{time.time()}.npy'), column_percentages)
            np.save(os.path.join(log_dir_path, f'layer{self.id_}_vbi_produit_map_data_{time.time()}.npy'), img_data.numpy())


        # attn
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            # Flash Attention
            out = F.scaled_dot_product_attention(q, k, v)

            # dropout
            out = self.dropout(out)

            # rearrange to original shape
            out = rearrange(out, 'b h n d -> b n (h d)')

            # project out
            # print("实际输出路径: ", log_dir_path)
            return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout=0.,
            is_train=False
    ):
        """
        Transformer Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Attention
        3. FeedForward

        Args:
        -----
        dim: int
            Dimension of input
        depth: int
            layers of transformers
        heads: int
            Number of heads
        dim_head: int
            Dimension of head
        mlp_dim: int
            Dimension of MLP
        dropout: float
            Dropout rate


        """
        super().__init__()

        self.is_train = is_train

        # layer norm
        self.norm = nn.LayerNorm(dim)

        # transformer layers data array
        self.layers = nn.ModuleList([])

        # add transformer layers as depth = transformer blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # attention
                Attention(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    id_=_
                ),
                # feedforward
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # layernorm before attention
            x = self.norm(x)

            # parallel
            x = x + attn(x) + ff(x)

        return self.norm(x)


class DensityModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step=0.1,
        cutoff_prob=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        print("AtomRepresentation cutoff set to: ", cutoff)
        self.atom_model = AtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            gaussian_expansion_step,
        )

        if cutoff_prob < 0:
            cutoff_prob = cutoff
        print("Prob cutoff set to: ", cutoff_prob)
        self.probe_model = ProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff_prob,
            gaussian_expansion_step,
        )

    def forward(self, input_dict):
        atom_representation = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result

class PainnDensityModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        cutoff_prob=-1,
        distance_embedding_size=30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        print("PAINN AtomRepresentation cutoff set to: ", cutoff)
        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

        if cutoff_prob < 0:
            cutoff_prob = cutoff
        print("PAINN Prob cutoff set to: ", cutoff_prob)
        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff_prob,
            distance_embedding_size,
        )

    def forward(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation_scalar, atom_representation_vector)
        return probe_result


class ProbeMessageModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step
        self.transformer_layers = 3
        self.transformer_heads = 4
        self.transformer_head_dim = 32
        self.transformer_mlp_dim = 128
        self.transformer_dropout = 0.1

        self.transformer_layers_later = 1
        self.transformer_heads_later = 4
        self.transformer_head_dim_later = 32
        self.transformer_mlp_dim_later = 128
        self.transformer_dropout_later = 0.1

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.messagesum_layers = nn.ModuleList(
            [
                layer.MessageSum(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup transitions networks
        self.probe_state_gate_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_state_size, hidden_state_size),
                    nn.Sigmoid(),
                )
                for _ in range(num_interactions)
            ]
        )
        self.probe_state_transition_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_state_size, hidden_state_size),
                    ShiftedSoftplus(),
                    nn.Linear(hidden_state_size, hidden_state_size),
                )
                for _ in range(num_interactions)
            ]
        )

        # Setup readout function
        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
            nn.Linear(hidden_state_size, 1),
        )

        self.transformer = Transformer(
            hidden_state_size,
            self.transformer_layers,
            self.transformer_heads,
            self.transformer_head_dim,
            self.transformer_mlp_dim,
            self.transformer_dropout
        )

        # initial fixations
        self.initial_fixations = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
        )

        self.transformer_gate_head_dsz = nn.Sequential(
            nn.Linear(hidden_state_size, 1),
            nn.Sigmoid())


        self.transformer_mid_add = Transformer(
            hidden_state_size,
            self.transformer_layers_later,
            self.transformer_heads_later,
            self.transformer_head_dim_later,
            self.transformer_mlp_dim_later,
            self.transformer_dropout_later
        )

        self.transformer_mid_produit = Transformer(
            hidden_state_size,
            self.transformer_layers_later,
            self.transformer_heads_later,
            self.transformer_head_dim_later,
            self.transformer_mlp_dim_later,
            self.transformer_dropout_later
        )

        self.transformer_later = Transformer(
            hidden_state_size,
            self.transformer_layers_later,
            self.transformer_heads_later,
            self.transformer_head_dim_later,
            self.transformer_mlp_dim_later,
            self.transformer_dropout_later
        )

        self.transformer_gate_head_mid = nn.Sequential(
            nn.Linear(hidden_state_size, 1),
            nn.Sigmoid())

        print("transformer_gate_head_mid : Sigmoid")

        self.transformer_gate_head = nn.Sequential(
            nn.Linear(hidden_state_size, 1),
            nn.LeakyReLU(negative_slope=0.05))
        print("transformer_gate_head  : Relu")

        print("*****************froget produit DSZs")

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation: List[torch.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):
        if compute_iri or compute_dori or compute_hessian:
            input_dict["probe_xyz"].requires_grad_()

        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]

        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        # print("Reshape probe_edges_displacement", probe_edges_displacement.shape)  # [total num edges, 3] not [batch * max edges, 3]
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        # print("edge_probe_offset", edge_probe_offset.shape, edge_probe_offset)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # print("input_dict atom_xyz", input_dict["atom_xyz"].shape)  # [batch, max atom, 3]
        # print("reshape atom_xyz", atom_xyz.shape)  # [total num atoms, 3]
        # print("input_dict probe_xyz", input_dict["probe_xyz"].shape)  # [batch, num prob, 3]
        # print("reshape probe_xyz", probe_xyz.shape)  # [batch * num prob, 3]
        # print("input_dict Cell", input_dict["cell"].shape, input_dict["cell"]) # [batch, 3, 3]
        # print("input_dict probe_edges", input_dict["probe_edges"].shape)  # [batch, max num edge, 2]
        # print("reshape probe_edges", probe_edges.shape)  # [total num edges, 2]
        # raise
        # Compute edge distances
        probe_edges_features = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )

        # print("probe_edges_features", probe_edges_features.shape) # [total num edges, 2]

        # Expand edge features in Gaussian basis
        probe_edge_state = layer.gaussian_expansion(
            probe_edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        # print("probe_edge_state", probe_edge_state.shape)  # [total num edges,  (cutoff - start(0.0))/0.4)]

        # Apply interaction layers
        probe_state = torch.zeros(
            (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
            device=atom_representation[0].device,
        )
        for msg_layer, gate_layer, state_layer, nodes in zip(
            self.messagesum_layers,
            self.probe_state_gate_functions,
            self.probe_state_transition_functions,
            atom_representation,
        ):
            msgsum = msg_layer(
                nodes,
                probe_edges,
                probe_edge_state,
                probe_edges_features,
                probe_state,
            )
            gates = gate_layer(probe_state)
            probe_state = probe_state * gates + (1 - gates) * state_layer(msgsum)

        # DSZs
        transformer_input = probe_state.unsqueeze(0)
        transformer_out = self.transformer(transformer_input)
        transformer_infos = transformer_out.squeeze()

        probe_state = self.initial_fixations(probe_state) + transformer_infos
        # print(probe_state.shape) # [xxx, 128]


        # forget gate
        transformer_input_mid = probe_state.unsqueeze(0)
        transformer_out_mid = self.transformer_mid_produit(transformer_input_mid)
        transformer_infos_mid = transformer_out_mid.squeeze()
        suppression_factor_mid = self.transformer_gate_head_mid(transformer_infos_mid)
        suppression_factor_mid = suppression_factor_mid.squeeze()
        suppression_factor_mid = suppression_factor_mid.unsqueeze(1)
        
        probe_state =  probe_state *  suppression_factor_mid

        # DSZs
        transformer_input_mid_1 = probe_state.unsqueeze(0)
        transformer_out_mid_1 = self.transformer_mid_add(transformer_input_mid_1)
        transformer_infos_mid_1 = transformer_out_mid_1.squeeze()

        probe_state = probe_state + transformer_infos_mid_1
        # print(probe_state.shape) # [xxx, 128]
        
        
        # Restack probe states
        probe_output = self.readout_function(probe_state).squeeze(1)

        # print("probe_state.shape:", probe_state.shape) # => torch.Size([xxx, 128])  
        transformer_input = probe_state.unsqueeze(0)
        # print("transformer_input.shape:", transformer_input.shape)  # => torch.Size([1, xxx, 128])

        transformer_gate = self.transformer_later(transformer_input)
        suppression_factor = self.transformer_gate_head(transformer_gate)
        suppression_factor = suppression_factor.squeeze()

        # print("suppression_factor.shape:", suppression_factor.shape, suppression_factor)  # => torch.Size([xxx])

        # print("probe_output.shape:", probe_output.shape, probe_output)
        probe_output = torch.mul(probe_output, suppression_factor)
        # raise
        
        probe_output = layer.pad_and_stack(
            torch.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )


        return probe_output


class AtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.gaussian_expansion_step = gaussian_expansion_step

        edge_size = int(math.ceil(self.cutoff / self.gaussian_expansion_step))

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.Interaction(
                    hidden_state_size, edge_size, self.cutoff, include_receiver=True
                )
                for _ in range(num_interactions)
            ]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def forward(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes = self.atom_embeddings(nodes)

        # Compute edge distances
        edges_features = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )

        # Expand edge features in Gaussian basis
        edge_state = layer.gaussian_expansion(
            edges_features, [(0.0, self.gaussian_expansion_step, self.cutoff)]
        )

        nodes_list = []
        # Apply interaction layers
        for int_layer in self.interactions:
            nodes = int_layer(nodes, edges, edge_state, edges_features)
            nodes_list.append(nodes)

        return nodes_list


class PainnAtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.PaiNNInteraction(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def forward(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        nodes_list_scalar = []
        nodes_list_vector = []
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector


class PainnProbeMessageModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.message_layers = nn.ModuleList(
            [
                layer.PaiNNInteractionOneWay(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, 1),
        )

        self.transformer_layers = 3
        self.transformer_heads = 4
        self.transformer_head_dim = 32
        self.transformer_mlp_dim = 128
        self.transformer_dropout = 0.1

        self.transformer_layers_later = 1
        self.transformer_heads_later = 4
        self.transformer_head_dim_later = 32
        self.transformer_mlp_dim_later = 128
        self.transformer_dropout_later = 0.1

        self.transformer = Transformer(
            hidden_state_size,
            self.transformer_layers,
            self.transformer_heads,
            self.transformer_head_dim,
            self.transformer_mlp_dim,
            self.transformer_dropout
        )

        # initial fixations
        self.initial_fixations = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
        )

        self.later_fixations = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            ShiftedSoftplus(),
        )

        self.transformer_mid_add = Transformer(
            hidden_state_size,
            self.transformer_layers_later,
            self.transformer_heads_later,
            self.transformer_head_dim_later,
            self.transformer_mlp_dim_later,
            self.transformer_dropout_later
        )

        self.transformer_mid_produit = Transformer(
            hidden_state_size,
            self.transformer_layers_later,
            self.transformer_heads_later,
            self.transformer_head_dim_later,
            self.transformer_mlp_dim_later,
            self.transformer_dropout_later
        )

        self.transformer_later = Transformer(
            hidden_state_size,
            self.transformer_layers_later,
            self.transformer_heads_later,
            self.transformer_head_dim_later,
            self.transformer_mlp_dim_later,
            self.transformer_dropout_later
        )

        self.transformer_gate_head_mid = nn.Sequential(
            nn.Linear(hidden_state_size, 1),
            nn.Sigmoid())

        self.transformer_gate_head = nn.Sequential(
            nn.Linear(hidden_state_size, 1),
            nn.LeakyReLU(negative_slope=0.05))

        print("ETINN painn forget + gated")

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation_scalar: List[torch.Tensor],
        atom_representation_vector: List[torch.Tensor],
        compute_iri=False,
        compute_dori=False,
        compute_hessian=False,
    ):
        if compute_iri or compute_dori or compute_hessian:
            input_dict["probe_xyz"].requires_grad_()

        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # Compute edge distances
        probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        probe_state_scalar = torch.zeros(
            (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )
        probe_state_vector = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )

        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):
            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        # Restack probe states
        
        # DSZs
        transformer_input = probe_state_scalar.unsqueeze(0)
        transformer_out = self.transformer(transformer_input)
        transformer_infos = transformer_out.squeeze()

        probe_state = self.initial_fixations(probe_state_scalar) + transformer_infos
        # print(probe_state.shape) # [xxx, 128]


        # forget gate
        transformer_input_mid = probe_state.unsqueeze(0)
        transformer_out_mid = self.transformer_mid_produit(transformer_input_mid)
        transformer_infos_mid = transformer_out_mid.squeeze()
        suppression_factor_mid = self.transformer_gate_head_mid(transformer_infos_mid)
        suppression_factor_mid = suppression_factor_mid.squeeze()
        suppression_factor_mid = suppression_factor_mid.unsqueeze(1)
        
        probe_state =  probe_state *  suppression_factor_mid

        # DSZs
        transformer_input_mid_1 = probe_state.unsqueeze(0)
        transformer_out_mid_1 = self.transformer_mid_add(transformer_input_mid_1)
        transformer_infos_mid_1 = transformer_out_mid_1.squeeze()

        probe_state = probe_state + transformer_infos_mid_1
        # print(probe_state.shape) # [xxx, 128]
        
        
        # Restack probe states
        probe_output = self.readout_function(probe_state).squeeze(1)

        # print("probe_state.shape:", probe_state.shape) # => torch.Size([xxx, 128])  
        transformer_input = probe_state.unsqueeze(0)
        # print("transformer_input.shape:", transformer_input.shape)  # => torch.Size([1, xxx, 128])

        transformer_gate = self.transformer_later(transformer_input)
        suppression_factor = self.transformer_gate_head(transformer_gate)
        suppression_factor = suppression_factor.squeeze()

        # print("suppression_factor.shape:", suppression_factor.shape, suppression_factor)  # => torch.Size([xxx])

        # print("probe_output.shape:", probe_output.shape, probe_output)
        probe_output = torch.mul(probe_output, suppression_factor)
        # raise
        
        probe_output = layer.pad_and_stack(
            torch.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )

        return probe_output
