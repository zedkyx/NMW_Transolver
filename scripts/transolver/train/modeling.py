import torch
import torch.nn as nn

from scripts.transolver.model_dict import get_model


class MultiNetWrapper(nn.Module):
    """
    Unified Wrapper for Transolver models.
    Supports both Multi-Net (one sub-net per channel) and Single-Net (one net for all channels) modes.
    """

    def __init__(self, args, input_dim: int, output_dim: int):
        super().__init__()
        self.models = nn.ModuleList()
        self.output_dim = output_dim
        # Use int(getattr(...)) to handle both YAML (bool/int) and CLI (int) inputs
        self.use_multi_net = bool(int(getattr(args, "use_multi_net", 1)))
        # Record model type to determine if SATO-specific args should be passed
        self.model_name = getattr(args, "model", "Transolver_1D")
        self.is_sato = (self.model_name == "SATO")
        # Transolver_1D + use_sato: insert Serialized_Attention (not the standalone SATO model)
        self.use_sato_module = (not self.is_sato) and bool(int(getattr(args, "use_sato", 0)))

        if self.use_multi_net:
            # Mode A: Independent sub-networks for each physical quantity
            for _ in range(output_dim):
                sub_model = self._build_submodel(args, input_dim, out_dim=1)
                self.models.append(sub_model)
        else:
            # Mode B: Standard single network predicting all quantities at once
            sub_model = self._build_submodel(args, input_dim, out_dim=output_dim)
            self.models.append(sub_model)

    def _build_submodel(self, args, input_dim, out_dim):
        use_edge_conv = bool(int(getattr(args, "use_edge_conv", 0)))
        model_module = get_model(args)
        
        # Check if the model class is named 'Model' (standard) or something else
        model_class = getattr(model_module, 'Model', None)
        if model_class is None:
            # Fallback for models that might export the class directly or under a different name
            model_class = model_module
        
        # Get model name to determine which parameters to pass
        model_name = getattr(args, "model", "Transolver_1D")
        is_sato = (model_name == "SATO")
        is_transolver_2d = (model_name == "Transolver_2D")
        
        # Common parameters (space_dim follows loaded pos; sdf_to_pos → 3)
        base_kwargs = {
            "space_dim": int(getattr(args, "space_dim", 2) or 2),
            "n_layers": args.n_layers,
            "n_hidden": args.n_hidden,
            "dropout": args.dropout,
            "n_head": args.n_heads,
            "mlp_ratio": args.mlp_ratio,
            "fun_dim": input_dim,
            "out_dim": out_dim,
            "slice_num": args.slice_num,
        }
        
        if is_sato:
            # SATO-specific parameters
            model = model_class(
                **base_kwargs,
                patch_size=getattr(args, "patch_size", 32),
                act=getattr(args, "act", "gelu"),
            )
        elif is_transolver_2d:
            # Transolver_2D-specific parameters (structured mesh)
            model = model_class(
                **base_kwargs,
                Time_Input=bool(int(getattr(args, "time_input", 0))),
                ref=args.ref,
                unified_pos=bool(args.unified_pos),
                H=getattr(args, "H", 256),
                W=getattr(args, "W", 256),
            )
        else:
            # Transolver_1D/plus: plus 使用 Eidetic Slice 分配 (token-dependent 温度 + Gumbel-Softmax)
            model_name = getattr(args, "model", "Transolver_1D")
            use_eidetic = (model_name == "Transolver_plus")
            use_linearno = bool(int(getattr(args, "use_linearno", 0)))
            use_sato = bool(int(getattr(args, "use_sato", 0)))
            use_geom_memory = bool(int(getattr(args, "use_geom_memory", 0)))
            gdim = int(getattr(args, "geom_memory_dim", 0) or 0)
            ghid = int(getattr(args, "geom_encoder_hidden", 0) or 0)
            model = model_class(
                **base_kwargs,
                Time_Input=bool(int(getattr(args, "time_input", 0))),
                ref=args.ref,
                unified_pos=bool(args.unified_pos),
                use_spectral=bool(int(getattr(args, "use_spectral", 0))),
                spectral_modes=int(getattr(args, "spectral_modes", 16)),
                spectral_hi_freq_boost=float(getattr(args, "spectral_hi_freq_boost", 1.0)),
                use_edge_conv=use_edge_conv,
                use_eidetic=use_eidetic,
                use_linearno=use_linearno,
                use_sato=use_sato,
                quad_sato=bool(int(getattr(args, "quad_sato", 0))),
                patch_size=int(getattr(args, "patch_size", 32)),
                shift=int(getattr(args, "shift", 1)),
                use_geom_memory=use_geom_memory,
                geom_memory_policy=str(getattr(args, "geom_memory_policy", "early")),
                geom_memory_dim=(gdim if gdim > 0 else None),
                geom_encoder_hidden=(ghid if ghid > 0 else None),
                geom_sdf_channel=int(getattr(args, "geom_sdf_channel", -1)),
            )
        return model

    def forward(
        self,
        pos,
        fx=None,
        mask=None,
        knn_idx=None,
        knn_valid=None,
        sato_indices=None,
        T=None,
        quadrature_weights=None,
        return_hidden=False,
        hidden_channel_idx: int = 0,
    ):
        kwargs = {}
        if fx is not None and fx.shape[-1] > 0:
            kwargs["fx"] = fx
        else:
            kwargs["fx"] = None
        if mask is not None:
            kwargs["mask"] = mask
        if knn_idx is not None:
            kwargs["knn_idx"] = knn_idx
        if knn_valid is not None:
            kwargs["knn_valid"] = knn_valid
        if (self.is_sato or self.use_sato_module) and sato_indices is not None:
            kwargs["sato_indices"] = sato_indices
        if T is not None:
            kwargs["T"] = T
        if quadrature_weights is not None:
            kwargs["quadrature_weights"] = quadrature_weights

        if self.use_multi_net:
            outputs = []
            hiddens = None
            for i, model in enumerate(self.models):
                want_h = bool(return_hidden) and (i == int(hidden_channel_idx))
                if want_h:
                    out_i, hiddens = model(pos, return_hidden=True, **kwargs)
                else:
                    out_i = model(pos, **kwargs)
                outputs.append(out_i)
            out = torch.cat(outputs, dim=-1)
            if return_hidden:
                return out, hiddens
            return out
        if return_hidden:
            return self.models[0](pos, return_hidden=True, **kwargs)
        return self.models[0](pos, **kwargs)

    def enable_attention_recording(self, enabled=True, channel_idx=None):
        if self.use_multi_net:
            if channel_idx is not None:
                if hasattr(self.models[channel_idx], 'enable_attention_recording'):
                    self.models[channel_idx].enable_attention_recording(enabled)
            else:
                for model in self.models:
                    if hasattr(model, 'enable_attention_recording'):
                        model.enable_attention_recording(enabled)
        else:
            if hasattr(self.models[0], 'enable_attention_recording'):
                self.models[0].enable_attention_recording(enabled)

    def get_last_attention(self, channel_idx=None):
        if self.use_multi_net:
            if channel_idx is not None:
                if hasattr(self.models[channel_idx], 'get_last_attention'):
                    return self.models[channel_idx].get_last_attention()
                else:
                    return []
            else:
                all_attn = []
                for model in self.models:
                    if hasattr(model, 'get_last_attention'):
                        all_attn.append(model.get_last_attention())
                    else:
                        all_attn.append([])
                return all_attn
        else:
            if hasattr(self.models[0], 'get_last_attention'):
                return self.models[0].get_last_attention()
            else:
                return []
