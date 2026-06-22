import torch
import torch.nn as nn
from zuko.transforms import Transform
from zuko.nn import MaskedMLP

class PermutationLayer(nn.Module):
    """
    A fixed permutation that accepts (x, context, m) but only permutes x.
    """
    def __init__(self, features, reverse=False):
        super().__init__()
        self.features = features
        
        if reverse:
            # Simple reverse permutation (standard for 2-layer flows)
            self.register_buffer('indices', torch.arange(features - 1, -1, -1))
        else:
            # Fixed random permutation
            self.register_buffer('indices', torch.randperm(features))
            
        # Calculate inverse indices for the inverse pass
        self.register_buffer('inv_indices', torch.argsort(self.indices))

    def forward(self, x, context=None, m=None):
        # Forward: x_new = x[indices]
        # LADJ is 0 because it's just a shuffle
        return x[:, self.indices], 0.0

    def inverse(self, y, context=None, m=None):
        # Inverse: x = y[inv_indices]
        return y[:, self.inv_indices], 0.0


class IndependentPolynomialResidualTransform(nn.Module):
    """
    Learns a residual correction: x_nominal = x_obs * exp(s) + t
    where s, t are polynomial functions of m, modeled by independent networks.
    Supports per-nuisance linear/quadratic terms and optional pairwise cross-terms.
    """
    def __init__(self, features, context_features, num_nuisances, hidden_features=[64, 64], activation=nn.ELU, central_nuisance_values=None, nuisance_scales=None, ladj_sign=1.0, quadratic_damping=0.01, cross_term_pairs=None, cross_term_damping=0.01):
        super().__init__()
        self.features = features
        self.num_nuisances = num_nuisances
        self.ladj_sign = float(ladj_sign)
        self.quadratic_damping = float(quadratic_damping)
        self.cross_term_damping = float(cross_term_damping)
        
        if central_nuisance_values is None:
            central_nuisance_values = torch.zeros(num_nuisances)
        self.register_buffer('central_nuisance_values', central_nuisance_values)

        if nuisance_scales is None:
            nuisance_scales = torch.ones(num_nuisances)
        self.register_buffer('nuisance_scales', nuisance_scales)

        validated_pairs = self._validate_cross_term_pairs(cross_term_pairs)
        self.register_buffer('cross_term_pairs', validated_pairs)
        
        # 1. Create the Autoregressive Adjacency Matrix
        # We need this to tell MaskedMLP which inputs are allowed to connect to which outputs.
        # Output has size (features * 4) because we predict 4 params (A_s, B_s, A_t, B_t) per feature.
        adjacency = self._create_adjacency(features, context_features, params_per_feature=4)
        # 2. Independent networks per nuisance
        self.nuisance_nets = nn.ModuleList()
        for _ in range(num_nuisances):
            # Zuko MaskedMLP signature: (adjacency, hidden_features, activation, ...)
            net = MaskedMLP(
                adjacency=adjacency,
                hidden_features=hidden_features,
                activation=activation
            )
            
            # Initialize to Identity (0 output)
            # The last layer in Zuko MaskedMLP is usually just a Linear layer
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nuisance_nets.append(net)

        # 3. One network per selected nuisance-pair cross-term (m_i * m_j)
        self.cross_nets = nn.ModuleList()
        if self.cross_term_pairs.shape[0] > 0:
            cross_adjacency = self._create_adjacency(features, context_features, params_per_feature=2)
            for _ in range(self.cross_term_pairs.shape[0]):
                cross_net = MaskedMLP(
                    adjacency=cross_adjacency,
                    hidden_features=hidden_features,
                    activation=activation
                )
                nn.init.zeros_(cross_net[-1].weight)
                nn.init.zeros_(cross_net[-1].bias)
                self.cross_nets.append(cross_net)

    def _validate_cross_term_pairs(self, cross_term_pairs):
        if cross_term_pairs is None:
            return torch.empty((0, 2), dtype=torch.long)

        if isinstance(cross_term_pairs, torch.Tensor):
            pairs_list = cross_term_pairs.tolist()
        else:
            pairs_list = list(cross_term_pairs)

        normalized_pairs = []
        seen = set()
        for raw_pair in pairs_list:
            if len(raw_pair) != 2:
                raise ValueError(f"Each cross-term pair must have exactly two indices, got: {raw_pair}")

            i, j = int(raw_pair[0]), int(raw_pair[1])
            if i == j:
                raise ValueError(f"Cross-term pair must have distinct indices, got ({i}, {j})")
            if not (0 <= i < self.num_nuisances and 0 <= j < self.num_nuisances):
                raise ValueError(
                    f"Cross-term pair ({i}, {j}) out of range for num_nuisances={self.num_nuisances}"
                )

            pair = (min(i, j), max(i, j))
            if pair not in seen:
                normalized_pairs.append(pair)
                seen.add(pair)

        if not normalized_pairs:
            return torch.empty((0, 2), dtype=torch.long)

        return torch.tensor(normalized_pairs, dtype=torch.long)

    def _create_adjacency(self, features, context_features, params_per_feature=4):
        """
        Constructs the boolean mask (adjacency matrix) for Zuko MaskedMLP.
        Shape: (out_features, in_features + context_features)
        """
        out_dim = features * params_per_feature
        
        # Indices of the output features (0, 0, 0, 0, 1, 1, 1, 1, ...)
        # Corresponds to which dimension 'i' the parameter controls
        row_indices = torch.arange(out_dim) // params_per_feature
        
        # Indices of the input features (0, 1, 2, ...)
        col_indices = torch.arange(features)
        
        # 1. Autoregressive Mask for Features
        # Rule: Output params for dim 'i' can only depend on inputs < 'i'
        # shape: (out_dim, features)
        mask_features = row_indices[:, None] > col_indices
        
        # 2. Context Mask
        # Rule: All outputs can depend on all context variables
        # shape: (out_dim, context_features)
        if context_features > 0:
            mask_context = torch.ones((out_dim, context_features), dtype=bool)
            # Concatenate features mask + context mask
            adjacency = torch.cat([mask_features, mask_context], dim=1)
        else:
            adjacency = mask_features

        return adjacency

    def _compute_params(self, x, context, m):
        B = x.shape[0]
        s_total = torch.zeros(B, self.features, device=x.device)
        t_total = torch.zeros(B, self.features, device=x.device)

        # num_nuisances == 0 → pure identity transform. Short-circuit before touching
        # the (size-0) nuisance buffers, which would otherwise fail to broadcast against
        # a non-empty m vector that a caller may still pass down.
        if self.num_nuisances == 0:
            return s_total, t_total

        m_scaled = (m - self.central_nuisance_values.unsqueeze(0)) / self.nuisance_scales.unsqueeze(0)
    
        for i, net in enumerate(self.nuisance_nets):
            # 1. Run MaskedMLP
            # Zuko MaskedMLP expects (x) or (x, context) depending on how it was built,
            # but internally it concatenates them. We pass both if context exists.
            if context is not None:
                # Zuko MaskedMLP forward signature is typically forward(x, context)
                raw_out = net(torch.cat([x, context], dim=-1))
            else:
                raw_out = net(x)
                
            # 2. Reshape [Batch, Features * 4] -> [Batch, Features, 4]
            coeffs = raw_out.view(B, self.features, 4)
            
            # 3. Get nuisance value (SCALED)
            m_val = m_scaled[:, i].view(B, 1)
            # Squaring boost the magnitude significantly if m_val > 1 (which it is for >1 sigma events)
            # We damp it to ensure linear term is the main driver initially.
            m_sq = m_val.pow(2) * self.quadratic_damping
            
            # 4. Contract: s = A*m + B*m^2_damped
            # coeffs[..., 0] = A_s, [..., 1] = B_s
            s_total = s_total + (coeffs[..., 0] * m_val + coeffs[..., 1] * m_sq)
            t_total = t_total + (coeffs[..., 2] * m_val + coeffs[..., 3] * m_sq)

        for pair_idx, cross_net in enumerate(self.cross_nets):
            if context is not None:
                cross_raw_out = cross_net(torch.cat([x, context], dim=-1))
            else:
                cross_raw_out = cross_net(x)

            # [Batch, Features * 2] -> [Batch, Features, 2]
            cross_coeffs = cross_raw_out.view(B, self.features, 2)
            i = int(self.cross_term_pairs[pair_idx, 0].item())
            j = int(self.cross_term_pairs[pair_idx, 1].item())
            m_cross = (m_scaled[:, i] * m_scaled[:, j]).view(B, 1) * self.cross_term_damping

            # cross_coeffs[..., 0] = C_s, [..., 1] = C_t
            s_total = s_total + cross_coeffs[..., 0] * m_cross
            t_total = t_total + cross_coeffs[..., 1] * m_cross
        
        return s_total, t_total


    def forward(self, x, context=None, m=None):
        if m is None: raise ValueError("Requires 'm'")
        s, t = self._compute_params(x, context, m)
        y = x * torch.exp(s) + t
        ladj = s.sum(dim=-1) * self.ladj_sign
        return y, ladj

    def inverse(self, y, context=None, m=None):
        if m is None: raise ValueError("Requires 'm'")
        x = torch.zeros_like(y)
        # AR inverse loop
        for i in range(self.features):
            s, t = self._compute_params(x, context, m)
            x[:, i] = (y[:, i] - t[:, i]) * torch.exp(-s[:, i])
        s, _ = self._compute_params(x, context, m)
        return x, -s.sum(dim=-1) * self.ladj_sign

    def set_central_nuisance_values(self, central_values):
        """
        Set the central (nominal) values for nuisance parameters.
        
        Args:
            central_values: Tensor of shape (num_nuisances,) with central values.
                           These are typically 0 for a well-defined nominal point.
        """
        if central_values.shape[0] != self.num_nuisances:
            raise ValueError(
                f"Expected {self.num_nuisances} nuisance values, "
                f"got {central_values.shape[0]}"
            )
        self.register_buffer('central_nuisance_values', central_values)
    
    def set_nuisance_scales(self, scales):
        """
        Set the scaling factors for nuisance parameters.
        """
        if scales.shape[0] != self.num_nuisances:
            raise ValueError(
                f"Expected {self.num_nuisances} nuisance scales, "
                f"got {scales.shape[0]}"
            )
        self.register_buffer('nuisance_scales', scales)

##############################################################
class SystematicCorrectedModel(nn.Module):
    def __init__(self, base_model, 
                 features_dim, 
                 context_dim, 
                 num_nuisances, 
                 num_residual_layers=1, 
                 hidden_features=[64, 64],
                 type="transform",
                 central_nuisance_values=None,
                 nuisance_scales=None,
                 ladj_sign=1.0,
                 quadratic_damping=0.01,
                 cross_term_pairs=None,
                 cross_term_damping=0.01):
        super().__init__()
        self.base_model = base_model
        
        # 1. Freeze Base Model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.feat_dim = features_dim
        self.ctx_dim = context_dim
        self.num_nuisances = num_nuisances
        self.type = 1 if type=="transform" else 0
        self.ladj_sign = float(ladj_sign)
        self.quadratic_damping = float(quadratic_damping)
        self.cross_term_damping = float(cross_term_damping)
        self.num_nuisances = num_nuisances
        self.type = 1 if type=="transform" else 0
        self.ladj_sign = float(ladj_sign)

        # Set central nuisance values (default to zeros)
        if central_nuisance_values is None:
            central_nuisance_values = torch.zeros(num_nuisances)
        elif not isinstance(central_nuisance_values, torch.Tensor):
            central_nuisance_values = torch.tensor(central_nuisance_values, dtype=torch.float32)
        self.register_buffer('central_nuisance_values', central_nuisance_values)

        # Set nuisance scales (default to ones)
        if nuisance_scales is None:
            nuisance_scales = torch.ones(num_nuisances)
        elif not isinstance(nuisance_scales, torch.Tensor):
            nuisance_scales = torch.tensor(nuisance_scales, dtype=torch.float32)
        self.register_buffer('nuisance_scales', nuisance_scales)

        # 3. Build Residual Stack with Permutations
        self.transforms = nn.ModuleList()
     
        for i in range(num_residual_layers):
            # A. Add Permutation (Between layers)
            # We add permutations for every layer after the first one to ensure mixing.
            # Alternatively, we can just alternate: Even=Identity, Odd=Reverse.
            if i > 0:
                # Use simple reversal for mixing. 
                # For >2 layers, you might prefer random permutations.
                self.transforms.append(
                    PermutationLayer(self.feat_dim, reverse=True)
                )

            # B. Add Residual Layer
            residual_layer = IndependentPolynomialResidualTransform(
                features=self.feat_dim,
                context_features=self.ctx_dim,
                num_nuisances=num_nuisances,
                hidden_features=hidden_features,
                central_nuisance_values=self.central_nuisance_values,
                quadratic_damping=self.quadratic_damping,
                cross_term_pairs=cross_term_pairs,
                cross_term_damping=self.cross_term_damping,
                nuisance_scales=self.nuisance_scales,
                ladj_sign=self.ladj_sign
            )
            self.transforms.append(residual_layer)

        # C. Final Permutation correction
        # If we have an even number of layers, we have applied an odd number of reversals (N-1).
        if num_residual_layers > 0 and num_residual_layers % 2 == 0:
            self.transforms.append(
                PermutationLayer(self.feat_dim, reverse=True)
            )

    def forward(self, x, context, m):
        """
        Flow: x_obs -> [Residual Stack] -> x_nom -> [Base] -> z
        """
        # 1. Apply Residual Stack (Correction)
        x_curr = x
        ladj_residual = 0.0
        for transform in self.transforms:
            x_curr, l = transform(x_curr, context=context, m=m)
            ladj_residual += l

        x_nominal = x_curr

        # 2. Apply Base Model
        if self.type ==1: 
            # Case A: TransferModel
            # self.base_model.T(context) returns a Transform object
            # call_and_ladj returns (z, ladj)
            z, ladj_base = self.base_model.T(context).call_and_ladj(x_nominal)
            
        else: 
            # Case B: Normal Zuko Flow
            # zuko_flow.transform(context) returns a Transform object
            z, ladj_base = self.base_model(context).transform.call_and_ladj(x_nominal)

        # 3. Base Probability
        if self.type == 0:
            log_prob_z = self.base_model.base(context).log_prob(z)
        else:
            log_prob_z = 0.

        return z, log_prob_z + ladj_base + ladj_residual

    def inverse(self, z, context, m):
        """
        Inverse Flow: z -> [Base Inv] -> x_nom -> [Residual Stack Inv] -> x_obs
        """
        # 1. Base Inverse
        if self.type == 1: 
            x_nom, log_det_inv = self.base_model.T(context).inv.call_and_ladj(z)
        else: 
            x_nom, log_det_inv = self.base_model(context).transform.inv.call_and_ladj(z)

        # 2. Residual Stack Inverse
        # IMPORTANT: Iterate list in REVERSE order
        x_curr = x_nom
        for transform in reversed(self.transforms):
            x_curr, l = transform.inverse(x_curr, context=context, m=m)
            log_det_inv += l

        return x_curr, log_det_inv

    @torch.no_grad()
    def sample(self, num_samples, context, m):
        """
        Flow: z -> [Base Inv] -> x_nom -> [Residual Stack Inv] -> x_obs
        """
        # 1. Sample z
        if self.type == 0:
            z = self.base_model.base(context).rsample((num_samples,))
        else:
            z = torch.randn(num_samples, self.feat_dim, device=context.device)

        x, _ = self.inverse(z, context, m)
        return x

    def set_central_nuisance_values(self, central_values):
        """
        Set the central (nominal) values for nuisance parameters across all residual layers.
        """
        if not isinstance(central_values, torch.Tensor):
            central_values = torch.tensor(central_values, dtype=torch.float32)
            
        if central_values.shape[0] != self.num_nuisances:
            raise ValueError(
                f"Expected {self.num_nuisances} nuisance values, "
                f"got {central_values.shape[0]}"
            )
            
        self.central_nuisance_values.copy_(central_values)
        
        for transform in self.transforms:
            if isinstance(transform, IndependentPolynomialResidualTransform):
                transform.set_central_nuisance_values(central_values)

    def set_nuisance_scales(self, scales):
        """
        Set the nuisance scales parameters across all residual layers.
        """
        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales, dtype=torch.float32)
        
        if scales.shape[0] != self.num_nuisances:
            raise ValueError(
                f"Expected {self.num_nuisances} nuisance scales, "
                f"got {scales.shape[0]}"
            )
        
        self.nuisance_scales.copy_(scales)
        
        for transform in self.transforms:
            if isinstance(transform, IndependentPolynomialResidualTransform):
                transform.set_nuisance_scales(scales)

    def get_central_nuisance_values(self):
        """
        Get the current central (nominal) values for nuisance parameters.
        
        Returns:
            Tensor of shape (num_nuisances,) with central values.
        """
        return self.central_nuisance_values.clone()
    
