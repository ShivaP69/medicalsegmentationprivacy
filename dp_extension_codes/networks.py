# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from collections import OrderedDict
from kornia.morphology import opening, closing,dilation,erosion
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def policy_v1(t, T_thin, T_thick, c):
    if t <= T_thin:
        return "close" # thin layers: avoid erosion; close bridges gaps
    elif t >= T_thick:
        return "open"  #thick layers: opening removes blobs/speckle
    else:
        return "both" # moderate: open->close (clean + fill)

def policy_v2(t, T_thin, T_thick, c):
    if t <= T_thin:
        return "close" # thin layers: avoid erosion; close bridges gaps
    elif t >= T_thick:
        return "both"  #thick layers: opening removes blobs/speckle
    else:
        return "both" # moderate: open->close (clean + fill)

def policy_v3(t, T_thin, T_thick, c):
    if t <= T_thin:
        return "close" # thin layers: avoid erosion; close bridges gaps
    elif t >= T_thick:
        return "close"  #thick layers: opening removes blobs/speckle
    else:
        return "both" # moderate: open->close (clean + fill)



@torch.no_grad()
def estimate_thickness_per_class_from_logits(
    logits: torch.Tensor,  # [B,C,H,W]
    *,
    num_classes: int,
) -> torch.Tensor:
    """
    Returns thickness [B,C] in pixels.
    thickness[b,c] = average over W of count_y( pred[b,y,x] == c )
    """
    pred = logits.argmax(dim=1)  # [B,H,W]
    B, H, W = pred.shape
    thickness = torch.zeros((B, num_classes), device=logits.device, dtype=torch.float32)

    for c in range(num_classes):
        mask = (pred == c).float()          # [B,H,W]
        col_counts = mask.sum(dim=1)        # [B,W]   count along height for each column
        thickness[:, c] = col_counts.mean(dim=1)  # [B]
    return thickness  # [B,C]

@torch.no_grad()
def batch_quantile_thresholds(
    thickness_bc: torch.Tensor,
    retinal_classes=range(1, 8),
    q_low: float = 0.20,
    q_high: float = 0.85,
) -> tuple[float, float]:

    vals_all = thickness_bc[:, list(retinal_classes)].reshape(-1).float()  # thickness quantile are calculated based on retinal_classes not all classes
    vals_pos = vals_all[vals_all > 0]

    if vals_pos.numel() >= 2: # If we have at least two positive thickness values, use them.
        vals = vals_pos
    elif vals_all.numel() >= 2: # If there are not enough positive values, use all values including zeros.
        vals = vals_all
    elif vals_all.numel() == 1: # If there is only one value, quantile cannot be computed. So return v and v
        v = vals_all.item()
        return v, v
    else:
        return 1.0, 5.0 # If the tensor is completely empty, use default thresholds.

    T_thin = torch.quantile(vals, q_low).item()
    T_thick = torch.quantile(vals, q_high).item()
    return T_thin, T_thick

@torch.no_grad()
def ops_from_batch_thickness(
    thickness_bc: torch.Tensor,          # [B,C]
    *,
    retinal_classes=range(1, 8),          # 1..7
    bg_class: int = 0,
    fluid_class: int | None = 8,
    q_low: float = 0.20,
    q_high: float = 0.85,
    policy_version="v1"
) -> dict[int, str]:
    """
    Returns ops_by_class for THIS batch.
    Uses ONE global (batch) threshold for all retinal layers, then bins each class
    using batch-average thickness of that class.
    """
    C = thickness_bc.size(1)

    T_thin, T_thick = batch_quantile_thresholds(
        thickness_bc, retinal_classes=retinal_classes, q_low=q_low, q_high=q_high
    )
    thickness_avg_c = thickness_bc.mean(dim=0)  # [C]
    ops = {}

    for c in range(C):
        if c == bg_class:
            ops[c] = "none"
            continue
        if fluid_class is not None and c == fluid_class:
            # safest default for fluid: usually DON'T morph, or maybe "close" if you see holes
            ops[c] = "none"
            continue
        if c not in set(retinal_classes):
            ops[c] = "none"
            continue
        t = float(thickness_avg_c[c].item())
        # binning with ONE global threshold
        if policy_version=="v1":
            ops[c]= policy_v1(t, T_thin, T_thick, c)
        if policy_version=="v2":
            ops[c]=policy_v2(t, T_thin, T_thick, c)
        if policy_version=="v3":
            ops[c]=policy_v3(t, T_thin, T_thick, c)
    return ops,T_thin, T_thick



def apply_kornia_morphology_multiclass(
    pred: torch.Tensor,
    operation: str = "close",
    kernel_size: int = 3,
    *,
    input_is_logits: bool = True,          # True if pred is logits (usual model output)
    apply_on_classes = [1,2,5,6,7],              # excluding 3 and 4
    keep_background_consistent: bool = True,
    alpha: float = 1.0,                    # 0=no morph, 1=full morph, in-between = blend
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    pred: [B,C,H,W] logits or probabilities
    returns: logits-like tensor (safe to feed to CE + your Dice pipeline)
    """
    if operation not in ["open", "close", "both", "none", "dilation", "erosion"]:
        raise ValueError("operation must be one of open/close/both/none/dilation/erosion")

    if operation == "none" or alpha <= 0.0:
        return pred

    # 1) Convert to probabilities (recommended for morphology)
    probs = F.softmax(pred, dim=1) if input_is_logits else pred
    probs = probs.clamp(min=0.0)
    probs_before=probs.clone()# for debuging

    B, C, H, W = probs.shape
    device = probs.device
    dtype = probs.dtype

    # 2) Build kernel
    k = torch.ones(kernel_size, kernel_size, device=device, dtype=dtype)

    # 3) Choose which classes to morph (VERY useful)
    if apply_on_classes is None:
        #class_idx = torch.arange(1, C, device=device)  # exclude background by default
        class_idx = torch.arange(C, device=device)
    else:
        class_idx = torch.tensor(list(apply_on_classes), device=device)

    probs_m = probs.clone()
    x = probs_m.index_select(1, class_idx)

    # 4) Apply morphology channel-wise
    if operation == "dilation":
        x = dilation(x, k)
    elif operation == "erosion":
        x = erosion(x, k)
    elif operation == "open":
        x = opening(x, k)
    elif operation == "close":
        x = closing(x, k)
    elif operation == "both":
        x = closing(opening(x, k), k)

    probs_m[:, class_idx] = x

    # 5) Optionally rebuild background as 1 - sum(non-bg)
    """if keep_background_consistent and C > 1:
        non_bg_sum = probs_m[:, 1:].sum(dim=1, keepdim=True)
        probs_m[:, 0:1] = (1.0 - non_bg_sum).clamp(min=0.0)"""

    # 6) Renormalize across classes (critical!)
    probs_m = probs_m.clamp(min=0.0)
    probs_m = probs_m / (probs_m.sum(dim=1, keepdim=True) + eps)

    # 7) Blend to reduce late-training instability
    probs_out = (1.0 - alpha) * probs + alpha * probs_m
    probs_out = probs_out / (probs_out.sum(dim=1, keepdim=True) + eps)
    #---------------------------------
    # Debug
    diff = (probs_before - probs_out).abs().mean().item()
    changed_argmax = (probs_before.argmax(dim=1) != probs_out.argmax(dim=1)).float().mean().item()
    print(f"[MORPH DEBUG] op={operation}, k={kernel_size}, prob_diff={diff:.8f}, changed_argmax={changed_argmax:.8f}")
    # ----------------------------------
    # 8) Return logits-like output so CrossEntropyLoss is happy
    # (CE accepts any real logits; log(probs) is valid)
    return torch.log(probs_out + eps)


def get_retinal_classes(num_classes: int, excluded_classes=None):
    if excluded_classes is None:
        excluded_classes = []

    if num_classes == 2:
        base = [1]
    elif num_classes == 9:
        base = list(range(1, 8))
    else:
        base = list(range(1, num_classes))

    return [c for c in base if c not in excluded_classes]

def apply_policy_morphology(
    logits: torch.Tensor,  # [B,C,H,W],
    kernel_size:int=3, # kernel size
    number_classes:int=9,
    q_low: float = 0.20,
    q_high: float = 0.85,
    return_debug:bool=False,
    policy_version="v1"
    excluded_classes=[3,4]):

    # call all necessary functions
    #retinal_classes=get_retinal_classes(number_classes)
    retinal_classes = get_retinal_classes(number_classes, excluded_classes=excluded_classes)
    thickness= estimate_thickness_per_class_from_logits(logits,num_classes=number_classes)
    ops_by_class,T_thin,T_thick=ops_from_batch_thickness(thickness,retinal_classes=retinal_classes,q_low=q_low,q_high=q_high,policy_version=policy_version)
    # -------------
    # Debug
    print("[POLICY DEBUG] ops_by_class:", ops_by_class)
    print("[POLICY DEBUG] mean thickness:", thickness.mean(dim=0).detach().cpu().numpy())
    # -------------
    # Group classes by operation (so we call the function a few times)
    groups = {}
    for c, op in ops_by_class.items():
        if op == "none":
            continue
        groups.setdefault(op, []).append(c)
    #------------
    print("[POLICY DEBUG] groups:", groups)
    #-------------
    out = logits
    for op, cls in groups.items():
        out = apply_kornia_morphology_multiclass(
            out,
            operation=op,
            kernel_size=kernel_size,
            input_is_logits=True,
            apply_on_classes=cls
        )
    #----------------
    diff = (logits - out).abs().mean().item()
    changed = (logits.argmax(dim=1) != out.argmax(dim=1)).float().mean().item()
    print(f"[POLICY DEBUG] total_logit_diff={diff:.8f}, changed_argmax={changed:.8f}")
    #----------------
    if not return_debug:
        return out
    else:
        # what should be saved
        debug_info = {
                "thickness_bc": thickness.detach().cpu(),              # [B,C]
                "thickness_mean_c": thickness.mean(dim=0).detach().cpu(),  # [C]
                "T_thin": float(T_thin),
                "T_thick": float(T_thick),
                "ops_by_class": dict(ops_by_class),
                "groups": {k: list(v) for k, v in groups.items()},
                "retinal_classes": list(retinal_classes),
            }
        return out,debug_info

class SoftDiskMorphology(nn.Module):
    """
    Differentiable morphology for multi-class logits [B, C, H, W].

    - max_kernel_size x max_kernel_size window (e.g. 9x9).
    - Learnable continuous radius r in [1, max_k].
    - Builds soft disk SE from radius.
    - Implements soft dilation / erosion / opening / closing.
    """

    def __init__(
        self,
        operation: str = "open",
        max_kernel_size: int = 9,
        init_kernel_size: int = 3,
        temperature: float = 0.5,
        strength: float = 10.0,   # how strongly SE biases the max/min
    ):
        super().__init__()
        assert operation in ["open", "close", "dilation", "erosion", "both", "none"]
        assert max_kernel_size % 2 == 1, "Use odd max_kernel_size like 3,5,7,9"

        self.operation = operation
        self.max_k = max_kernel_size
        self.temperature = temperature
        self.strength = strength

        # ---- learnable radius r \in [1, max_k] via logit ----
        init_r = float(init_kernel_size)
        init_sig = (init_r - 1.0) / (max_kernel_size - 1.0)  # in [0,1]
        init_sig = min(max(init_sig, 1e-3), 1.0 - 1e-3)
        p0 = math.log(init_sig / (1.0 - init_sig))          # logit
        self.logit_radius = nn.Parameter(torch.tensor(p0, dtype=torch.float32))

        # ---- precompute distances in K x K grid ----
        center = max_kernel_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(max_kernel_size),
            torch.arange(max_kernel_size),
            indexing="ij",
        )
        dist = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)  # [K, K]
        self.register_buffer("dist", dist)

    def current_radius(self) -> float:
        """For logging: effective radius as a Python float."""
        with torch.no_grad():
            sig = torch.sigmoid(self.logit_radius)
            r = 1.0 + sig * (self.max_k - 1.0)
            return float(r.item())

    def _build_soft_se(self, device):
        """
        Build soft disk SE in [0,1] of shape [K*K].
        """
        sig = torch.sigmoid(self.logit_radius)              # in (0,1)
        r = 1.0 + sig * (self.max_k - 1.0)                  # in [1, max_k]
        se_2d = torch.sigmoid((r - self.dist.to(device)) / self.temperature)  # [K, K]
        se_flat = se_2d.flatten()                           # [K*K]
        return se_flat  # in (0,1)

    def _soft_dilation(self, x, se_flat):
        """
        Soft dilation using additive bias from SE.

        x: [B, C, H, W]
        se_flat: [K*K]
        """
        B, C, H, W = x.shape
        K = self.max_k
        # 1) Extract local patches
        patches = F.unfold(x, kernel_size=K, padding=K // 2)  # [B, C*K*K, H*W]
        patches = patches.view(B, C, K * K, H * W)            # [B, C, S, N]

        # 2) SE shaped as [1,1,S,1] and broadcast – converted to bias
        se = se_flat.view(1, 1, K * K, 1)                     # [1,1,S,1]
        bias = self.strength * (se - 0.5)                     # centered around 0

        # 3) Add bias and take max over S dimension
        scores = patches + bias                               # [B,C,S,N]
        y, _ = scores.max(dim=2)                              # [B,C,N]

        return y.view(B, C, H, W)

    def _soft_erosion(self, x, se_flat):
        """
        Soft erosion as min with reversed bias.
        """
        B, C, H, W = x.shape
        K = self.max_k
        patches = F.unfold(x, kernel_size=K, padding=K // 2)  # [B, C*K*K, H*W]
        patches = patches.view(B, C, K * K, H * W)

        se = se_flat.view(1, 1, K * K, 1)
        bias = self.strength * (se - 0.5)

        # For erosion, higher SE weight should push value down,
        # so subtract the bias.
        scores = patches - bias                               # [B,C,S,N]
        y, _ = scores.min(dim=2)

        return y.view(B, C, H, W)

    def forward(self, pred_mask: torch.Tensor) -> torch.Tensor:
        """
        pred_mask: [B, C, H, W] logits.
        """
        if self.operation == "none":
            return pred_mask

        se_flat = self._build_soft_se(pred_mask.device)

        if self.operation == "dilation":
            return self._soft_dilation(pred_mask, se_flat)
        elif self.operation == "erosion":
            return self._soft_erosion(pred_mask, se_flat)
        elif self.operation == "open":
            x = self._soft_erosion(pred_mask, se_flat)
            x = self._soft_dilation(x, se_flat)
            return x
        elif self.operation == "close":
            x = self._soft_dilation(pred_mask, se_flat)
            x = self._soft_erosion(x, se_flat)
            return x
        elif self.operation == "both":
            x = self._soft_erosion(pred_mask, se_flat)
            x = self._soft_dilation(x, se_flat)
            x = self._soft_dilation(x, se_flat)
            x = self._soft_erosion(x, se_flat)
            return x
        else:
            raise ValueError(f"Unsupported operation {self.operation}")


def get_model(model_name, in_channels=1, num_classes=1, ratio=0.5,morphology=False,operation='None',kernel_size='None',learnable_radius=False,use_morph=False,retinal_layer_wise=False,deep_supervision=True,policy_version="v1",excluding_3_4_layer=True):
    #elif model_name == "y_net_gen":
        #model = YNet_general(in_channels, num_classes, ffc=False)
    #elif model_name == "y_net_gen_ffc":
        #model = YNet_general(in_channels, num_classes, ffc=True, ratio_in=ratio)
    if model_name == 'unet':
        model = UNet(in_channels, num_classes,morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)
    elif model_name == 'ReLayNet':
        model = ReLayNet(in_channels, num_classes,morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)
    elif model_name == 'LFUNet':
        model = LFUNet(in_channels, num_classes,morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)
    elif model_name == 'FCN8s':
        model = FCN8s(in_channels, num_classes,morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)
    elif model_name == 'NestedUNet':
        model = NestedUNet(in_channels, num_classes, deep_supervision=deep_supervision, morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)
    elif model_name == 'SimplifiedFCN8s':
        model= SimplifiedFCN8s(in_channels, num_classes,morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)
    elif model_name == 'ConvNet':
        model = ConvNet(in_channels, num_classes,morphology=morphology,operation=operation,kernel_size=kernel_size,learnable_radius=learnable_radius,use_morph=use_morph,retinal_layer_wise=retinal_layer_wise,policy_version=policy_version,excluding_3_4_layer=excluding_3_4_layer)

    else:
        print("Model name not found")
        assert False

    return model


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, num_groups=8):
        super().__init__()
        self.relu = nn.ReLU(inplace=False) # False for to be compatible with dpsgd
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(middle_channels)
        self.bn1 = nn.GroupNorm(num_groups, middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


"""class UNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
"""

class NestedUNet(nn.Module):
    def __init__(self, input_channels, num_classes, deep_supervision=True,morphology=False,operation='None',kernel_size='None',learnable_radius=False,use_morph=False,retinal_layer_wise=False,policy_version="v1",excluding_3_4_layer=True, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.policy_version=policy_version
        if excluding_3_4_layer:
            self.excluded_classes=[3,4]
            self.apply_on_classes=[1,2,5,6,7]
        else:
            self.excluding_classes=None
            self.apply_on_classes=[1,2,3,4,5,6,7]

        self.store_morph_debug = False
        self.last_morph_debug = None
        self.last_policy_debug = None
        self.morphology=morphology
        self.use_morph=use_morph
        self.num_classes=num_classes
        self.deep_supervision = deep_supervision
        self.learnable_radius=learnable_radius
        self.kernel_size= kernel_size
        self.operation = operation

        self.retinal_layer_wise = retinal_layer_wise
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # ---- NEW: learnable morphology layer ----
        if self.morphology and self.learnable_radius:
            # max_kernel_size can be > kernel_size (gives room to grow)
            self.morph_layer = SoftDiskMorphology(
                operation=operation,
                max_kernel_size=9,          # or 7, etc.
                init_kernel_size=kernel_size,
                temperature=0.5,
                strength=10.0,
            )
        else:
            self.morph_layer = None

    def _store_morph_debug(self, before, after, stage="final"): # Helper function for debugging
        if not self.store_morph_debug:
            return
        self.last_morph_debug = {
            "stage": stage,
            "before": before.detach().cpu(),
            "after": after.detach().cpu(),
        }

    def forward(self, input,use_morph: bool | None = None):
        # decide which morphology flag is *effective* in this call
        effective_morph = self.morphology and (self.use_morph if use_morph is None else use_morph)
        self.last_morph_debug = None
        self.last_policy_debug = None
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)

            """if self.morphology:
                out1 = apply_kornia_morphology_multiclass(out1, operation=self.operation, kernel_size=self.kernel_size)
                out2 = apply_kornia_morphology_multiclass(out2, operation=self.operation, kernel_size=self.kernel_size)
                out3 = apply_kornia_morphology_multiclass(out3, operation=self.operation, kernel_size=self.kernel_size)
                out4 = apply_kornia_morphology_multiclass(out4, operation=self.operation, kernel_size=self.kernel_size)"""
            if effective_morph:
                before_morph = out4.detach().clone()

                if self.morph_layer is not None:
                    out1 = self.morph_layer(out1)
                    out2 = self.morph_layer(out2)
                    out3 = self.morph_layer(out3)
                    out4 = self.morph_layer(out4)

                elif self.retinal_layer_wise:

                    out1=apply_policy_morphology(out1,kernel_size=self.kernel_size,number_classes=self.num_classes,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                    out2=apply_policy_morphology(out2,kernel_size=self.kernel_size,number_classes=self.num_classes,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                    out3=apply_policy_morphology(out3,kernel_size=self.kernel_size,number_classes=self.num_classes,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                    out4, policy_debug = apply_policy_morphology(out4,kernel_size=self.kernel_size,number_classes=self.num_classes,return_debug=True,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                    self.last_policy_debug = policy_debug
                else:

                    out1 = apply_kornia_morphology_multiclass(out1, operation=self.operation,
                                                              kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes)
                    out2 = apply_kornia_morphology_multiclass(out2, operation=self.operation,
                                                              kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes)
                    out3 = apply_kornia_morphology_multiclass(out3, operation=self.operation,
                                                              kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes)
                    out4 = apply_kornia_morphology_multiclass(out4, operation=self.operation,
                                                              kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes)
                after_morph = out4.detach().clone()
                self._store_morph_debug(before_morph, after_morph, stage="out4")
            return [out1, out2, out3, out4]

        else:
            out = self.final(x0_4)
            if effective_morph:
                before_morph = out.detach().clone()
                if self.learnable_radius:
                    out = self.morph_layer(out)
                elif self.retinal_layer_wise:
                    out,policy_debug=apply_policy_morphology(out,kernel_size=self.kernel_size,number_classes=self.num_classes,return_debug=True,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                    self.last_policy_debug = policy_debug
                else:  out = apply_kornia_morphology_multiclass(out, operation=self.operation, kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes) # this means 1..7 only
                after_morph = out.detach().clone()
                self._store_morph_debug(before_morph, after_morph, stage="out")
            return out



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32,morphology=False,operation='None',kernel_size='None',learnable_radius=False,use_morph=False,retinal_layer_wise=False,policy_version="v1",excluding_3_4_layer=True):
        super(UNet, self).__init__()
        self.policy_version=policy_version
        if excluding_3_4_layer:
            self.excluded_classes=[3,4]
            self.apply_on_classes=[1,2,5,6,7]
        else:
            self.excluding_classes=None
            self.apply_on_classes=[1,2,3,4,5,6,7]
        self.store_morph_debug = False
        self.last_morph_debug = None
        self.last_policy_debug = None
        self.morphology = morphology
        self.use_morph =use_morph
        self.kernel_size = kernel_size
        self.operation = operation
        self.out_channels=out_channels
        self.retinal_layer_wise=retinal_layer_wise
        features = init_features
        self.learnable_radius=learnable_radius
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        if self.morphology and self.learnable_radius:
                # max_kernel_size can be > kernel_size (gives room to grow)
                self.morph_layer = SoftDiskMorphology(
                    operation=operation,
                    max_kernel_size=9,          # or 7, etc.
                    init_kernel_size=kernel_size,
                    temperature=0.5,
                    strength=10.0,
                )
        else:
                self.morph_layer = None

        """if out_channels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax2d()"""
    def _store_morph_debug(self, before, after, stage="final"): # Helper function for debugging
            if not self.store_morph_debug:
                return
            self.last_morph_debug = {
                "stage": stage,
                "before": before.detach().cpu(),
                "after": after.detach().cpu(),
            }
    def forward(self, x,use_morph: bool | None = None):
        effective_morph = self.morphology and (self.use_morph if use_morph is None else use_morph)
        self.last_morph_debug = None
        self.last_policy_debug = None
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output=self.conv(dec1)

        if effective_morph:
            before_morph = output.detach().clone()
            if self.morph_layer is not None:
                output = self.morph_layer(output)
            elif self.retinal_layer_wise:
                output,policy_debug=apply_policy_morphology(output,kernel_size=self.kernel_size,number_classes=self.out_channels,return_debug=True,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                self.last_policy_debug = policy_debug
            else:
                output = apply_kornia_morphology_multiclass(output, operation=self.operation, kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes)
            after_morph=output.detach().clone()
            self._store_morph_debug(before_morph, after_morph, stage="final")
        return output
        #return self.activation(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(32, features)),
                    (name + "relu1", nn.ReLU(inplace=False)),  # inplace=False
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(32, features)),
                    (name + "relu2", nn.ReLU(inplace=False)),  # inplace=False
                ]
            )
        )


"""
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        if out_channels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.activation(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(32, features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(32, features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
"""
class BasicUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64, conv_ker=(3, 3),morphology=False,operation='None',kernel_size='None', **kwargs):
        super(BasicUNet, self).__init__()

        self.morphology = morphology
        self.kernel_size = kernel_size
        self.operation = operation
        conv_pad = (int((conv_ker[0] - 1) / 2), int((conv_ker[1] - 1) / 2))
        features = init_features
        self.encoder1 = BasicUNet._block(in_channels, features, name="enc1", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder2 = BasicUNet._block(features, features * 2, name="enc2", conv_kernel=conv_ker, conv_pad=conv_pad)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder3 = BasicUNet._block(features * 2, features * 4, name="enc3", conv_kernel=conv_ker,
                                         conv_pad=conv_pad)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.encoder4 = BasicUNet._block(features * 4, features * 8, name="enc4", conv_kernel=conv_ker,
                                         conv_pad=conv_pad)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck = BasicUNet._block(features * 8, features * 16, name="bottleneck", conv_kernel=conv_ker,
                                           conv_pad=conv_pad)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=(2, 2), stride=2)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = BasicUNet._block((features * 8) * 2, features * 8, name="dec4", conv_kernel=conv_ker,
                                         conv_pad=conv_pad)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(2, 2), stride=2)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder3 = BasicUNet._block((features * 4) * 2, features * 4, name="dec3", conv_kernel=conv_ker,
                                         conv_pad=conv_pad)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder2 = BasicUNet._block((features * 2) * 2, features * 2, name="dec2", conv_kernel=conv_ker,
                                         conv_pad=conv_pad)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=(2, 2), stride=2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = BasicUNet._block(features * 2, features, name="dec1", conv_kernel=conv_ker, conv_pad=conv_pad)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=(1, 1))
        self.softmax = nn.Softmax2d()

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        pool1, indices1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(enc3)
        enc4 = self.encoder4(pool3)
        pool4, indices4 = self.pool4(enc4)
        bottleneck = self.bottleneck(pool4)
        return enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck

    def forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck):
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return dec1

    def forward(self, x):
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck = BasicUNet.forward_encoder(
            self, x)
        dec1 = BasicUNet.forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck)
        output=self.softmax(self.conv(dec1))
        if self.morphology:
            output = apply_kornia_morphology_multiclass(output, operation=self.operation, kernel_size=self.kernel_size)

        return output

    @staticmethod
    def _block(in_channels, features, name, conv_kernel, conv_pad, repetitions=2):
        block_sequence = nn.Sequential()
        for count in range(repetitions):
            block_sequence.add_module(name + "conv" + str(count + 1), nn.Conv2d(in_channels=in_channels,
                                                                                out_channels=features,
                                                                                kernel_size=conv_kernel,
                                                                                padding=conv_pad, bias=False))

            block_sequence.add_module(name + "norm" + str(count + 1), nn.GroupNorm(32, features))
            block_sequence.add_module(name + "relu" + str(count + 1), nn.ReLU(inplace=False))
            in_channels = features
        return block_sequence


# ReLayNet - first UNet adapted for retina layers segmentation
# A. Roy, et al., “ReLayNet: retinal layer and fluid segmentation of macular optical coherence tomography using
# fully convolutional networks,” Biomed. Opt. Express, vol. 8, no. 8, pp. 3627–3642, Aug. 2017.
class ReLayNet(BasicUNet):
    def __init__(self, in_channels=3, num_classes=9, features=64, kernel=(7, 3),morphology=False,operation='None',kernel_size='None'):
        super().__init__(in_channels=in_channels, out_channels=num_classes, init_features=features, conv_ker=kernel)

        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        self.encoder1 = ReLayNet._block(in_channels, features, name="enc1", conv_kernel=kernel, conv_pad=conv_pad,
                                        repetitions=1)
        self.encoder2 = ReLayNet._block(features, features, name="enc2", conv_kernel=kernel, conv_pad=conv_pad,
                                        repetitions=1)
        self.encoder3 = ReLayNet._block(features, features, name="enc3", conv_kernel=kernel, conv_pad=conv_pad,
                                        repetitions=1)

        self.bottleneck = ReLayNet._block(features, features, name="bottleneck", conv_kernel=kernel, conv_pad=conv_pad,
                                          repetitions=1)

        self.decoder3 = ReLayNet._block(features * 2, features, name="dec3", conv_kernel=kernel, conv_pad=conv_pad,
                                        repetitions=1)
        self.decoder2 = ReLayNet._block(features * 2, features, name="dec2", conv_kernel=kernel, conv_pad=conv_pad,
                                        repetitions=1)
        self.decoder1 = ReLayNet._block(features * 2, features, name="dec1", conv_kernel=kernel, conv_pad=conv_pad,
                                        repetitions=1)
        self.softmax = nn.Softmax2d()

    def forward_encoder(self, x):
        enc1 = self.encoder1(x)
        pool1, indices1 = self.pool1(enc1)
        enc2 = self.encoder2(pool1)
        pool2, indices2 = self.pool2(enc2)
        enc3 = self.encoder3(pool2)
        pool3, indices3 = self.pool3(enc3)
        bottleneck = self.bottleneck(pool3)
        return enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck

    def forward_decoder(self, enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck):
        # enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck = ReLayNet.forward_encoder(self, x)

        dec3 = self.unpool3(bottleneck, indices3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.unpool2(dec3, indices2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.unpool1(dec2, indices1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.decoder1(dec1)

        # return self.softmax(self.conv(dec1))
        return self.decoder1(dec1)

    def forward(self, x):
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, bottleneck = ReLayNet.forward_encoder(self,
                                                                                                                   x)
        dec1 = ReLayNet.forward_decoder(self, enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3,
                                        bottleneck)
        return (
            self.softmax(self.conv(dec1)))


# LF-UNet - dual network combining UNet and FCN
# D. Ma, et al. “Cascade Dual-branch Deep Neural Networks for Retinal Layer and fluid Segmentation of Optical
# Coherence Tomography Incorporating Relative Positional Map,” in Proceedings of the Third
# Conference on Medical Imaging with Deep Learning, vol. 121, pp. 493–502, 2020.
class LFUNet(BasicUNet):
    def __init__(self, in_channels=3, num_classes=10, features=64, kernel=(7, 3),morphology=False,operation='None',kernel_size='None',learnable_radius=False,use_morph=False,retinal_layer_wise=False,policy_version="v1",excluding_3_4_layer=True):
        super().__init__(in_channels=in_channels, out_channels=num_classes, init_features=features, conv_ker=kernel)
        self.policy_version=policy_version
        if excluding_3_4_layer:
            self.excluded_classes=[3,4]
            self.apply_on_classes=[1,2,5,6,7]
        else:
            self.excluding_classes=None
            self.apply_on_classes=[1,2,3,4,5,6,7]
        self.store_morph_debug = False
        self.last_morph_debug = None
        self.last_policy_debug = None
        self.morphology = morphology
        self.kernel_size = kernel_size
        self.operation = operation
        self.num_classes=num_classes
        self.use_morph=use_morph
        self.retinal_layer_wise=retinal_layer_wise
        self.learnable_radius=learnable_radius
        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        self.upconv4a = nn.ConvTranspose2d(features * 16, features * 4, kernel_size=(2, 2), stride=2)
        self.upconv4b = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=2)
        self.upconv4c = nn.ConvTranspose2d(features * 2, features, kernel_size=(2, 2), stride=2)
        self.upconv4d = nn.ConvTranspose2d(features, features, kernel_size=(2, 2), stride=2)

        dilation4 = (kernel[0] + 1, kernel[1] + 1)
        dilation6 = (kernel[0] + 3, kernel[1] + 3)
        dilation8 = (kernel[0] + 5, kernel[1] + 5)
        paddil4 = (conv_pad[0] * dilation4[0], conv_pad[1] * dilation4[1])
        paddil6 = (conv_pad[0] * dilation6[0], conv_pad[1] * dilation6[1])
        paddil8 = (conv_pad[0] * dilation8[0], conv_pad[1] * dilation8[1])
        self.convdil4 = nn.Conv2d(features * 2, int(features / 2), kernel_size=kernel, padding=paddil4, bias=False,
                                  dilation=dilation4)
        self.convdil6 = nn.Conv2d(features * 2, int(features / 2), kernel_size=kernel, padding=paddil6, bias=False,
                                  dilation=dilation6)
        self.convdil8 = nn.Conv2d(features * 2, int(features / 2), kernel_size=kernel, padding=paddil8, bias=False,
                                  dilation=dilation8)

        self.dropout = nn.Dropout2d()
        self.conv = nn.Conv2d(in_channels=int(features / 2 * 3), out_channels=num_classes, kernel_size=(1, 1))
        if self.morphology and self.learnable_radius:
                # max_kernel_size can be > kernel_size (gives room to grow)
                self.morph_layer = SoftDiskMorphology(
                    operation=operation,
                    max_kernel_size=9,          # or 7, etc.
                    init_kernel_size=kernel_size,
                    temperature=0.5,
                    strength=10.0,
                )
        else:
                self.morph_layer = None
    def _store_morph_debug(self, before, after, stage="final"):
        if not self.store_morph_debug:
            return
        self.last_morph_debug = {
            "stage": stage,
            "before": before.detach().cpu(),
            "after": after.detach().cpu(),
        }
    def forward(self, x, use_morph: bool | None = None):
        effective_morph = self.morphology and (self.use_morph if use_morph is None else use_morph)
        self.last_morph_debug = None
        self.last_policy_debug = None
        enc1, pool1, indices1, enc2, pool2, indices2, enc3, pool3, indices3, enc4, pool4, indices4, bottleneck = LFUNet.forward_encoder(
            self, x)
        dec1 = LFUNet.forward_decoder(self, enc1, enc2, enc3, enc4, bottleneck)

        fcn4 = self.upconv4a(bottleneck) + pool3
        fcn3 = self.upconv4b(fcn4) + pool2
        fcn2 = self.upconv4c(fcn3) + pool1
        fcn1 = self.upconv4d(fcn2)

        fcn = torch.cat((dec1, fcn1), dim=1)
        dil8 = self.convdil8(fcn)
        dil6 = self.convdil6(fcn)
        dil4 = self.convdil4(fcn)

        cat_layer = torch.cat((dil8, dil6, dil4), dim=1)
        drop = self.dropout(cat_layer)
        # return self.conv(drop)
        #return self.softmax(self.conv(drop))
        output=self.conv(drop)
        if effective_morph:
            before_morph = output.detach().clone()
            if self.morph_layer is not None:
                output = self.morph_layer(output)
            elif self.retinal_layer_wise:
                output,policy_debug =apply_policy_morphology(output,kernel_size=self.kernel_size,number_classes=self.num_classes,return_debug=True,policy_version=self.policy_version,excluded_classes=self.excluded_classes)
                self.last_policy_debug = policy_debug
            else:
                output = apply_kornia_morphology_multiclass(output, operation=self.operation, kernel_size=self.kernel_size,apply_on_classes=self.apply_on_classes)
            after_morph = output.detach().clone()
            self._store_morph_debug(before_morph, after_morph, stage="final")
        return output


class FCN8s(nn.Module):

    def __init__(self, in_channels=3, num_classes=4, features=64, kernel=(3, 3),morphology=False,operation='None',kernel_size='None', **kwargs):
        super().__init__()
        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))
        self.morphology = morphology
        self.kernel_size = kernel_size
        self.operation = operation
        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel, padding=(100, 100))
        self.relu1_1 = nn.ReLU(inplace=False)
        self.conv1_2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel, padding=conv_pad)
        self.relu1_2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(in_channels=features, out_channels=(features * 2), kernel_size=kernel, padding=conv_pad)
        self.relu2_1 = nn.ReLU(inplace=False)
        self.conv2_2 = nn.Conv2d(in_channels=(features * 2), out_channels=(features * 2), kernel_size=kernel, padding=conv_pad)
        self.relu2_2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = nn.Conv2d(in_channels=(features * 2), out_channels=(features * 4), kernel_size=kernel, padding=conv_pad)
        self.relu3_1 = nn.ReLU(inplace=False)
        self.conv3_2 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 4), kernel_size=kernel, padding=conv_pad)
        self.relu3_2 = nn.ReLU(inplace=False)
        self.conv3_3 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 4), kernel_size=kernel, padding=conv_pad)
        self.relu3_3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu4_1 = nn.ReLU(inplace=False)
        self.conv4_2 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu4_2 = nn.ReLU(inplace=False)
        self.conv4_3 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu4_3 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu5_1 = nn.ReLU(inplace=False)
        self.conv5_2 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu5_2 = nn.ReLU(inplace=False)
        self.conv5_3 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu5_3 = nn.ReLU(inplace=False)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(in_channels=(features * 8), out_channels=((features * 8) * 8), kernel_size=(7, 7))
        self.relu6 = nn.ReLU(inplace=False)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(in_channels=((features * 8) * 8), out_channels=((features * 8) * 8), kernel_size=(1, 1))
        self.relu7 = nn.ReLU(inplace=False)
        self.drop7 = nn.Dropout2d()

        self.score_conv7 = nn.Conv2d(in_channels=((features * 8) * 8), out_channels=num_classes, kernel_size=(1, 1))
        self.score_pool4 = nn.Conv2d(in_channels=(features * 8), out_channels=num_classes, kernel_size=(1, 1))
        self.score_pool3 = nn.Conv2d(in_channels=(features * 4), out_channels=num_classes, kernel_size=(1, 1))

        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, bias=False)

    def forward(self, x):
        conv1 = self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x))))
        pool1 = self.pool1(conv1)

        conv2 = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(pool1))))
        pool2 = self.pool2(conv2)

        conv3 = self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(pool2))))))
        pool3 = self.pool3(conv3)

        conv4 = self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(pool3))))))
        pool4 = self.pool4(conv4)

        conv5 = self.relu5_3(self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(pool4))))))
        pool5 = self.pool5(conv5)

        conv6 = self.relu6(self.fc6(pool5))
        drop6 = self.drop6(conv6)

        conv7 = self.relu7(self.fc7(drop6))
        drop7 = self.drop7(conv7)

        upscore2 = self.upscore_conv7(self.score_conv7(drop7))  # 1/16
        score_pool4 = self.score_pool4(pool4)
        score_pool4c = score_pool4[:, :, 5:(5 + upscore2.size()[2]), 5:(5 + upscore2.size()[3])]  # 1/8
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4c)
        score_pool3 = self.score_pool3(pool3)
        score_pool3c = score_pool3[:, :, 9:(9 + upscore_pool4.size()[2]), 9:(9 + upscore_pool4.size()[3])]  # 1/8

        final = self.upscore_final(upscore_pool4 + score_pool3c)
        output = final[:, :, 31:(31 + x.size()[2]), 31:(31 + x.size()[3])].contiguous()
        if self.morphology:
            output = apply_kornia_morphology_multiclass(output, operation=self.operation, kernel_size=self.kernel_size)

        return output


"""
class FCN8s(nn.Module):

    def __init__(self, in_channels=3, num_classes=4, features=64, kernel=(3, 3)):
        super().__init__()
        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel, padding=(100, 100))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel, padding=conv_pad)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(in_channels=features, out_channels=(features * 2), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=(features * 2), out_channels=(features * 2), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = nn.Conv2d(in_channels=(features * 2), out_channels=(features * 4), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 4), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 4), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 8), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=(features * 8), out_channels=(features * 8), kernel_size=kernel,
                                 padding=conv_pad)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(in_channels=(features * 8), out_channels=((features * 8) * 8), kernel_size=(7, 7))
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(in_channels=((features * 8) * 8), out_channels=((features * 8) * 8), kernel_size=(1, 1))
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_conv7 = nn.Conv2d(in_channels=((features * 8) * 8), out_channels=num_classes, kernel_size=(1, 1))
        self.score_pool4 = nn.Conv2d(in_channels=(features * 8), out_channels=num_classes, kernel_size=(1, 1))
        self.score_pool3 = nn.Conv2d(in_channels=(features * 4), out_channels=num_classes, kernel_size=(1, 1))

        self.upscore_conv7 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4,
                                                stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4,
                                                stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16,
                                                stride=8, bias=False)

    def forward(self, x):
        conv1 = self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x))))
        pool1 = self.pool1(conv1)

        conv2 = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(pool1))))
        pool2 = self.pool2(conv2)

        conv3 = self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(pool2))))))
        pool3 = self.pool3(conv3)

        conv4 = self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(pool3))))))
        pool4 = self.pool4(conv4)

        conv5 = self.relu5_3(self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(pool4))))))
        pool5 = self.pool5(conv5)

        conv6 = self.relu6(self.fc6(pool5))
        drop6 = self.drop6(conv6)

        conv7 = self.relu7(self.fc7(drop6))
        drop7 = self.drop7(conv7)

        upscore2 = self.upscore_conv7(self.score_conv7(drop7))  # 1/16
        score_pool4 = self.score_pool4(pool4)
        score_pool4c = score_pool4[:, :, 5:(5 + upscore2.size()[2]), 5:(5 + upscore2.size()[3])]  # 1/8
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4c)
        score_pool3 = self.score_pool3(pool3)
        score_pool3c = score_pool3[:, :, 9:(9 + upscore_pool4.size()[2]), 9:(9 + upscore_pool4.size()[3])]  # 1/8

        final = self.upscore_final(upscore_pool4 + score_pool3c)
        output = final[:, :, 31:(31 + x.size()[2]), 31:(31 + x.size()[3])].contiguous()

        return output
"""


class SimplifiedFCN8s(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, features=64, kernel=(3, 3),morphology=False,operation='None',kernel_size='None', **kwargs):
        super().__init__()
        conv_pad = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2))

        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=kernel, padding=conv_pad)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(in_channels=features, out_channels=(features * 2), kernel_size=kernel, padding=conv_pad)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = nn.Conv2d(in_channels=(features * 2), out_channels=(features * 4), kernel_size=kernel, padding=conv_pad)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = nn.Conv2d(in_channels=(features * 4), out_channels=(features * 8), kernel_size=kernel, padding=conv_pad)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Simplified fully connected layer
        self.fc = nn.Conv2d(in_channels=(features * 8), out_channels=num_classes, kernel_size=(1, 1))
        self.relu_fc = nn.ReLU(inplace=True)

        # Upsampling layers
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, bias=False)

    def forward(self, x):
        conv1 = self.relu1_1(self.conv1_1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.relu2_1(self.conv2_1(pool1))
        pool2 = self.pool2(conv2)

        conv3 = self.relu3_1(self.conv3_1(pool2))
        pool3 = self.pool3(conv3)

        conv4 = self.relu4_1(self.conv4_1(pool3))
        pool4 = self.pool4(conv4)

        conv_fc = self.relu_fc(self.fc(pool4))

        upscore_pool4 = self.upscore_pool4(conv_fc)
        final = self.upscore_final(upscore_pool4)
        output = F.interpolate(final, size=(224, 224), mode='bilinear', align_corners=False)
        if self.morphology:
            output = apply_kornia_morphology_multiclass(output, operation=self.operation, kernel_size=self.kernel_size)

        return output
"""
class YNet_general(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(YNet_general, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        ############### Regular ##################################
        self.encoder1 = YNet_general._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = YNet_general._block(features, features * 2, name="enc2")  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = YNet_general._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = YNet_general._block(features * 4, features * 4, name="enc4")  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            ################ FFC #######################################
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            ############### Regular ##################################
            self.encoder1_f = YNet_general._block(in_channels, features, name="enc1_2")
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = YNet_general._block(features, features * 2, name="enc2_2")  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = YNet_general._block(features * 2, features * 4, name="enc3_2")  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = YNet_general._block(features * 4, features * 4, name="enc4_2")  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = YNet_general._block(features * 8, features * 16, name="bottleneck")  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 8) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 3, features, name="dec1")  # 2,3

        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = YNet_general._block((features * 6) * 2, features * 8, name="dec4")  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = YNet_general._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = YNet_general._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = YNet_general._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        batch = x.shape[0]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))

        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)

            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)

        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)

        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)

        else:
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(32, features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(32, features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


import torch
import torch.nn as nn
import torch.nn.functional as F

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=9, init_features=32,morphology=False,operation='None',kernel_size='None', **kwargs):
        super(ConvNet, self).__init__()
        self.morphology = morphology
        self.kernel_size = kernel_size
        self.operation = operation
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Assuming input size is 224x224
        self.flatten_dim = 64 * (224 // 4) * (224 // 4)

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels * (224 // 4) * (224 // 4))

        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample_final = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape the output to match the dimensions expected in the training function
        x = x.view(x.size(0), -1, 56, 56)  # Assuming output size should be 56x56

        x = self.upsample(x)  # Upsample to 112x112
        x = self.upsample_final(x)  # Upsample to 224x224
        if self.morphology:
            x = apply_kornia_morphology_multiclass(x, operation=self.operation, kernel_size=self.kernel_size)

        return x




class UNet_VGG(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32,morphology=False,operation='None',kernel_size='None', **kwargs):
        super(UNet_VGG, self).__init__()
        self.morphology = morphology
        self.kernel_size = kernel_size
        self.operation = operation
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        if out_channels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax2d()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output =self.activation(self.conv(dec1))
        if self.morphology:
            output = apply_kornia_morphology_multiclass(output, operation=self.operation, kernel_size=self.kernel_size)

        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.GroupNorm(32, features)),
                    (name + "relu1", nn.ReLU(inplace=False)),  # inplace=False
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.GroupNorm(32, features)),
                    (name + "relu2", nn.ReLU(inplace=False)),  # inplace=False
                ]
            )
        )


