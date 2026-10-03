"""
Multi-Architecture Analysis Framework
=====================================

Unified framework for analyzing informative correlation selectivity
across different generative model architectures:
- Diffusion Models
- Pre-trained ResNet50 (ImageNet classifier, Fisher Information)
- Autoregressive Models (ImageGPT, PixelCNN)
- Discriminative Models (Vision Transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
import scipy.io as sio
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

try:
    from diffusers import DDPMPipeline
except ImportError:
    DDPMPipeline = None

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    StableDiffusionPipeline = None

try:
    from transformers import AutoImageProcessor, AutoModel, ImageGPTForCausalImageModeling
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_informative_file(data_dir: Path, gamma: float) -> Optional[Path]:
    """Resolve informative stimulus filename across naming variants."""
    compact = str(gamma).replace('.', '')
    padded = f"{int(round(gamma * 100)):03d}"
    legacy = f"{int(round(gamma * 10)):02d}"

    candidates = [
        data_dir / f"informative_alpha_{compact}_64x64.mat",
        data_dir / f"informative_alpha_{padded}_64x64.mat",
        data_dir / f"informative_alpha_{legacy}.mat",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class GenerativeModelAnalyzer(ABC):
    """Abstract base class for generative model analysis."""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def estimate_likelihood(self, images: torch.Tensor) -> np.ndarray:
        """Estimate likelihood of images. Returns array of shape (n_images,)."""
        pass
    
    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate representations. Returns dict of tensors."""
        pass
    
    def load_stimuli(self, file_path: str, max_stimuli: Optional[int] = None) -> torch.Tensor:
        """Load stimuli from .mat file."""
        data = sio.loadmat(file_path)
        maps = data['maps']
        
        if maps.ndim == 3:
            maps = np.transpose(maps, (2, 0, 1))
        elif maps.ndim == 2:
            maps = maps[np.newaxis, :, :]
        
        # Limit number of stimuli if specified
        if max_stimuli is not None and maps.shape[0] > max_stimuli:
            maps = maps[:max_stimuli]
        
        maps = (maps - np.min(maps)) / (np.max(maps) - np.min(maps) + 1e-8)
        return torch.tensor(maps, dtype=torch.float32)
    
    def analyze_psychometric_set(self, data_dir: str = "data/stimuli/",
                                  alphas: List[float] = None,
                                  max_stimuli: Optional[int] = None) -> Dict:
        """Analyze likelihood across psychometric set."""
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        data_dir = Path(data_dir)
        results = {}

        # Informative
        logger.info("Analyzing informative stimuli...")
        informative_likelihoods = []
        loaded_alphas = []
        for alpha in alphas:
            filepath = resolve_informative_file(data_dir, alpha)

            if filepath is not None:
                stimuli = self.load_stimuli(str(filepath), max_stimuli=max_stimuli)
                likelihood = self.estimate_likelihood(stimuli)
                informative_likelihoods.append(likelihood)
                loaded_alphas.append(alpha)
                logger.info(f"  α={alpha}: {np.mean(likelihood):.4f} ± {np.std(likelihood):.4f}")

        if informative_likelihoods:
            results['informative'] = {
                'alphas': loaded_alphas,
                'likelihoods': np.array(informative_likelihoods),
                'mean': np.array([np.mean(l) for l in informative_likelihoods]),
                'std': np.array([np.std(l) for l in informative_likelihoods])
            }
        
        # Random
        logger.info("Analyzing random baseline...")
        filepath = data_dir / "random_baseline_64x64.mat"
        if filepath.exists():
            stimuli = self.load_stimuli(str(filepath), max_stimuli=max_stimuli)
            likelihood = self.estimate_likelihood(stimuli)
            results['random'] = {
                'mean': np.mean(likelihood),
                'std': np.std(likelihood)
            }
            logger.info(f"  Random: {np.mean(likelihood):.4f} ± {np.std(likelihood):.4f}")
        
        return results


class DiffusionModelAnalyzer(GenerativeModelAnalyzer):
    """Analyzer for Diffusion Models."""
    
    def __init__(self, model_name="google/ddpm-cifar10-32", num_steps=50, device=None):
        super().__init__(device)
        
        logger.info(f"Loading diffusion model: {model_name}")
        try:
            self.pipeline = DDPMPipeline.from_pretrained(model_name)
            self.pipeline = self.pipeline.to(self.device)
            self.scheduler = self.pipeline.scheduler
            self.unet = self.pipeline.unet
            self.num_steps = num_steps
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}")
            raise
    
    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for diffusion model."""
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        target_size = 32
        if image.shape[-1] != target_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        image = 2.0 * image - 1.0
        return image
    
    def estimate_likelihood(self, images: torch.Tensor, batch_size: int = 16,
                            n_timestep_samples: int = 16) -> np.ndarray:
        """
        Estimate VLB log-likelihood using the diffusion denoising objective.

        For each image, independently samples `n_timestep_samples` random
        diffusion timesteps and averages the MSE of the denoising prediction
        across them. This is an unbiased Monte Carlo estimate of the ELBO
        loss term and is consistent across stimulus conditions.
        """
        likelihoods = []
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        T = self.scheduler.config.num_train_timesteps

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Diffusion likelihood"):
                batch = images[i:i+batch_size]
                x_0 = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
                B = x_0.shape[0]

                total_loss = torch.zeros(B, device=self.device)

                for _ in range(n_timestep_samples):
                    # Sample one independent timestep per image in the batch
                    t_idx = torch.randint(0, T, (B,), device=self.device)

                    # Gather noise schedule coefficients per image
                    sqrt_alpha = torch.sqrt(alphas_cumprod[t_idx]).view(B, 1, 1, 1)
                    sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[t_idx]).view(B, 1, 1, 1)

                    # Forward diffusion: add noise
                    noise = torch.randn_like(x_0)
                    x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise

                    # Predict noise (UNet accepts batched timesteps)
                    noise_pred = self.unet(x_t, t_idx).sample

                    # Per-image MSE
                    loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2, 3))
                    total_loss += loss

                # Average over timestep samples; negate so higher = more likely
                batch_likelihood = -(total_loss / n_timestep_samples).cpu().numpy()
                likelihoods.append(batch_likelihood)

        return np.concatenate(likelihoods)
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate UNet features."""
        features = {}
        
        # This would require hooking into UNet layers
        # For now, return empty dict
        return features


class ResNetFisherAnalyzer(GenerativeModelAnalyzer):
    """
    Analyzer using pre-trained ResNet50 (ImageNet classifier) with Fisher Information.

    Registers forward hooks on each residual stage to extract per-layer activations,
    then computes Fisher Information w.r.t. alpha — the same methodology used for ViT
    in representational_geometry.py. This is principled and directly comparable to
    the ViT analysis.
    """

    LAYER_NAMES = ['stem', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

    def __init__(self, model_variant="resnet50", device=None):
        super().__init__(device)
        from torchvision import models

        logger.info(f"Loading pre-trained {model_variant} (ImageNet classifier)")
        if model_variant == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)

        self.model.to(self.device).eval()
        self._activations: Dict[str, torch.Tensor] = {}
        self._hook_handles: List = []
        self._register_hooks()
        logger.info(f"Hooked {len(self.LAYER_NAMES)} ResNet stages")

    def _register_hooks(self):
        hook_targets = {
            'stem':   self.model.maxpool,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
            'avgpool': self.model.avgpool,
        }
        for name, layer in hook_targets.items():
            handle = layer.register_forward_hook(
                lambda module, inp, out, n=name: self._save_activation(n, out)
            )
            self._hook_handles.append(handle)

    def _save_activation(self, name: str, activation: torch.Tensor):
        if activation.ndim == 4:
            self._activations[name] = F.adaptive_avg_pool2d(activation, 1).squeeze(-1).squeeze(-1)
        else:
            self._activations[name] = activation.squeeze()

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for ResNet with ImageNet normalization."""
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[-1] != 224:
            image = F.interpolate(
                image.unsqueeze(0), size=(224, 224),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        return (image - mean) / std

    def _forward_batch(self, images: torch.Tensor, batch_size: int = 16) -> Dict[str, List[np.ndarray]]:
        """Run forward pass and collect hook outputs for all images."""
        layer_acts: Dict[str, List[np.ndarray]] = {n: [] for n in self.LAYER_NAMES}

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="ResNet forward"):
                batch = images[i:i+batch_size]
                x = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
                self.model(x)
                for name in self.LAYER_NAMES:
                    layer_acts[name].append(self._activations[name].cpu().numpy())

        return {name: np.vstack(chunks) for name, chunks in layer_acts.items()}

    def estimate_likelihood(self, images: torch.Tensor, batch_size: int = 16) -> np.ndarray:
        """Per-image L2 norm of avgpool features (framework compatibility)."""
        acts = self._forward_batch(images, batch_size)
        return np.linalg.norm(acts['avgpool'], axis=1)

    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        acts = self._forward_batch(images)
        return {name: torch.tensor(v) for name, v in acts.items()}

    @staticmethod
    def _fisher_information(activations: np.ndarray, params: np.ndarray) -> float:
        """Fisher Information via finite differences on per-condition mean activations."""
        sorted_idx = np.argsort(params)
        sorted_acts = activations[sorted_idx]
        sorted_params = params[sorted_idx]

        gradients = []
        for i in range(len(sorted_acts) - 1):
            dp = sorted_params[i+1] - sorted_params[i]
            if dp > 1e-8:
                gradients.append((sorted_acts[i+1] - sorted_acts[i]) / dp)

        if not gradients:
            return 0.0
        g = np.array(gradients)
        return float(np.trace(np.cov(g.T))) if g.shape[1] > 1 else float(np.var(g))

    def analyze_psychometric_set(self, data_dir: str = "data/stimuli/",
                                  alphas: List[float] = None,
                                  max_stimuli: Optional[int] = None) -> Dict:
        """
        Layer-wise Fisher Information analysis across alpha levels.
        Returns results in the same format as other analyzers for framework compatibility,
        plus a 'layer_fisher' key with per-layer sensitivity.
        """
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        data_dir = Path(data_dir)
        all_stimuli: Dict[str, torch.Tensor] = {}
        loaded_alphas: List[float] = []

        for alpha in alphas:
            filepath = resolve_informative_file(data_dir, alpha)
            if filepath is not None:
                stimuli = self.load_stimuli(str(filepath), max_stimuli=max_stimuli)
                all_stimuli[f'alpha_{alpha}'] = stimuli
                loaded_alphas.append(alpha)

        filepath = data_dir / "random_baseline_64x64.mat"
        if filepath.exists():
            stimuli = self.load_stimuli(str(filepath), max_stimuli=max_stimuli)
            all_stimuli['random'] = stimuli

        # Extract activations for all stimuli in one pass
        combined = torch.cat(list(all_stimuli.values()), dim=0)
        all_acts = self._forward_batch(combined)

        # Split back by condition
        idx = 0
        condition_acts: Dict[str, Dict[str, np.ndarray]] = {}
        for key, stimuli in all_stimuli.items():
            n = len(stimuli)
            condition_acts[key] = {name: all_acts[name][idx:idx+n] for name in self.LAYER_NAMES}
            idx += n

        # Fisher Information per layer
        alpha_params = np.array(loaded_alphas)
        layer_fisher: Dict[str, float] = {}
        for layer_name in self.LAYER_NAMES:
            per_cond_means = np.array([
                np.mean(condition_acts[f'alpha_{a}'][layer_name], axis=0)
                for a in loaded_alphas
            ])
            fi = self._fisher_information(per_cond_means, alpha_params)
            layer_fisher[layer_name] = fi
            logger.info(f"  {layer_name}: Fisher = {fi:.4f}")

        peak_layer = max(layer_fisher, key=layer_fisher.get)
        logger.info(f"  Peak: {peak_layer} (Fisher = {layer_fisher[peak_layer]:.4f})")

        # Per-image likelihood (avgpool norm) for framework plot compatibility
        informative_likelihoods = []
        for alpha in loaded_alphas:
            lhood = np.linalg.norm(condition_acts[f'alpha_{alpha}']['avgpool'], axis=1)
            informative_likelihoods.append(lhood)
            logger.info(f"  α={alpha}: feature norm = {np.mean(lhood):.4f} ± {np.std(lhood):.4f}")

        random_lhood = np.linalg.norm(condition_acts['random']['avgpool'], axis=1)
        logger.info(f"  Random: feature norm = {np.mean(random_lhood):.4f} ± {np.std(random_lhood):.4f}")

        return {
            'informative': {
                'alphas': loaded_alphas,
                'likelihoods': np.array(informative_likelihoods),
                'mean': np.array([np.mean(l) for l in informative_likelihoods]),
                'std': np.array([np.std(l) for l in informative_likelihoods]),
            },
            'random': {
                'mean': float(np.mean(random_lhood)),
                'std': float(np.std(random_lhood)),
            },
            'layer_fisher': {
                'layer_names': self.LAYER_NAMES,
                'fisher_values': [layer_fisher[n] for n in self.LAYER_NAMES],
                'peak_fisher': float(layer_fisher[peak_layer]),
                'peak_layer': peak_layer,
            },
        }


class AutoregressiveAnalyzer(GenerativeModelAnalyzer):
    """Analyzer for Autoregressive Models (ImageGPT, PixelCNN)."""
    
    def __init__(self, model_name="openai/imagegpt-medium", device=None):
        super().__init__(device)
        
        logger.info(f"Loading autoregressive model: {model_name}")
        try:
            from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
            
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = ImageGPTForCausalImageModeling.from_pretrained(model_name)
            self.model.to(self.device).eval()
            self.is_imagegpt = True
            
        except Exception as e:
            logger.error(f"Could not load {model_name}: {e}")
            # Fallback to Vision Transformer
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.model.to(self.device).eval()
            self.is_imagegpt = False
    
    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for autoregressive model."""
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        # Normalize to [0, 1] for processor
        image = torch.clamp(image, 0, 1)
        return image
    
    def estimate_likelihood(self, images: torch.Tensor, batch_size: int = 16) -> np.ndarray:
        """Estimate likelihood using model confidence."""
        scores = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="AR likelihood"):
                batch = images[i:i+batch_size]
                
                from PIL import Image
                pil_images = []
                for img in batch:
                    img_np = (img * 255).numpy().astype(np.uint8)
                    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                        if img_np.shape[0] == 1:
                            img_np = np.repeat(img_np, 3, axis=0) # Make it 3 channels
                        img_np = np.transpose(img_np, (1, 2, 0))
                    elif img_np.ndim == 2:
                        img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
                    pil_images.append(Image.fromarray(img_np))
                
                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if getattr(self, "is_imagegpt", False) and "input_ids" in inputs:
                    # For causal modeling, compute batch-wise loss manually for granular score
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    batch_loss = batch_loss.view(shift_labels.size(0), -1).mean(dim=-1)
                    score = -batch_loss.cpu().numpy()
                else:
                    outputs = self.model(**inputs)
                    last_hidden = outputs.last_hidden_state
                    score = torch.norm(last_hidden, dim=(1, 2)).cpu().numpy()
                    
                scores.append(score)
        
        return np.concatenate(scores)
    
    def extract_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract model features."""
        features = {}
        with torch.no_grad():
            from PIL import Image
            pil_images = []
            for img in images:
                img_np = (img * 255).numpy().astype(np.uint8)
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                    if img_np.shape[0] == 1:
                        img_np = np.repeat(img_np, 3, axis=0) # Make it 3 channels
                    img_np = np.transpose(img_np, (1, 2, 0))
                elif img_np.ndim == 2:
                    img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
                pil_images.append(Image.fromarray(img_np))

            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                features['hidden_states'] = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                features['hidden_states'] = outputs.last_hidden_state

        return features

    def _extract_mean_features(self, images: torch.Tensor, batch_size: int = 16) -> np.ndarray:
        """Mean-pool the last hidden state to get a (N, D) feature matrix."""
        all_feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="AR features"):
                batch = images[i:i + batch_size]
                from PIL import Image as PILImage
                pils = []
                for img in batch:
                    arr = (img.clamp(0, 1) * 255).numpy().astype(np.uint8)
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    elif arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                        if arr.shape[-1] == 1:
                            arr = np.repeat(arr, 3, axis=-1)
                    pils.append(PILImage.fromarray(arr))
                inputs = self.processor(images=pils, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    h = outputs.hidden_states[-1]  # (B, seq, D)
                elif hasattr(outputs, "last_hidden_state"):
                    h = outputs.last_hidden_state
                else:
                    h = outputs.logits
                feat = h.float().mean(dim=1).cpu()  # mean pool over sequence → (B, D)
                all_feats.append(feat.numpy())
        return np.vstack(all_feats)

    def analyze_psychometric_set(self, data_dir: str = "data/stimuli/",
                                  alphas: List[float] = None,
                                  max_stimuli: Optional[int] = None) -> Dict:
        results = super().analyze_psychometric_set(data_dir, alphas, max_stimuli)
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        data_dir = Path(data_dir)
        all_feats, all_labels = [], []
        for alpha in alphas:
            fp = resolve_informative_file(data_dir, alpha)
            if fp is not None:
                stimuli = self.load_stimuli(str(fp), max_stimuli=max_stimuli)
                feats = self._extract_mean_features(stimuli)
                all_feats.append(feats)
                all_labels.extend([alpha] * len(feats))
        if all_feats:
            results['features'] = {
                'hidden': np.vstack(all_feats),
                'labels': np.array(all_labels),
            }
        return results


class SDUNetAnalyzer(GenerativeModelAnalyzer):
    """
    Analyzer extracting intermediate UNet representations from Stable Diffusion.

    Images are encoded to the VAE latent space, noise is added at a fixed timestep
    (default t=500), and the noisy latents are passed through the UNet. The UNet
    mid-block (bottleneck) output is spatially pooled to produce a feature vector.
    This isolates the denoiser's internal geometry from the VAE's KL-regularized
    bottleneck, which was confounding the prior SD-VAE accessibility result.
    """

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5",
                 probe_timestep=500, device=None):
        super().__init__(device)
        self.probe_timestep = probe_timestep
        self._mid_block_feat: Optional[torch.Tensor] = None

        if StableDiffusionPipeline is None:
            raise ImportError("diffusers>=0.21 required: pip install diffusers")

        logger.info(f"Loading Stable Diffusion UNet: {model_name} (t={probe_timestep})")
        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.vae = pipe.vae.to(self.device).eval()
        self.unet = pipe.unet.to(self.device).eval()
        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(1000)
        self._dtype = dtype

        # Cache null (unconditional) text embedding once.
        # text_encoder stays on CPU; move input_ids to CPU explicitly,
        # then move the resulting embedding to the target device.
        null_ids = pipe.tokenizer(
            [""], padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids  # CPU tensor — matches text_encoder device
        self._null_emb = pipe.text_encoder(null_ids)[0].detach().to(self.device)  # (1, 77, 768)

        # Hook the UNet mid-block output
        self._hook = self.unet.mid_block.register_forward_hook(
            lambda m, inp, out: self._capture(out)
        )
        logger.info("  UNet mid-block hooked")

    def _capture(self, output):
        feat = output[0] if isinstance(output, (tuple, list)) else output
        self._mid_block_feat = feat  # (B, C, H, W)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """3-channel, 512×512, scaled to [-1, 1]."""
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.shape[-1] != 512:
            image = F.interpolate(
                image.unsqueeze(0), size=(512, 512),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        return (2.0 * image - 1.0).to(self._dtype)

    def _vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.mean
        return latents * self.vae.config.scaling_factor

    def _add_noise(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = latents.shape[0]
        noise = torch.randn_like(latents)
        t = torch.full((B,), self.probe_timestep, device=self.device, dtype=torch.long)
        noisy = self.scheduler.add_noise(latents, noise, t)
        return noisy, noise, t

    def _null_cond(self, batch_size: int) -> torch.Tensor:
        return self._null_emb.expand(batch_size, -1, -1).to(self._dtype)

    def extract_features(self, images: torch.Tensor, batch_size: int = 2) -> Dict[str, torch.Tensor]:
        """Returns {'mid_block': (N, 1280)} after spatial average pooling."""
        all_feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="SD UNet features"):
                batch = images[i:i + batch_size]
                x = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
                latents = self._vae_encode(x)
                noisy, _, t = self._add_noise(latents)
                self.unet(noisy, t, encoder_hidden_states=self._null_cond(len(batch)))
                feat = F.adaptive_avg_pool2d(self._mid_block_feat, 1).squeeze(-1).squeeze(-1)
                all_feats.append(feat.float().cpu())
        return {'mid_block': torch.cat(all_feats, dim=0)}

    def estimate_likelihood(self, images: torch.Tensor, batch_size: int = 2) -> np.ndarray:
        """Negative noise-prediction MSE at probe_timestep (higher = more predictable)."""
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="SD UNet likelihood"):
                batch = images[i:i + batch_size]
                x = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
                latents = self._vae_encode(x)
                noisy, noise, t = self._add_noise(latents)
                noise_pred = self.unet(
                    noisy, t, encoder_hidden_states=self._null_cond(len(batch))
                ).sample
                mse = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2, 3))
                scores.append((-mse).float().cpu().numpy())
        return np.concatenate(scores)

    def analyze_psychometric_set(self, data_dir: str = "data/stimuli/",
                                  alphas: List[float] = None,
                                  max_stimuli: Optional[int] = None) -> Dict:
        """Standard analysis + feature collection for downstream probing."""
        results = super().analyze_psychometric_set(data_dir, alphas, max_stimuli)
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        data_dir = Path(data_dir)
        all_feats, all_labels = [], []
        for alpha in alphas:
            fp = resolve_informative_file(data_dir, alpha)
            if fp is not None:
                stimuli = self.load_stimuli(str(fp), max_stimuli=max_stimuli)
                feats = self.extract_features(stimuli)['mid_block'].numpy()
                all_feats.append(feats)
                all_labels.extend([alpha] * len(feats))
        if all_feats:
            results['features'] = {
                'mid_block': np.vstack(all_feats),
                'labels': np.array(all_labels),
            }
        return results


class CLIPAnalyzer(GenerativeModelAnalyzer):
    """
    Analyzer using CLIP's vision encoder (ViT-B/32).

    The CLS token from the final encoder layer serves as the image representation.
    Likelihood proxy: L2 norm of the CLS embedding (higher norm = stronger response).
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        super().__init__(device)
        logger.info(f"Loading CLIP: {model_name}")
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        logger.info("  CLIP vision encoder ready")

    def _pil_batch(self, images: torch.Tensor):
        from PIL import Image
        out = []
        for img in images:
            arr = (img.clamp(0, 1) * 255).numpy().astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
            out.append(Image.fromarray(arr))
        return out

    def extract_features(self, images: torch.Tensor, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Returns {'cls': (N, 512)} CLS token embeddings."""
        all_feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="CLIP features"):
                pils = self._pil_batch(images[i:i + batch_size])
                inputs = self.processor(images=pils, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                vision_outputs = self.model.vision_model(**inputs)
                cls = vision_outputs.pooler_output  # (B, 512)
                all_feats.append(cls.float().cpu())
        return {'cls': torch.cat(all_feats, dim=0)}

    def estimate_likelihood(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        feats = self.extract_features(images, batch_size)['cls']
        return feats.norm(dim=1).numpy()

    def analyze_psychometric_set(self, data_dir: str = "data/stimuli/",
                                  alphas: List[float] = None,
                                  max_stimuli: Optional[int] = None) -> Dict:
        results = super().analyze_psychometric_set(data_dir, alphas, max_stimuli)
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        data_dir = Path(data_dir)
        all_feats, all_labels = [], []
        for alpha in alphas:
            fp = resolve_informative_file(data_dir, alpha)
            if fp is not None:
                stimuli = self.load_stimuli(str(fp), max_stimuli=max_stimuli)
                feats = self.extract_features(stimuli)['cls'].numpy()
                all_feats.append(feats)
                all_labels.extend([alpha] * len(feats))
        if all_feats:
            results['features'] = {
                'cls': np.vstack(all_feats),
                'labels': np.array(all_labels),
            }
        return results


class MAEAnalyzer(GenerativeModelAnalyzer):
    """
    Analyzer using Masked Autoencoder (ViT-MAE-Base).

    Likelihood proxy: negative reconstruction MSE (lower error = image more predictable).
    Features: CLS token from the MAE encoder (before masking).
    """

    def __init__(self, model_name="facebook/vit-mae-base", device=None):
        super().__init__(device)
        logger.info(f"Loading MAE: {model_name}")
        from transformers import ViTMAEForPreTraining, AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEForPreTraining.from_pretrained(model_name).to(self.device).eval()
        logger.info("  MAE encoder+decoder ready")

    def _pil_batch(self, images: torch.Tensor):
        from PIL import Image
        out = []
        for img in images:
            arr = (img.clamp(0, 1) * 255).numpy().astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
            out.append(Image.fromarray(arr))
        return out

    def extract_features(self, images: torch.Tensor, batch_size: int = 16) -> Dict[str, torch.Tensor]:
        """
        Returns {'cls': (N, 768)} — CLS token from the encoder on unmasked input.
        Uses noise=None to disable random masking for deterministic feature extraction.
        """
        all_feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="MAE features"):
                pils = self._pil_batch(images[i:i + batch_size])
                inputs = self.processor(images=pils, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Encode without masking: pass noise=None to encoder
                enc_out = self.model.vit(**inputs, noise=None)
                cls = enc_out.last_hidden_state[:, 0, :]  # CLS token, (B, 768)
                all_feats.append(cls.float().cpu())
        return {'cls': torch.cat(all_feats, dim=0)}

    def estimate_likelihood(self, images: torch.Tensor, batch_size: int = 16) -> np.ndarray:
        """Negative reconstruction MSE from masked autoencoding."""
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="MAE likelihood"):
                pils = self._pil_batch(images[i:i + batch_size])
                inputs = self.processor(images=pils, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # outputs.loss is mean per-image reconstruction MSE
                # Re-derive per-image loss from logits
                logits = outputs.logits  # (B, num_patches, patch_size**2 * 3)
                target = self.model.patchify(inputs['pixel_values'])
                # Compute per-image MSE over masked patches
                mask = outputs.mask  # (B, num_patches), 1=masked
                per_patch = ((logits - target) ** 2).mean(dim=-1)  # (B, num_patches)
                per_image = (per_patch * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                scores.append((-per_image).float().cpu().numpy())
        return np.concatenate(scores)

    def analyze_psychometric_set(self, data_dir: str = "data/stimuli/",
                                  alphas: List[float] = None,
                                  max_stimuli: Optional[int] = None) -> Dict:
        results = super().analyze_psychometric_set(data_dir, alphas, max_stimuli)
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        data_dir = Path(data_dir)
        all_feats, all_labels = [], []
        for alpha in alphas:
            fp = resolve_informative_file(data_dir, alpha)
            if fp is not None:
                stimuli = self.load_stimuli(str(fp), max_stimuli=max_stimuli)
                feats = self.extract_features(stimuli)['cls'].numpy()
                all_feats.append(feats)
                all_labels.extend([alpha] * len(feats))
        if all_feats:
            results['features'] = {
                'cls': np.vstack(all_feats),
                'labels': np.array(all_labels),
            }
        return results


class MultiArchitectureAnalyzer:
    """Unified framework for analyzing multiple architectures."""
    
    def __init__(self):
        self.analyzers: Dict[str, GenerativeModelAnalyzer] = {}
    
    def register_analyzer(self, name: str, analyzer: GenerativeModelAnalyzer):
        """Register a model analyzer."""
        self.analyzers[name] = analyzer
        logger.info(f"Registered analyzer: {name}")
    
    def analyze_all(self, data_dir: str = "data/stimuli/", max_stimuli: Optional[int] = None) -> Dict:
        """Run analysis on all registered analyzers."""
        results = {}
        
        for name, analyzer in self.analyzers.items():
            logger.info(f"\n=== Analyzing {name} ===")
            try:
                results[name] = analyzer.analyze_psychometric_set(data_dir=data_dir, max_stimuli=max_stimuli)
            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
                results[name] = None
        
        return results
    
    def save_results(self, results: Dict, output_path: str = "results/multi_architecture_results.json"):
        """Save results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        def _make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            return obj
        
        output = _make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def plot_comparison(self, results: Dict, output_path: str = "results/architecture_comparison.png"):
        """Plot comparison across architectures, with layer-wise Fisher panel for ResNet."""
        import matplotlib.pyplot as plt

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        has_layer_fisher = any(
            r is not None and 'layer_fisher' in r for r in results.values()
        )
        n_panels = 2 if has_layer_fisher else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        # Panel 1: likelihood / feature-norm vs alpha
        ax = axes[0]
        for arch_name, result in results.items():
            if result is None or 'informative' not in result:
                continue
            info = result['informative']
            ax.plot(info['alphas'], info['mean'], 'o-', label=arch_name, linewidth=2, markersize=6)

        ax.set_xlabel('Alpha (4th-order correlation strength)')
        ax.set_ylabel('Mean Score (likelihood / feature norm)')
        ax.set_title('Multi-Architecture Sensitivity to Alpha')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: ResNet layer-wise Fisher Information
        if has_layer_fisher:
            ax2 = axes[1]
            for arch_name, result in results.items():
                if result is None or 'layer_fisher' not in result:
                    continue
                lf = result['layer_fisher']
                x = range(len(lf['layer_names']))
                ax2.plot(x, lf['fisher_values'], 'o-', label=arch_name, linewidth=2, markersize=6)
                ax2.set_xticks(list(x))
                ax2.set_xticklabels(lf['layer_names'], rotation=30, ha='right')
            ax2.set_ylabel('Fisher Information')
            ax2.set_title('Layer-wise Sensitivity to Alpha (ResNet50)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved plot to {output_path}")
        plt.close()


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-architecture analysis")
    parser.add_argument("--archs", nargs="+",
                       default=["diffusion", "resnet", "autoregressive"],
                       help="Architectures to analyze: diffusion resnet autoregressive sd_unet clip mae")
    parser.add_argument("--data-dir", default="data/stimuli/", help="Stimulus directory")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--max-stimuli", type=int, default=None,
                       help="Maximum number of stimuli to analyze per condition")
    args = parser.parse_args()
    
    # Create framework
    framework = MultiArchitectureAnalyzer()
    
    # Register analyzers
    if "diffusion" in args.archs:
        try:
            framework.register_analyzer("Diffusion (DDPM)", 
                                       DiffusionModelAnalyzer("google/ddpm-cifar10-32"))
        except Exception as e:
            logger.warning(f"Could not load diffusion model: {e}")
    
    if "resnet" in args.archs:
        try:
            framework.register_analyzer("ResNet50 (ImageNet, Fisher)",
                                       ResNetFisherAnalyzer("resnet50"))
        except Exception as e:
            logger.warning(f"Could not load ResNet50: {e}")
    
    if "autoregressive" in args.archs:
        try:
            ar_analyzer = AutoregressiveAnalyzer()
            ar_label = "Autoregressive (ImageGPT)" if getattr(ar_analyzer, "is_imagegpt", False) else "Autoregressive (ViT-fallback)"
            framework.register_analyzer(ar_label, ar_analyzer)
        except Exception as e:
            logger.warning(f"Could not load autoregressive model: {e}")

    if "sd_unet" in args.archs:
        try:
            framework.register_analyzer("Diffusion (SD UNet)",
                                        SDUNetAnalyzer(probe_timestep=500))
        except Exception as e:
            logger.warning(f"Could not load SD UNet: {e}")

    if "clip" in args.archs:
        try:
            framework.register_analyzer("Contrastive (CLIP)", CLIPAnalyzer())
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}")

    if "mae" in args.archs:
        try:
            framework.register_analyzer("Reconstructive (MAE)", MAEAnalyzer())
        except Exception as e:
            logger.warning(f"Could not load MAE: {e}")

    # Run analysis
    results = framework.analyze_all(data_dir=args.data_dir, max_stimuli=args.max_stimuli)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    framework.save_results(results, str(output_dir / "multi_architecture_results.json"))
    framework.plot_comparison(results, str(output_dir / "architecture_comparison.png"))
    
    logger.info("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
