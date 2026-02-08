import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch
import torch.nn as nn

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
class ViT(nn.Module):
    def __init__(self, *, image_size, param_dim, patch_size, dim, depth, heads, mlp_dim, channels = 2, dim_head = 32, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.channels = channels
        self.param_dim = param_dim
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches_per_channel = (image_height // patch_height) * (image_width // patch_width) 
        num_patches = num_patches_per_channel * channels
        patch_dim = patch_height * patch_width

        self.sim_rearrange = Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1 = patch_height, p2 = patch_width)
        self.sim_to_patch_embedding = nn.ModuleList([nn.Sequential(nn.LayerNorm(patch_dim),
                                                     nn.Linear(patch_height * patch_width, dim),
                                                     nn.LayerNorm(dim))
             for _ in range(channels)
        ])
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches_per_channel, dim))#nn.Parameter(
        #    build_2d_sincos_posemb(image_height // patch_height,image_width // patch_width, dim).flatten(2).permute(0,2,1), requires_grad=False)
        self.param_to_embedding = nn.Sequential(nn.Linear(param_dim,dim),
                                               nn.LayerNorm(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.to_latent = nn.Identity()

    def forward(self, sim, param):
        x = self.sim_rearrange(sim)
        patches_per_channel = x.shape[1]//self.channels
        out = []
        for i in range(self.channels):
            out.append(self.sim_to_patch_embedding[i](x[:,i*patches_per_channel:(i+1)*patches_per_channel]) + self.pos_embedding)
        out.append(self.param_to_embedding(param).unsqueeze(1))
        x = torch.cat(out, dim=1)
        x = self.dropout(x)

        x = self.transformer(x)
        return x
class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 2,
        decoder_heads = 8,
        decoder_dim_head = 16
    ):
        super().__init__()
        assert masking_ratio >= 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.channels = encoder.channels
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.sim_rearrange
        self.patch_to_emb = encoder.sim_to_patch_embedding
        self.param_to_emb = encoder.param_to_embedding
        pixel_values_per_patch = encoder.sim_to_patch_embedding[0][1].weight.shape[-1]
        self.encoder_pos_emb = self.encoder.pos_embedding
        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(self.channels, decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.ModuleList([nn.Linear(decoder_dim, pixel_values_per_patch) for _ in range(self.channels)])
        self.to_param = nn.Linear(decoder_dim, encoder.param_dim)
    def to_pos_indices(self, indices):
        sim_pos_size = self.encoder_pos_emb.shape[1]
        last_index = self.channels * sim_pos_size
        return indices % sim_pos_size
    def forward(self, sim, param, return_value='recon_loss'):
        device = sim.device

        # get patches

        patches = self.to_patch(sim)
        batch, num_sim_patches, *_ = patches.shape
        num_patches= num_sim_patches + 1
        # patch to encoder tokens and add positions
        patches_per_channel = num_sim_patches//self.channels
        tokens = []
        for i in range(self.channels):
            tokens.append(self.patch_to_emb[i](patches[:,i*patches_per_channel:(i+1)*patches_per_channel]) + self.encoder_pos_emb)
        tokens = torch.cat(tokens, dim=1)
        param_tokens = self.param_to_emb(param).unsqueeze(1)
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        masked_indices, unmasked_indices = [], []
        num_masked = int(self.masking_ratio * num_sim_patches)
        for i in range(self.channels):
            rand_indices = torch.rand(batch, num_sim_patches//self.channels, device = device).argsort(dim = -1) + num_sim_patches//self.channels*i
            masked_indices.append(rand_indices[:, :num_masked//self.channels])
            unmasked_indices.append(rand_indices[:, num_masked//self.channels:])
        masked_indices = torch.cat(masked_indices,dim=1)
        unmasked_indices = torch.cat(unmasked_indices,dim=1)
        masked_pos_indices, unmasked_pos_indices = self.to_pos_indices(masked_indices), self.to_pos_indices(unmasked_indices)
        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        
        masked_patches = patches[batch_range, masked_indices]
        unmasked_patches = patches[batch_range, unmasked_indices]
        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(torch.cat((tokens,param_tokens),dim=1))

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        param_decoder_tokens = decoder_tokens[:,-1]
        decoder_tokens = decoder_tokens[:,:-1]
        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_pos_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = torch.cat([repeat(self.mask_token[i], 'd -> b n d', b = batch, n = num_masked//self.channels) for i in range(self.channels)],dim=1)
       
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_pos_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        if return_value == 'mae_tokens':
            return {'mae_tokens': decoder_tokens,
                    'masked_indices': masked_indices,
                   'unmasked_indices': unmasked_indices}
        decoded_tokens = self.decoder(decoder_tokens)
        if return_value == 'unmasked_patches':
            num_patches_per_channel = self.encoder_pos_emb.shape[1]
            num_unmasked_indices_per_channel = unmasked_indices.shape[1]//self.channels
            pred_pixel_values = []
            for i in range(self.channels):
                pred_pixel_values.append(self.to_pixels[i](decoded_tokens[batch_range, unmasked_indices[:,num_unmasked_indices_per_channel*i:num_unmasked_indices_per_channel*(i+1)]]))
            pred_pixel_values = torch.cat(pred_pixel_values,dim=1)
            return {'pred_patches': pred_pixel_values,
                    'unmasked_indices': unmasked_indices,
                   'unmasked_patches': unmasked_patches}
        num_patches_per_channel = self.encoder_pos_emb.shape[1]
        num_masked_indices_per_channel = masked_indices.shape[1]//self.channels
        pred_pixel_values = []
        for i in range(self.channels):
            pred_pixel_values.append(self.to_pixels[i](decoded_tokens[batch_range, masked_indices[:,num_masked_indices_per_channel*i:num_masked_indices_per_channel*(i+1)]]))
        pred_pixel_values = torch.cat(pred_pixel_values,dim=1)
        # calculate reconstruction loss

        
        if return_value == 'recon_loss':
            # In training, compute and return the reconstruction loss.
            recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
            return recon_loss
        if return_value == 'masked_patches':
            # In evaluation mode, return the predicted pixel values for the masked patches.
            return {'pred_patches': pred_pixel_values,
                    'masked_indices': masked_indices,
                   'masked_patches': masked_patches}
    def reconstruct_sim(self, sim, pred_masked_patches, masked_indices):
        """
        Reconstruct the simulation image by replacing the patches at the masked indices 
        with the predicted masked patches.

        Args:
            sim (torch.Tensor): The original simulation image of shape (B, channels, H, W).
            pred_masked_patches (torch.Tensor): Predicted pixel values for masked patches,
                                                 shape (B, num_masked, patch_dim) where 
                                                 patch_dim = patch_height * patch_width.
            masked_indices (torch.Tensor): The masked patch indices of shape (B, num_masked).

        Returns:
            torch.Tensor: The reconstructed simulation image of shape (B, channels, H, W).
        """
        device = sim.device
        batch = sim.shape[0]

        # 1. Extract patches from the simulation image.
        #    Expected shape: (B, num_patches, patch_dim)
        patches = self.to_patch(sim)

        # 2. Replace the masked patches with the predicted ones.
        # Clone the patches to form a base for reconstruction.
        reconstructed_patches = patches.clone()
        batch_range = torch.arange(batch, device=device).unsqueeze(1)
        reconstructed_patches[batch_range, masked_indices] = pred_masked_patches

        # 3. Determine patch dimensions (assuming square patches).
        patch_dim = patches.shape[-1]
        patch_side = int(patch_dim ** 0.5)

        # 4. Compute the number of patches along height and width.
        #print(sim.shape)
        H, W = sim.shape[2], sim.shape[3]
        h_patches = H // patch_side
        w_patches = W // patch_side

        # 5. Rearrange the patches back into the image.
        reconstructed_sim = rearrange(
            reconstructed_patches,
            'b (c h w) (p1 p2) -> b c (h p1) (w p2)',
            c=self.channels, h=h_patches, w=w_patches, p1=patch_side, p2=patch_side
        )

        return reconstructed_sim
class Decoder(nn.Module):
    def __init__(self, *, num_patches, pixel_values_per_patch, decoder_dim, decoder_depth=2, decoder_heads=8,
                 decoder_dim_head=16, channels=2):
        """
        Args:
            num_patches (int): Total number of tokens (patches) in the sequence.
            decoder_dim (int): Dimension of the decoder embeddings.
            decoder_depth (int): Number of transformer layers in the decoder.
            decoder_heads (int): Number of attention heads.
            decoder_dim_head (int): Dimension per head in the decoder.
            channels (int): Number of channels (each with its own pixel prediction head).
            pixel_values_per_patch (int): Number of pixel values per patch.
        """
        super().__init__()
        self.num_patches = num_patches
        self.decoder_dim = decoder_dim
        self.channels = channels

        # Learnable mask token which will be repeated for every masked position.
        self.mask_token = nn.Parameter(torch.randn(self.channels, decoder_dim))
        # Positional embeddings for each patch position in the decoder.
        self.decoder_pos_emb = nn.Embedding(num_patches//channels, decoder_dim)

        # Transformer decoder that will process the complete token sequence.
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4  # Typically the MLP dimension is 4x the hidden size.
        )

        # For each channel, we use a linear layer to predict the pixel values of a patch.
        self.to_pixels = nn.ModuleList([nn.Linear(decoder_dim, pixel_values_per_patch) for _ in range(self.channels)])
        self.pos_emb_indices = torch.cat([torch.arange(self.num_patches//channels).unsqueeze(0) for _ in range(channels)],dim=1)
    def forward(self, mae_tokens):
        """
        Args:
            unmasked_tokens (torch.Tensor): Tensor of shape (B, num_unmasked, decoder_dim).
                These are the encoder-projected tokens for the unmasked patches.
            unmasked_indices (torch.Tensor): LongTensor of shape (B, num_unmasked) containing
                the indices (in the full token sequence) for unmasked patches.
            masked_indices (torch.Tensor): LongTensor of shape (B, num_masked) containing
                the indices (in the full token sequence) for masked patches.
                
        Returns:
            pred_pixel_values (torch.Tensor): Predicted pixel values for the masked patches,
                concatenated over channels. Its shape will be (B, num_masked, pixel_values_per_patch).
            decoded_tokens (torch.Tensor): The full token sequence after decoding (B, num_patches, decoder_dim).
        """
        batch = mae_tokens.shape[0]
        device = mae_tokens.device
        mask_tokens = repeat(self.mask_token[-1], 'd -> b n d', b = batch, n = self.num_patches)
        mask_tokens = mask_tokens + self.decoder_pos_emb(self.pos_emb_indices.to(device))
        batch_range = torch.arange(batch, device=device).unsqueeze(1)
        
        decoder_input = torch.cat([mask_tokens,mae_tokens], dim=1)
        # Pass the complete sequence through the decoder transformer.
        decoded_tokens = self.decoder(decoder_input)[:,:self.num_patches]
        num_patches_per_channel = self.num_patches//self.channels
        pred_pixel_values = []
        for i in range(self.channels):    
            tokens_channel = decoded_tokens[:, num_patches_per_channel * i: num_patches_per_channel * (i + 1)]
            pred_channel = self.to_pixels[i](tokens_channel)
            pred_pixel_values.append(pred_channel)
        
        # Concatenate predictions along the patch dimension.
        pred_pixel_values = torch.cat(pred_pixel_values, dim=1)
        
        return pred_pixel_values
