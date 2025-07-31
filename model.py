#other module will be released soon

class Block(nn.Module):
    def __init__(self, dim, n_heads=16, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., cross=False):
        super().__init__()
        self.config = Config()
        self.cross = cross
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CausalSelfAttention(self.config)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        #self.mlp = WeightedPermuteMLP(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        if self.cross:
            x_ = [self.norm1(_x) for _x in x]
            # x_ = x
            out = x[2] + self.attn(x_)
            out = out + self.mlp(self.norm2(out))
            out = [x_[0], out, out]
        else:
            #print(f"x:{x.shape}")
            #print(f"self.attn(self.norm1(x):{self.attn(self.norm1(x)).shape}")
            out = x + self.attn(self.norm1(x))
            out = out + self.mlp(self.norm2(out))
            
        return out
    
#MLP
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.fusion = Block(dim=2048, n_heads=4)
        self.fc1 = nn.Linear(2048, 256)
        self.fch = nn.Linear(256, 128)
        self.fc2_mean = nn.Linear(128, 1)
        self.fc2_logvar = nn.Linear(128, 1)

    def encode(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fch(h0))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)#shape=(4,1024)
        z = self.reparametrization(mu, logvar)
        return z, mu, logvar

# LCMS

# model
