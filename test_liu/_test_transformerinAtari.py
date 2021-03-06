class Visiontransformer(BaseFeaturesExtractor):
    """ Vision Transformer

        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929

        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=512,img_size=84, patch_size=16, in_chans=4, embed_dim=512, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super(Visiontransformer,self).__init__(observation_space, features_dim)
        #self.features_dim = features_dim
        #self.features_dim = 512
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(th.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or th.nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = th.nn.Parameter(th.zeros(1, 1, embed_dim))
        self.dist_token = th.nn.Parameter(th.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = th.nn.Parameter(th.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = th.nn.Dropout(p=drop_rate)
        self.results = th.nn.Linear(self.embed_dim, self.features_dim)

        dpr = [x.item() for x in th.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = th.nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = th.nn.Sequential(OrderedDict([
                ('fc', th.nn.Linear(embed_dim, representation_size)),
                ('act', th.nn.Tanh())
            ]))
        else:
            self.pre_logits = th.nn.Identity()
        self.Linear = th.nn.Linear(self.embed_dim, self.num_features)
        # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #    self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = th.nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else th.nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = th.nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else th.nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x=x.transpose(1,2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        pdb.set_trace()
        if self.dist_token is None:
            x = th.cat((cls_token, x), dim=1)
        else:
            x = th.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        pdb.set_trace()
        x = self.forward_features(x)
        x = self.results(x)
        pdb.set_trace()
        """
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not th.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        """
        return x