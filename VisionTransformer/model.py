
# Import modules
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from einops.layers.tensorflow import Rearrange  # pip install einops
import tensorflow.keras.layers as layers
from modelUtils import get_activation, gelu, model_sanity_check, get_model_memory_usage


class Residual(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x


class PreNorm(tf.keras.Model):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=0.5)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))


class FeedForward(tf.keras.Model):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation=get_activation(gelu)),
            layers.Dense(dim)
        ])

    def call(self, x):
        return self.net(x)


class Attention(tf.keras.Model):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** (-0.5)

        self.to_qkv = layers.Dense(dim*3, use_bias=False)
        self.to_out = layers.Dense(dim)

        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = tf.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return out


class Transformer(tf.keras.Model):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _  in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)


class ViT(tf.keras.Model):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size  == 0    # image dimension must be divisible by the patch size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim

        # Positional embedding
        self.pos_embedding = self.add_weight(
            "position_embeddings",
            shape=[num_patches + 1, dim],
            initializer = tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )

        # Classification token
        self.cls_token = self.add_weight(
            "cls_token",
            shape = [1, 1, dim],
            initializer = tf.keras.initializers.RandomNormal(),
            dtype = tf.float32
        )

        # Rearrange the patch vector
        self.rearrange = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size
        )

        # Convert each patch to a vector embedding
        self.patch_to_embedding = tf.keras.layers.Dense(dim)

        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = tf.identity

        # Multi-layer perceptron head
        self.mlp_head = tf.keras.Sequential(
            [
                layers.Dense(
                    mlp_dim,
                    activation=get_activation("gelu")
                ),
                layers.Dense(num_classes)
            ]
        )

    @tf.function
    def call(self, img):
        shapes = tf.shape(img)

        x = self.rearrange(img)
        x = self.patch_to_embedding(x)

        cls_tokens = tf.broadcast_to(self.cls_token, (shapes[0], 1,  self.dim))
        x = tf.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


model_config = {"image_size":2048,
                "patch_size":32,
                "num_classes":10,
                "dim":64,
                "depth":3,
                "heads":4,
                "mlp_dim":128}
model = ViT(**model_config)
model.build((1, 3, model_config["image_size"], model_config["image_size"]))
print(model.summary())

model_sanity_check(
    model,
    image_size=model_config["image_size"],
    n_classes=model_config["num_classes"]
)

model_memory = get_model_memory_usage(8, model, model_config)
