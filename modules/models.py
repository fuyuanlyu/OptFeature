import torch
from modules.layer import MultiLayerPerceptron, FactorizationMachine, FeaturesLinear, FeatureEmbedding
import modules.layer as layer


class BasicModel(torch.nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.latent_dim = args.dim
        self.feature_num = args.feature
        self.field_num = args.field
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
    
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """
        pass

    def reg(self):
        return 0.0
    
class LR(BasicModel):
    def __init__(self, args):
        super(LR, self).__init__(args)
        self.linear = FeaturesLinear(self.feature_num)

    def forward(self, x):
        logit = self.linear(x)
        return logit

class FM(BasicModel):
    def __init__(self, args):
        super(FM, self).__init__(args)
        self.fm = FactorizationMachine(reduce_sum=True)
    
    def forward(self, x):
        x_embedding = self.embedding(x)
        output_fm = self.fm(x_embedding)
        logit = output_fm
        return logit

class FNN(BasicModel):
    def __init__(self, args):
        super(FNN, self).__init__(args)
        mlp_dims = args.mlp_dims
        dropout = args.mlp_dropout
        use_bn = args.mlp_bn
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        logit = self.dnn(x_dnn)
        return logit

class DeepFM(FM):
    def __init__(self, args):
        super(DeepFM, self).__init__(args)
        embed_dims = args.mlp_dims
        dropout = args.mlp_dropout
        use_bn = args.mlp_bn
        self.dnn_dim = self.field_num*self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn + output_fm
        return logit

class DeepCrossNet(BasicModel):
    def __init__(self, args):
        super(DeepCrossNet, self).__init__(args)
        cross_num = 3
        mlp_dims = args.mlp_dims
        dropout = args.mlp_dropout
        use_bn = args.mlp_bn
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross,output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit

class InnerProductNet(BasicModel):
    def __init__(self, args):
        super(InnerProductNet, self).__init__(args)
        mlp_dims = args.mlp_dims
        dropout = args.mlp_dropout
        use_bn = args.mlp_bn
        self.dnn_dim = self.field_num * self.latent_dim + int(self.field_num * (self.field_num -1)/2)
        self.inner = layer.InnerProduct(self.field_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=True, dropout=dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.field_num*self.latent_dim)
        x_innerproduct = self.inner(x_embedding)
        x_dnn= torch.cat((x_dnn, x_innerproduct), 1)
        logit = self.dnn(x_dnn)
        return logit
