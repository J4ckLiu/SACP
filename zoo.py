import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatAttn(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class SpatAttn_(nn.Module):
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn_, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.Sequential(nn.ReLU(),
                        nn.BatchNorm2d(in_dim))
        
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out
        return self.bn(out)

class SARes(nn.Module):
    def __init__(self, in_dim, ratio=8, resin=False):
        super(SARes, self).__init__()
        if resin:
            self.sa1 = SpatAttn(in_dim, ratio)
            self.sa2 = SpatAttn(in_dim, ratio)
        else:
            self.sa1 = SpatAttn_(in_dim, ratio)
            self.sa2 = SpatAttn_(in_dim, ratio)            
        
    def forward(self, x):
        identity = x 
        x = self.sa1(x)
        x = self.sa2(x)
        
        return F.relu(x + identity)

class SPC32(nn.Module):
    def __init__(self, msize=24, outplane=49, kernel_size=[7,1,1], stride=[1,1,1], padding=[3,0,0], spa_size=9,  bias=True):
        super(SPC32, self).__init__()                                    
        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        
        if identity is None:
            identity = x 
        n,c,h,w = identity.size()
        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2) 
        mask0 = torch.softmax(mask0.view(n,-1,h*w), -1)                    
        mask0 = mask0.view(n,-1,h,w)
        _,d,_,_ = mask0.size()
        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)

        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0)
        out = F.leaky_relu(out)
        out = out.sum(2)
        out = out #+ identity
        out0 = self.bn1(out.view(n,-1,h,w))
        
        mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)
        mask2 = torch.softmax(mask2.view(n,-1,h*w), -1)                    
        mask2 = mask2.view(n,-1,h,w)
        fk = torch.einsum('ndhw,nchw->ncd', mask2, x)
        
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2)
        out = F.leaky_relu(out)
        out = out.sum(2)
        out = out + identity
        out = self.bn2(out.view(n,-1,h,w))

        return out 

class SSNet_AEAE_IP(nn.Module):
    def __init__(self, num_classes=16, msize=16, inter_size=49):
        super(SSNet_AEAE_IP, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(200, inter_size, 1),
                                   nn.BatchNorm2d(inter_size),)
        self.layer2 = SARes(inter_size, ratio=8)
        self.layer3 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size,1,1],  padding=[0,0,0])        
        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)
        self.layer5 = SARes(msize, ratio=8)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize,1,1],  padding=[0,0,0]) 
        self.fc = nn.Linear(msize, num_classes)

    def forward(self, x):
        n,c,h,w = x.size()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        return x

class SSNet_AEAE_PU(nn.Module):
    def __init__(self, num_classes=9, msize=18, inter_size=49):
        super(SSNet_AEAE_PU, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(103, inter_size, 1, bias=False),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(inter_size),)
        self.layer2 = SARes(inter_size, ratio=8) 
        self.layer3 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size,1,1],  padding=[0,0,0])        
        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)
        self.layer5 = SARes(msize, ratio=8)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize,1,1],  padding=[0,0,0])
        self.fc = nn.Linear(msize, num_classes)

    def forward(self, x):
        n,c,h,w = x.size()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        return x 

class SSNet_AEAE_SA(nn.Module):
    def __init__(self, num_classes=16, msize=16, inter_size=49):
        super(SSNet_AEAE_SA, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(204, inter_size, 1),
                                   nn.BatchNorm2d(inter_size),)
        self.layer2 = SARes(inter_size, ratio=8) 
        self.layer3 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size,1,1],  padding=[0,0,0])        
        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)
        self.layer5 = SARes(msize, ratio=8)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize,1,1],  padding=[0,0,0]) 
        self.fc = nn.Linear(msize, num_classes)

    def forward(self, x):
        n,c,h,w = x.size()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        return x 

class HybridSN(nn.Module):
    def __init__(self, band, classes):
        super(HybridSN, self).__init__()
        self.name = 'HybridSN'
        self.conv1 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=1,
                    out_channels=8,
                    kernel_size=(7, 3, 3)),
                    nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=(5, 3, 3)),
                    nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=(3, 3, 3)),
                    nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=576,
                    out_channels=64,
                    kernel_size=(3, 3)),
                    nn.ReLU(inplace=True))
        self.dense1 = nn.Sequential(
                    nn.Linear(18496,256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        self.dense2 = nn.Sequential(
                    nn.Linear(256,128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        self.dense3 = nn.Sequential(
                    nn.Linear(128,classes)
                   )
        
    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),x.size(1)*x.size(2),x.size(3),x.size(4))
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

class cnn3d(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(cnn3d, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x

class cnn1d(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            nn.init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(cnn1d, self).__init__()
        if kernel_size is None:
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

