from torch import nn
import argparse
import torch
from torch.nn import Transformer

args = argparse.ArgumentParser() # 인자값 받을 수 있는 인스턴스 생성
args.add_argument("--dim_model", default=512, type = int, help='size')
args.add_argument("--num_heads", default=8, type = int, help='size')
args.add_argument("--num_encoder_layers", default=6, type = int, help='size')
args.add_argument("--num_decoder_layers", default=6, type = int, help='size')
args.add_argument("--dropout", default=0.1, type = float, help='size')
args = args.parse_args() #내용 저장
"""
 num_tokens, dim_model, num_heads, num_encoder_layers, 
 num_decoder_layers, dropout_p, 
 """

class Custom_Model(nn.Module) :
    def __init__(self,args):
        super().__init__()

        self.transformer = Transformer(
            d_model=args.dim_model, # Default = 512
            nhead=args.num_heads, #Default = 8
            num_encoder_layers=args.num_encoder_layers, #Default = 6
            num_decoder_layers=args.num_decoder_layers, # Default = 6
            dropout=args.dropout #Default = 0.1
        )

        self.classification = nn.Linear(in_features= 512,
                                        out_features=10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        output = self.transformer(x,y)
        #print(output)
        print(output.shape)
        output = self.classification(output)
        #print(output)
        print(output.shape)
        output = self.softmax(output)

        print(output.shape)
        return output
model = Custom_Model(args)

src = torch.rand(1,32,512)

target = torch.rand(2,32,512)

output = model(src,target)
#print(output.shape)
print(output)

