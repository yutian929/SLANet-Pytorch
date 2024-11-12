from PaddlePPLCNet import PPLCNet as PaddlePPLCNet
from TorchPPLCNet import PPLCNet as TorchPPLCNet
import torch
import paddle

def test_pplcnet():
    paddle_model = PaddlePPLCNet()
    torch_model = TorchPPLCNet()

    input_data = torch.ones(1, 3, 488, 488)

    paddle_out = paddle_model(paddle.to_tensor(input_data.numpy()))
    torch_out = torch_model(input_data)

    # Compare each corresponding output shape
    assert len(paddle_out) == len(torch_out), f'len: paddle_out-{len(paddle_out)} torch_out-{len(torch_out)}.'
    for p_out, t_out in zip(paddle_out, torch_out):
        assert p_out.shape == list(t_out.shape), f"Output shapes do not match: Paddle - {p_out.shape}, Torch - {t_out.shape}"
        print(f"Output shapes match: Paddle - {p_out.shape}, Torch - {t_out.shape}")
    

if __name__ == '__main__':
    test_pplcnet()