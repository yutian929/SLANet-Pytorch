import torch
import paddle
import numpy as np
from PaddleCSPPAN import CSPPAN as PaddleCSPPAN
from TorchCSPPAN import CSPPAN as TorchCSPPAN

def test_csppan():
    # Define the input and output channel sizes. Adjust these according to your specific model configuration.
    in_channels = [64,128,256,512]
    out_channel = 96

    # Initialize both models
    paddle_model = PaddleCSPPAN(in_channels=in_channels, out_channels=out_channel)
    torch_model = TorchCSPPAN(in_channels=in_channels, out_channels=out_channel)

    # Create a test input tensor
    paddle_input = [
        paddle.ones((1, 116, 224, 224)),
        paddle.ones((1, 232, 112, 112)),
        paddle.ones((1, 464, 56, 56)),
    ]
    torch_input = [
        torch.ones((1, 116, 224, 224)),
        torch.ones((1, 232, 112, 112)),
        torch.ones((1, 464, 56, 56)),
    ]

    # Get the outputs
    try:
        paddle_out = paddle_model(paddle_input)
    except Exception as e:
        print(f"Paddle error: {e}")
        for i, data in enumerate(paddle_input):
            print(f"Paddle input {i} shape: {data.shape}")

    try:
        torch_out = torch_model(torch_input)
    except Exception as e:
        print(f"Torch error: {e}")
        for i, data in enumerate(torch_input):
            print(f"Torch input {i} shape: {data.shape}")

    # Compare each corresponding output shape
    assert len(paddle_out) == len(torch_out), f'len: paddle_out-{len(paddle_out)} torch_out-{len(torch_out)}.'
    for p_out, t_out in zip(paddle_out, torch_out):
        assert p_out.shape == list(t_out.shape), f"Output shapes do not match: Paddle - {p_out.shape}, Torch - {t_out.shape}"
        print(f"Output shapes match: Paddle - {p_out.shape}, Torch - {t_out.shape}")

    # # Compare each corresponding output
    # for p_out, t_out in zip(paddle_out, torch_out):
    #     # Convert paddle tensor to numpy and then to torch tensor for comparison
    #     p_out_torch = torch.from_numpy(p_out.numpy())
        
    #     # Ensure the outputs are close
    #     assert torch.allclose(p_out_torch, t_out, atol=1e-6), "Outputs are not equal!"

    #     print("Test passed: Both models produce the same output for the given input.")

if __name__ == '__main__':
    test_csppan()