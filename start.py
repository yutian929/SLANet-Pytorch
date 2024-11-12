import csv
import paddle
import pandas as pd
import torch
from lib.Arch.TorchBaseModel import BaseModel

def transpose_weights(paddle_weights, transpose_layers):
    """
    转置指定的线性层权重。

    Args:
        paddle_weights (dict): PaddlePaddle 模型的权重字典。
        transpose_layers (list): 需要转置的层名称列表。

    Returns:
        dict: 处理后的权重字典。
    """
    for name, param in paddle_weights.items():
        if name in transpose_layers:
            # 转置权重矩阵
            paddle_weights[name] = param.transpose(perm=[1, 0])
            # print(f"Transposed layer: {name}")
    return paddle_weights

def copy_weights_to_pytorch(paddle_weights, pytorch_model, layer_mapping=None):
    """
    将 PaddlePaddle 的权重复制到 PyTorch 模型中。

    Args:
        paddle_weights (dict): PaddlePaddle 模型的权重字典。
        pytorch_model (torch.nn.Module): PyTorch 模型实例。
        layer_mapping (dict, optional): 如果两个模型的层名称不同，可以提供一个映射字典。
    """
    pytorch_state_dict = pytorch_model.state_dict()

    for paddle_name, paddle_param in paddle_weights.items():
        # 如果提供了层名映射，则使用映射后的名称
        pytorch_name = layer_mapping.get(paddle_name, paddle_name) if layer_mapping else paddle_name

        if pytorch_name in pytorch_state_dict:
            # 将 PaddlePaddle 的参数转换为 NumPy，再转换为 PyTorch 的张量
            pytorch_param = torch.from_numpy(paddle_param.numpy())
            # 确保数据类型一致
            pytorch_param = pytorch_param.type_as(pytorch_state_dict[pytorch_name])
            # 复制参数
            pytorch_state_dict[pytorch_name].copy_(pytorch_param)
            # print(f"Copied layer: {pytorch_name}")
        else:
            print(f"Layer {pytorch_name} not found in PyTorch model.")

    # 加载更新后的 state_dict
    pytorch_model.load_state_dict(pytorch_state_dict)
    print("Successfully loaded all weights into the PyTorch model.")

def save_name_param_pairs_to_csv(weights, filename, ignore_layers=None):
    if ignore_layers is None:
        ignore_layers = []
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer name', 'Parameter shape'])
        for name, param in weights.items():
            is_ignore = False
            for ignore in ignore_layers:
                if ignore in name:
                    is_ignore = True
                    break
                # print(f"Layer name: {name}, Parameter shape: {param.shape}")
            if not is_ignore:
                writer.writerow([name, list(param.shape)])
    
    


if __name__ == '__main__':
    # 配置参数（直接修改my_build）
    CONFIG = {
        "model_type": "table",
        "algorithm": "SLANet",
        "Backbone": {},
        "Neck": {},
        "Head": {},
    }

    # 初始化 PyTorch 模型
    model = BaseModel(CONFIG)
    print("Successfully initialized the PyTorch model.")

    # 加载 PaddlePaddle 模型的权重
    paddle_weights = paddle.load("./weights/ch_PP-StructrureV2_SLANet_plus_trained.pdparams")
    print("Loaded PaddlePaddle weights.")

    # 转置前的模型参数，保存为csv
    pytorch_weights = model.state_dict()
    save_name_param_pairs_to_csv(pytorch_weights, "weights/pytorch_weights.csv", ignore_layers=['num_batches_tracked'])
    save_name_param_pairs_to_csv(paddle_weights, "weights/paddle_weights_before_transpose.csv")

    # 指定需要转置的层名称
    layers_to_transpose = [
        'head.structure_attention_cell.i2h.weight',
        'head.structure_attention_cell.h2h.weight',
        'head.structure_attention_cell.score.weight',
        'head.structure_generator.0.weight',
        'head.structure_generator.1.weight',
        'head.loc_generator.0.weight',
        'head.loc_generator.1.weight',
        ]

    # 转置指定的层
    paddle_weights = transpose_weights(paddle_weights, layers_to_transpose)

    # 转置后的模型参数，保存为csv
    save_name_param_pairs_to_csv(paddle_weights, "weights/paddle_weights_after_transpose.csv")

    df_paddle = pd.read_csv("weights/paddle_weights_after_transpose.csv", usecols=[0], header=None, encoding='utf-8')
    df_pytorch = pd.read_csv("weights/pytorch_weights.csv", usecols=[0], header=None, encoding='utf-8')
    # 假设有相同的行数
    assert len(df_paddle) == len(df_pytorch), "Dataframes have different number of rows."
    layer_name_mapping = pd.Series(df_pytorch.iloc[:,0].values, index=df_paddle.iloc[:,0].str.strip()).to_dict()
    # for key, value in layer_name_mapping.items():
    #     print(key, value)

    # 将权重复制到 PyTorch 模型
    copy_weights_to_pytorch(paddle_weights, model, layer_mapping=layer_name_mapping)

    # 保存转换后的 PyTorch 模型权重
    torch.save(model.state_dict(), 'weights/pytorch_model_transferred.pth')
    print("Saved the PyTorch model weights to weights/pytorch_model_transferred.pth")

    # 测试转换后的模型
    test_input = torch.randn(1, 3, 488, 488)
    output = model(test_input)
    # 模型预测两个值，一是structure_pobs，表格结构的html代码，二是loc_preds，回归单元格四个点坐标。
    print("Output shapes:")
    print(output[0].shape)
    print(output[1].shape)