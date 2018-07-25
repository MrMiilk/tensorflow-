layer_types = [
    'c_p',
    'c_p',
    'c_p',
    'c_p',
    'c_p',
    'ft',
    'fc',
    'fc',
    'fc'
]

layer_params = [
    [   #parameters for conv_block
        [
            [[3, 3, 3, 64], [64], "relu", [1, 1, 1, 1], "SAME"],#filter, bias, nolinear_func, stride, padding
            [[3, 3, 64, 64], [64], "relu", [1, 1, 1, 1], "SAME"]
        ],
        #parameters for max_pooling
        [[2, 2], "MAX", "SAME", [2, 2]]#window_shape, pooling_type, padding, stride
    ],
    [
        [
            [[3, 3, 64, 128], [128], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 128, 128], [128], "relu", [1, 1, 1, 1], "SAME"],
        ],
        [[2, 2], "MAX", "SAME", [2, 2]]
    ],
    [
        [
            [[3, 3, 128, 256], [256], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 256, 256], [256], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 256, 256], [256], "relu", [1, 1, 1, 1], "SAME"],
        ],
        [[2, 2], "MAX", "SAME", [2, 2]]
    ],
    [
        [
            [[3, 3, 256, 512], [512], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 512, 512], [512], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 512, 512], [512], "relu", [1, 1, 1, 1], "SAME"],
        ],
        [[2, 2], "MAX", "SAME", [2, 2]]
    ],
    [
        [
            [[3, 3, 512, 512], [512], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 512, 512], [512], "relu", [1, 1, 1, 1], "SAME"],
            [[3, 3, 512, 512], [512], "relu", [1, 1, 1, 1], "SAME"],
        ],
        [[2, 2], "MAX", "SAME", [2, 2]]
    ],
    [#parameters for flatten layer
        4096,
    ],
    [#parameters for FC layer
        [4096, 4096], [4096], "relu",#W_shape, b_shape, nolinear_func
    ],
    [
        [4096, 4096], [4096], "relu"
    ],
    [
        [4096, 1000], [1000], ""
    ]
]