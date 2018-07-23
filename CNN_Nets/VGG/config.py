layer_type = [
    'c_p',
    'c_p',
    'c_p',
    'c_p',
    'c_p',
    'fc',
    'fc',
    'fc'
]

layer_param = [
    [   #parameters for conv_block
        [
            [3, 1, "SAME", 64],#filter_Size, stride, padding, channel
            [3, 1, "SAME", 64],
        ],
        #parameters for max_pooling
        [2, 2]#pooling_size, stride
    ],
    [
        [
            [3, 1, "SAME", 128],
            [3, 1, "SAME", 128],
        ],
        [2, 2]
    ],
    [
        [
            [3, 1, "SAME", 256],
            [3, 1, "SAME", 256],
            [3, 1, "SAME", 256],
        ],
        [2, 2]
    ],
    [
        [
            [3, 1, "SAME", 512],
            [3, 1, "SAME", 512],
            [3, 1, "SAME", 512],
        ],
        [2, 2]
    ],
    [
        [
            [3, 1, "SAME", 512],
            [3, 1, "SAME", 512],
            [3, 1, "SAME", 512],
        ],
        [2, 2]
    ],
    [
        [],#parameters for FC layer
    ],
    [
        [],
    ],
    [
        [],
    ]
]