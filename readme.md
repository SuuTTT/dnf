goal: a script can play dnf automatically

features:
1. to log the result of game mateiral


# 开发日志

## 1024 start detect_material.py with feature 1


`x, y, w, h = 1670, 900, 700, 400`

challenge:
- 交易碳，角色碳，帐绑碳 过于相似，难以区分
    - 会和球形宠物装备混淆
    - threshold 设置为0.4一下才能监测到，但是不准。
- OCR数字区域太小，为包括数字，还未调参


## 1025 init git, 
new idea:
1.枚举蓝色边框 grids（可以不用识别，因为位置是固定的）
2. for grid in grids : 
    归一化
    for icons:
        if match:
            ocr the number
