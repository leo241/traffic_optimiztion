# traffic_optimiztion
数据清洗预处理.ipynb：用于从源数据生成一份清洗数据和独热数据，您可以选择不运行这个文件，直接从网盘中下载全部三份数据。因为原始数据集较大，整个运行完成需要一小时。  
描述性分析.ipynb：直接查看或者运行描述性分析的全部可视化结果。  
建模分析.ipynb:直接查看或者运行建模部分，包括单类预测、逻辑回归、有序逻辑回归、K近邻、决策树、随机森林、LightGBM的训练、评估，您可以直接看到每种方法的准确率、AUC和困惑矩阵。整个运行需要一小时。  
验证集调优.py:直接运行部分验证集网格调参的过程。由于数据集的庞大，调参秉持着从少、从优的原则。  
deepfm_train.py:直接运行即可训练deepfm模型。过程中会打印、保存日志信息到data文件夹下，会将模型以验证集最优回滚的方式（总是只保留一个最优的模型）保存在data文件夹下。  
deepfm_eval.py:直接运行即可查看当前保存模型在测试集上的准确率、AUC、困惑矩阵。您可以自己训练，也可以下载网盘中我训练好的模型‘deepfm_best_0601.pth’放在data文件夹下。注意修改加载模型的名称。  
调配模拟.ipynb:直接查看或者运行5.1.2地图调配模拟的过程。期间需要加载训练好的lightgbm模型，您也可以从网盘中下载。代码中包含着优化问题+后处理，地图可视化、以及与随机对比效能。  
