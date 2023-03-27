# AWS Comprehend 自定义标签

原作者Keith，原文链接LINK: https://keith-aws.notion.site/Comprehend-8f21d864312c4d74931d7c0ac04554d5



# 一、背景

实验将从数据注释开始完成步骤（无需使用SageMaker GT即可轻松CSV）.实验室应该有一个已经注释的数据集，并准备用于训练分类器。

通过本次实现将:

  ( 1 )了解训练模型，了解训练metrica和confusion matrix，并在需要时利用它们来提高准确性。

  ( 2 )了解如何在同步和异步模式下使用训练有素的模型。如需要同步，将使用自动缩放和调度选项来管理端点。

```python
The Red Sox never said their first offer to Pedro Martinez was their last. And while one prominent member of the Yankees after another wooed the three-time Cy Young Award winner in recent days, the Sox sweetened their offer to Martinez, though not enough to reach an agreement with the prized free agent.
====
红袜队从未说过他们对佩德罗·马丁内斯的首次报价是最后一次。虽然最近几天，洋基队的一位著名成员接一个地吸引了这位三届赛扬奖得主，但红袜队对马丁内斯的提议更加甜蜜，尽管还不足以与这位珍贵的自由球员达成协议。
《SPORT》
```

# 二、数据准备

## 1. 安装 SageMarker NoteBook

▶️【注释】按照启动项的步骤一步步的进行选择,即可以创建出来SageMaker笔记本实例. 使用体验等同于Jupter NoteBook

![image-20230327142921918](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327142921918.png)

## 2. 整理数据

> 将AG News数据集转换为可由Comprehend用于自定义分类的格式。
> 

### 2.1 基础环境构建

```python
[1]!pip install --upgrade  s3fs pandas  tqdm

[2]import pandas as pd
   import tqdm
   import boto3
   region_name='ap-southeast-1'
   import matplotlib

[3]! wget https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz
   ! tar xvzf ag_news_csv.tgz
```

### 2.2添加File 列定义

![image-20230327144752978](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327144752978.png)

▶️【注释】使用 Pandas 库的 read_csv() 函数从名为 "ag_news_csv/train.csv" 的 CSV 文件中读取数据，并将其存储在名为 train 的 Pandas DataFrame 中。

                 数据集包含三列：类别（category）、标题（title）和文本（text），并且使用 names 参数指定列名，以便在 DataFrame 中进行访问。

```python
[1] train=pd.read_csv("ag_news_csv/train.csv", names=['category','title','text'])

[2] train
```

![image-20230327144854066](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327144854066.png)

### 2.3 数据Limit并进行Rename

▶️【注释】为了将训练时间缩短到合理的时间，我们将数据限制在1000行。

- 为了容易观察，将把标签从数字改为字符串。数据集提供商告诉我们class.txt文件中的数据是什么样子的.⇒trainstr 文件

```python
[1] train = train.sample(axis='index',n=1000,random_state=100)
[2] labeldict={'1': 'WORLD', '2' :  'SPORTS', '3' : 'BUSINESS', '4': 'SCI_TECH'}
    trainstr=train.astype(str)
    trainstr['label']=trainstr['category'].replace(labeldict)
```

![image-20230327144908125](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327144908125.png)

### 2.4 删除Text内容

▶️【注释】将标题和文本放在一列中，用于我们的培训。现在只写出我们的标签和文本，因为这是理解所期望的输入。

```python
[1] dfout=trainstr[["label", 'text']]

[2] dfout

[3] dfout['label'].value_counts()
```

![image-20230327144922164](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327144922164.png)

## 3. 标准资料上传S3

▶️【注释】将标题和文本放在一列中，但通常最佳做法是开始给文本分类器“所有”相关数据。现在只写出我们的标签和文本，因为这是理解所期望的输入。

```python
[1]s3 = boto3.resource('s3')
   s3_client = boto3.client('s3')
   file_name="s3://" + "keith-ai-workshop" + "/custom_news_classification.csv"

[2] print(file_name)
```

![image-20230327144937284](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327144937284.png)



# 三、自定义Comprehend 模型输出

## 1. 训练Muti- Model模型

### 1.1 创建comprehend模型描述

▶️【注释】按下“创建新模型”按钮. 将模型命名为“新闻” 将版本命名为“v1”

![image-20230327144957687](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327144957687.png)

### 1.2 选择Muti- Model分类

- 在“数据规格”下选择“使用多标签模式”
- 在“S3上的训练数据位置”下，从您复制的笔记本中粘贴s3位置
- 在输出位置下，粘贴存储桶名称。一些额外的指标将放在这里。例如，s3://keith-ai-workshop/comprehend/Output/

![image-20230327145012960](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145012960.png)

### 1.3 创建Output与IAM权限

▶️【注释】在IAM角色下：

a 点击“创建IAM角色”

b.在访问权限下，选择“任何S3桶”

c.在名称后缀下，键入“ComprehendLabs”d.按“创建”按钮。

![image-20230327145027806](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145027806.png)

## 2.  观察自定义分类 Performance

[自定义分类器指标 - Amazon Comprehend](https://docs.aws.amazon.com/zh_cn/comprehend/latest/dg/cer-doc-class.html)

### **(一) 精确度 Accuracy VS 召回 Recall**

在AWS Comprehend中，精度和召回是两种不同的评估指标，用于评估情感分析、实体识别和主题建模等自然语言处理任务的性能。

精度（Precision）是指分类器正确预测为正例的样本数占所有预测为正例的样本数的比例。简单地说，精度是指分类器正确预测为目标类别的能力。

召回（Recall）是指分类器正确预测为正例的样本数占所有实际为正例的样本数的比例。简单地说，召回是指分类器能够正确找到所有目标类别的能力。

> 因此，精度和召回都是衡量模型性能的重要指标。通常情况下，精度和召回是相互矛盾的，提高精度可能会降低召回，反之亦然。在实际应用中，需要根据任务的具体要求来平衡精度和召回的取值。
> 

### **(二) 召回（Recall）Example**

当涉及到二分类问题时，召回（Recall）是指正确识别出所有正例的比例。让我们通过一个简单的二分类例子来理解召回：

假设你是一家医院的医生，你需要建立一个模型来识别患有癌症的患者。你有一个数据集，其中有100个样本，其中10个是患有癌症的患者，另外90个则是健康的患者。

你使用机器学习算法来构建一个分类器，该分类器能够根据某些特征将患者分为患有癌症的患者和健康的患者。在这个问题中，正例是患有癌症的患者，负例则是健康的患者。

现在，你的分类器给出了以下结果：

- 正例：预测有癌症的患者数为7，实际有癌症的患者数为10
- 负例：预测健康的患者数为83，实际健康的患者数为90

在这个情况下，召回是指正确识别出所有实际有癌症患者的能力。因此，召回率为7/10，即70%。这意味着你的分类器只能正确识别出实际患有癌症的患者的70%。因此，在这个例子中，召回率较低，需要对模型进行改进来提高识别患有癌症的能力。

### **(三) F1 得分（宏 F1 分数）**

在AWS Comprehend中，F1得分是一种常用的评估文本分类模型性能的指标，它是基于模型的精度（Precision）和召回率（Recall）的综合评估指标。F1得分的取值范围是0到1，值越接近1表示模型的性能越好。

                  `F1 = 2 * (precision * recall) / (precision + recall)`

> 在二元分类（Binary Classification）中，模型的预测结果可以分为四类：True Positive（真正例）、False Positive（假正例）、True Negative（真负例）和False Negative（假负例）。其中，True Positive指的是模型正确预测为正例的样本数量，False Positive指的是模型错误地将负例样本预测为正例的数量，True Negative是模型正确地预测为负例的样本数量，False Negative是模型错误地将正例样本预测为负例的数量。
> 

![image-20230327145040237](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145040237.png)

## 3. 创建自定义分类端点( Endpoint -实时)

▶️【注释】在您的模型版本中，单击“端点”选项卡，然后单击“创建端点”。这是您将在实时推理应用程序中使用的内容。称其为“新闻端点”，并给它一个“推理单元”。推理单位是量表的度量。每个单元表示每秒100个字符的吞吐量，每秒最多2个文档。每个端点最多可以分配10个推理单元。

![image-20230327145105673](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145105673.png)



### 3.1 控制台验证

![image-20230327145120655](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145120655.png)

### 3.2  使用Python验证

[classify_document — Boto3 Docs 1.26.84 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehend/client/classify_document.html)

```python
import boto3
comprehend = boto3.client(service_name='comprehend', region_name='ap-southeast-1')
text = "As Florida State quarterback Chris Rix returns to health, Coach Bobby Bowden will again start sophomore Wyatt Sexton in Saturday night's game against sixth-ranked Virginial."
custom_classifier_arn = 'arn:aws:comprehend:ap-southeast-1:473479646495:document-classifier-endpoint/News-endpoint'
response = comprehend.classify_document( Text=text, EndpointArn=custom_classifier_arn)
print(response)
```

![image-20230327145134797](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145134797.png)

## 4. 创建自定义分类端点( Job -异步)

### 4.1 数据准备

▶️【注释】我这里从中筛选了几行数据,构建News_Inference.csv的数据表, 并上传至S3

![image-20230327145155150](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145155150.png)

### 4.2 数据验证

![image-20230327145219412](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145219412.png)



![image-20230327145231619](https://raw.githubusercontent.com/liangyimingcom/storage/master/PicGo/image-20230327145231619.png)

```python
{"File": "custom_news_Inference.csv", "Line": "0", "Labels": [{"Name": "WORLD", "Score": 0.9974}, {"Name": "SPORTS", "Score": 0.0025}, {"Name": "BUSINESS", "Score": 0.0024}]}
{"File": "custom_news_Inference.csv", "Line": "1", "Labels": [{"Name": "BUSINESS", "Score": 0.9949}, {"Name": "WORLD", "Score": 0.006}, {"Name": "SPORTS", "Score": 0.0022}]}
{"File": "custom_news_Inference.csv", "Line": "2", "Labels": [{"Name": "SPORTS", "Score": 0.9977}, {"Name": "WORLD", "Score": 0.0026}, {"Name": "SCI_TECH", "Score": 0.0021}]}
{"File": "custom_news_Inference.csv", "Line": "3", "Labels": [{"Name": "SPORTS", "Score": 0.9978}, {"Name": "WORLD", "Score": 0.0031}, {"Name": "SCI_TECH", "Score": 0.0019}]}
{"File": "custom_news_Inference.csv", "Line": "4", "Labels": [{"Name": "WORLD", "Score": 0.9975}, {"Name": "SPORTS", "Score": 0.0031}, {"Name": "SCI_TECH", "Score": 0.0022}]}
{"File": "custom_news_Inference.csv", "Line": "5", "Labels": [{"Name": "BUSINESS", "Score": 0.9906}, {"Name": "WORLD", "Score": 0.0057}, {"Name": "SPORTS", "Score": 0.0019}]}
```

# 四、总结

这些指标可让您深入了解自定义分类器在分类作业期间的表现。如果指标较低，则分类模型很可能对您的用例无效。如果发生这种情况，你有几个选项来提高你的分类器性能。

- 在训练数据中，提供更具体的数据，以便轻松区分类别。例如，提供能够以唯一的单词/句子最好地表示标签的文档。
- 在训练数据中为代表性不足的标签添加更多数据。
- 尽量减少类别中的偏差。如果数据中最大的标签大于最小标签中文档的 10 倍，请尝试增加最小标签中的文档数量，并确保将高度代表性和代表性最少的类之间的偏斜率降低到至少 10:1。你也可以尝试从高度代表的类中删除一些文档。
