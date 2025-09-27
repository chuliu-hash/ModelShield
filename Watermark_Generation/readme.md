# Watermark Generation 

generating watermarked responses using OpenAI's language model (you can also use it to other LMaaS). The generated watermarked text includes special watermark words embedded in the answers to provided questions.


## Input and Output

### Input
- **Questions File**: A file containing a list of questions in JSON format, with each question having an `id` and `instruction`.
- **API Key**: OpenAI API key(s) for generating responses.

### Output
- A JSON file storing results for each processed question:
  - Question ID and content.
  - Generated text with embedded watermark words.
  - Original question and model prediction.

## Usage
```bash
python watermark_generation.py


使用OpenAI的语言模型生成带有水印的响应（也可用于其他语言模型即服务）。生成的水印文本会在回答中嵌入特殊的水印词汇，以响应提供的提问。

输入与输出
输入
问题文件：一个包含问题列表的JSON格式文件，每个问题包含id和instruction字段
API密钥：用于生成响应的OpenAI API密钥（可支持多个密钥）
输出
存储每个问题处理结果的JSON文件：
问题ID和内容
嵌有水印词汇的生成文本
原始问题及模型预测结果