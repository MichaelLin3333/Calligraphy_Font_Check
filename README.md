# 汉字书法识别器

An classification on five different Chinese Calligraphy font as well as around 1.7k Chinese characters based on https://github.com/jfxia/shufa.git

![](https://github.com/MichaelLin3333/Calligraphy_Font_Check/blob/main/assets/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-02-03%20083032.png)

run
```bash
python scraper.py
```
for running data crawling from https://sf.zdic.net/

run
```bash
bash train.sh
```
for running data train of a ResNet-50 model with two classification heads

run
```bash
python main.py
```
to start the web implemenetation of the model for showcase.


Due to timely constraints, the project resulted in model heavily overfitting to the crawled dataset.
