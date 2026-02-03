# 汉字书法识别器

An classification on five different Chinese Calligraphy font as well as around 1.7k Chinese characters based on https://github.com/jfxia/shufa.git

![]('/assets/屏幕截图 2026-02-03 083032.png')

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
