
<h1 align="center">Решение 2-го места для соревнования <a href="https://hacks-ai.ru/championships/758453">Всероссийский чемпионат, кейс РЖД</a> 

## Для воспроизведения результатов выполните следующие действия:

#### Сгенерируем списки тёмных и светлых изображений:
```
python split_dark_light.py --dataset-path <путь к папке "images" в датасете>
```

#### Запускаем обучение для каждой модели
```
python main.py --cfg configs/segformer_1024_b4_g16_adamW_cosine.yml
python main.py --cfg configs/segformer_1024_b4_adamW_cosine.yml
python main.py --cfg configs/segformer_864_b4_8_cosine_light.yml
python main.py --cfg configs/segformer_864_b4_8_cosine_dark.yml
```

#### Генерируем предикты

```
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_light/segformer_864_b4_8_cosine_light.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_light/segformer_864_b4_8_cosine_light.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_light/segformer_864_b4_8_cosine_light.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.651

python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/segformer_864_b4_8_cosine_dark.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/segformer_864_b4_8_cosine_dark.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_864_b4_8_cosine_dark/segformer_864_b4_8_cosine_dark.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.651



python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/segformer_1024_b4_g16_adamW_cosine.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou>
python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/segformer_1024_b4_g16_adamW_cosine.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.741
python generate_predictions.py --cfg experiments/segformer_1024_b4_g16_adamW_cosine/segformer_1024_b4_g16_adamW_cosine.yml  --checkpoint-path <Путь до чекпоинта с наивысшим miou> --scale 0.635
```

#### Обьъедените папки predictions/segformer_864_b4_8_cosine_dark и predictions/segformer_864_b4_8_cosine_light
#### Запускаем скрипт, который усреднит все предикты и сгенерирует итоговую маску
```
python create_submission_ensemble.py
```
