# Рекомендации фильмов

Компактный сервис на FastAPI: MF+BPR, neural two-tower, ALS и SANSA,
создание пользователей и сохранение оценок в SQLite. Зависимости управляются uv;
готовые модели и отчёт об обучении находятся в `artifacts/`.

## Запуск

Нужны [uv](https://docs.astral.sh/uv/getting-started/installation/) и Python 3.12.

```bash
uv sync --locked --no-dev
uv run --no-dev uvicorn movie_recs.api:app --host 127.0.0.1 --port 8000
```

[Swagger](http://localhost:8000/docs) позволяет отправлять все запросы из браузера.
Локальная база создаётся в `state/users.sqlite3`; путь меняется через `USER_DB`.

## Создать пользователя и получить рекомендации

Найти фильмы в обученном каталоге:

```bash
curl 'http://localhost:8000/movies?query=Matrix&limit=5'
```

Создать пользователя с начальными оценками:

```bash
curl -X POST 'http://localhost:8000/users' \
  -H 'Content-Type: application/json' \
  -d '{"ratings":[{"movie_id":1,"rating":5},{"movie_id":2,"rating":4.5}]}'
```

Ответ содержит новый `user_id` — например, 611. Подставьте ID из ответа в следующие запросы:

```bash
curl 'http://localhost:8000/recommendations/611?k=5&model=als'
curl 'http://localhost:8000/recommendations/611?k=5&model=sansa'
```

Добавить или заменить одну оценку:

```bash
curl -X PUT 'http://localhost:8000/users/611/ratings/2571' \
  -H 'Content-Type: application/json' -d '{"rating":5}'
```

Рекомендации учитывают изменение на следующем запросе. Оценки разрешены от 0.5 до 5
с шагом 0.5; положительными считаются оценки от 4. Фильмы с любыми оценками
исключаются из выдачи. Повторная отправка оценки заменяет предыдущую.
PUT работает и для существующих обученных пользователей MovieLens.

Можно создать пустой профиль: `POST /users` с телом `{}`. Пока положительных оценок нет,
выдаётся популярность, что явно отмечается полем `strategy: "popularity"`.
При изменении последней положительной оценки на низкую профиль также переходит к популярности.
ID новых пользователей начинаются после диапазона MovieLens 1–610 и не пересекаются с ним.

## API

| Запрос | Назначение |
|---|---|
| `POST /users` | Создать пользователя; начальные `ratings` необязательны |
| `PUT /users/{user_id}/ratings/{movie_id}` | Добавить или заменить оценку |
| `GET /movies?query=...&limit=20` | Найти фильмы по части названия |
| `GET /recommendations/{user_id}?k=10&model=als` | Получить рекомендации |
| `GET /models` | Метрики всех моделей и модель по умолчанию |
| `GET /health` | Проверить готовность сервиса |

`model` принимает `mf`, `neural`, `als`, `sansa`. Без него используется модель с лучшим
validation NDCG@10. `k` — от 1 до 100; при нехватке доступных фильмов ответ короче.
Каждый результат содержит `movie_id`, `title`, `genres`, `score`.
Score не является вероятностью или оценкой по пятибалльной шкале и несопоставим между моделями.

Поле `strategy` объясняет, откуда взят пользовательский профиль:

| Значение | Значение для запроса |
|---|---|
| `trained_user` | Сохранённый вектор обученного пользователя |
| `als_fold_in` | Точный шаг ALS для пользовательского вектора при фиксированных фильмах |
| `profile_fold_in` | Приближённая адаптация профиля в пространстве MF или neural |
| `history` | SANSA применила обученные веса к истории положительных оценок |
| `popularity` | Положительных оценок нет; выбранная модель указана в `model`, выдача — популярность |

Для нового профиля MF и neural не получают новый обученный ID embedding:
используется implicit least-squares fold-in по фиксированным выходным векторам фильмов.
Это приближение, отдельно обозначенное в ответе. Для ALS fold-in соответствует её
обучающей функции потерь. SANSA обрабатывает новый профиль напрямую.
Полное переобучение для добавления пользователя не требуется.

Пользователи MovieLens 214, 442, 508 не имели положительных train-оценок и не обучены;
для них возвращается 404. Остальные 607 поддерживаются. Неизвестные пользователи
и фильмы вне обученного каталога дают 404, некорректные параметры и оценки — 422.
Новые фильмы и глобальное обновление моделей через API не реализованы.

## Docker

Готовый образ содержит четыре обученные модели; обучение при запуске не требуется.
База профилей хранится в **именованном volume**, чтобы сохраняться после удаления контейнера.

```bash
docker build -t movie-recs:local .
docker run -d --rm --name movie-recs-demo \
  -p 127.0.0.1:8000:8000 \
  -v movie-recs-data:/app/state movie-recs:local
```

Остановить: `docker stop movie-recs-demo`. Для повторного запуска используйте ту же
команду `docker run` и тот же volume. Если контейнер с этим именем уже работает,
его нужно сначала остановить.

Готовый архив образа Linux ARM64 (Apple Silicon) лежит рядом с папкой исходников:

```bash
docker load -i ../movie-recs-linux-arm64.tar.gz
```

На x86-64 соберите образ из Dockerfile. Runtime работает от пользователя 10001,
имеет healthcheck и использует NumPy/SciPy для выдачи. PyTorch, implicit, SANSA,
SuiteSparse и компиляторы нужны только для обучения и в runtime не устанавливаются.

## Данные и модели

[MovieLens latest-small, версия 26.09.2018](https://grouplens.org/datasets/movielens/latest/):
100 836 оценок, 610 пользователей и 9 742 фильма. SHA-256 архива зафиксирован
в `movie_recs/data.py`. Обученный каталог включает 5 320 фильмов с положительными train-оценками.

| Модель | Реализация |
|---|---|
| MF+BPR | PyTorch, два ID embedding по 64 координаты, скалярное произведение |
| Neural two-tower | PyTorch, независимые MLP-башни; item ID embedding + multi-hot жанры |
| ALS | `implicit.cpu.als.AlternatingLeastSquares`, 64 фактора, confidence 20, regularization 0.1 |
| SANSA | Официальная `sansa==1.1.0`, CHOLMOD + UMR, l2 100, density 0.01 |

ALS оптимизирует implicit-feedback weighted least squares. SANSA — разреженная
линейная модель item-to-item на основе приближённой обратной матрицы, а не двухбашенная модель.
MF и neural обучаются через BPR с четырьмя негативами на положительную пару.
Негативный сэмплер исключает все train-оценки и не смотрит в отложенные данные.
Низкие оценки используются для исключения просмотренного, а не как явные отрицательные цели.
Popularity — число положительных train-оценок; используется как ориентир и при пустом профиле.

Источники алгоритмов: [ALS / implicit](https://benfred.github.io/implicit/api/models/cpu/als.html),
[официальная SANSA](https://github.com/glami/sansa),
[статья SANSA](https://doi.org/10.1145/3604915.3608827).

## Обучение

Все четыре модели можно обучить в Docker: target `trainer` устанавливает SuiteSparse,
необходимую официальной SANSA, и зависимости из `uv.lock`.

```bash
docker build --target trainer -t movie-recs-trainer:local .
mkdir -p artifacts data
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" -v "$PWD/data:/app/data" \
  movie-recs-trainer:local
```

Для обучения только новых моделей к команде `docker run` после имени образа добавьте:

```bash
python -m movie_recs.train --models als sansa
```

Без SuiteSparse локально можно обучить MF, neural и ALS:

```bash
uv sync --locked --group train
uv run --locked --group train python -m movie_recs.train --models mf neural als
```

Датасет скачивается и проверяется автоматически. Обучение идёт на CPU. Для MF/neural:
seed 42, до 40 эпох, patience 6, Adam с learning rate 0.003, batch 1024.
ALS обучается 30 итераций с выбором лучшей по validation. SANSA обучается одним запуском;
её конфигурация зафиксирована, подбора по test нет. Пользовательские оценки из SQLite
в этот offline-эксперимент не добавляются.

`--models` позволяет обновлять выбранные модели, сохраняя отчёты остальных в том же каталоге.
Обновлённые артефакты применяются после перезапуска API или пересборки Docker-образа.
Обучение запускайте отдельно от API. `--output-dir` задаёт каталог результатов;
API читает его через `ARTIFACT_DIR`.

## Протокол оценки

1. Оценки каждого пользователя сортируются по `(timestamp, movie_id)` и делятся 80/10/10.
2. Пользователи и кандидаты формируются только из положительных train-пар.
3. Все модели используют один train-снимок истории. Validation исключает все train-оценки,
   test — train+validation. В SANSA входом и для test остаётся train-история.
4. Поиск идёт по полному каталогу кандидатов. Recall@10 и NDCG@10 усредняются по
   пользователям хотя бы с одним доступным положительным отложенным фильмом.
5. Выбор эпохи и модели по умолчанию делается только по validation NDCG@10.

Это временное разбиение внутри пользователя, не общий глобальный cutoff.
Метрики относятся к тёплым пользователям и фильмам: validation — 551 пользователь,
test — 563. В test исключены 618 положительных оценок холодных фильмов и 3 оценки
холодных пользователей. Online fold-in новых профилей этими offline-метриками не измеряется.

<!-- RESULTS -->
| Модель | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| popularity | 0.0589 | 0.0603 | 0.0509 | 0.0516 |
| mf | 0.1003 | 0.0866 | 0.0794 | 0.0662 |
| neural | 0.0862 | 0.0768 | 0.0635 | 0.0592 |
| als | 0.1067 | 0.0850 | 0.0681 | 0.0570 |
| sansa | 0.1126 | 0.0938 | 0.0807 | 0.0762 |

Модель по умолчанию: **sansa**. Полные метрики, конфигурации и история эпох — в `artifacts/metrics.json`.
<!-- END RESULTS -->

## Проверки

```bash
uv run --locked --group train --group dev pytest -q
uv run --locked --group dev ruff check .
```

Тесты проверяют создание и перезапуск профилей, конкурентную выдачу ID, обновление
оценок, холодный старт и исключение просмотренного для всех четырёх моделей.
Также проверяются split, метрики, BPR-сэмплер, экспорт башен и совпадение ALS fold-in
с `implicit.recalculate_user`.

Проверено: 32 теста проходят, Ruff не находит ошибок. Предсказания экспортированной
SANSA сопоставлены с официальным `forward` на реальных профилях. В Docker проверены
HTTP-запросы для четырёх моделей и сохранение пользователя и оценок после пересоздания
контейнера с тем же volume.

## Структура

```text
movie_recs/
  data.py         # датасет, checksum, split и признаки
  models.py       # MF и neural two-tower
  classical.py    # обучение ALS и официальной SANSA
  train.py        # общий запуск обучения, выбор модели, экспорт и отчёт
  fold_in.py      # адаптация пользовательского вектора
  profiles.py     # SQLite-профили и оценки
  ranking.py      # общий top-k и offline-метрики
  service.py      # рекомендации и история пользователя
  api.py          # HTTP API
artifacts/        # обученные модели, каталог, метрики и условия датасета
tests/
Dockerfile        # targets trainer и runtime
pyproject.toml
uv.lock
```

## Условия использования данных

MovieLens разрешён для исследовательского использования; коммерческое использование
требует разрешения GroupLens. При распространении производных данных сохраняются
исходные условия; полный README включён в `artifacts/DATASET_LICENSE.txt`.
[Условия GroupLens](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*.
ACM TiiS 5(4), Article 19. <https://doi.org/10.1145/2827872>.
