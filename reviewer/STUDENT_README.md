# Финальный проект Physical AI

Два робота, четыре задачи, полный цикл RL → демонстрации → BC → робастность.
Начните с [задания](ASSIGNMENT.md). Справочник API: [MuJoCo](MUJOCO_GUIDE.md).

![Восемь сцен](docs/images/scenes.png)

## Установка

Python 3.12, Linux. На Windows используйте WSL2. GPU необязательна для просмотра
и функциональных проверок; для больших BC-экспериментов рекомендуется GPU.
CPU-версия PPO использует процессы MuJoCo. Полное обучение всех моделей существенно
дороже короткой проверки установки; измеряйте скорость на своём оборудовании.
Поставляемый рецепт запрашивает около 10.33 млн PPO-переходов суммарно для восьми
политик с переиспользованием общих стадий. Фактический бюджет округляется до
размера rollout; сбор RGB-данных и BC требуют дополнительных вычислений.

Распакуйте студенческий архив и откройте терминал в его корне, либо клонируйте
выданный организаторами студенческий репозиторий.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv libosmesa6 libgl1 libglfw3
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-lock.txt
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
python -m pytest -q
python -m physical_ai.scene_tools --robot ur5e --task cup_plate --snapshot runs/scene.png
```

Работайте из корня проекта: там находятся assets и конфигурации. Альтернатива
локальной установке — `./scripts/run_container.sh`; CPU Docker содержит системные
библиотеки, сохраняет результаты в смонтированный каталог и поддерживает snapshots
и видео без дисплея. Он не предоставляет графический рабочий стол.

Для интерактивного окна на Linux с графической сессией:

```bash
MUJOCO_GL=glfw python -m physical_ai.scene_tools --robot iiwa14 --task cup_distractor
```

Для сервера без дисплея используйте snapshots/видео. При рабочем NVIDIA-драйвере
можно выбрать `MUJOCO_GL=egl`. Для CUDA-обучения BC установите PyTorch 2.7.1
под свою CUDA-среду; поставляемый Docker использует CPU-сборку.

Сцены: `scenes/{ur5e,iiwa14}_{cup_plate,cup_shelf,cup_distractor,color_match}.xml`.
В команду просмотра подставляйте любую пару. Исходные роботы и лицензии в `assets/`.

## Проверка запуска обучения

```bash
python -m physical_ai.rl --robot ur5e --task cup_plate \
  --out runs/smoke_rl --total-steps 64 --n-steps 32 --n-envs 1
```

Это проверка обновления весов и сохранения, не обучение успешного эксперта.
Полные команды обучения, сбора и оценки находятся в `ASSIGNMENT.md`.
Начните с `configs/ppo_calibrated.json`: он задаёт проверенную последовательность
стадий PPO и сохраняет отдельный checkpoint для каждой пары робот × задача.
Промежуточные состояния curriculum — предоставленная помощь обучению; итоговый
SR измеряйте на обычных начальных состояниях.
Логи: `tensorboard --logdir runs`; метрики оценки сохраняются в JSON и CSV.

| Каталог/файл | Назначение |
|---|---|
| `physical_ai/env.py`, `scenes.py` | среда, управление, генерация сцен |
| `physical_ai/rl.py`, `curriculum.py`, `train_curriculum.py` | PPO, перенос между роботами, curriculum и воспроизведение рецепта |
| `physical_ai/data.py` | сбор и чтение эпизодов |
| `physical_ai/bc.py` | визуальный baseline, проприоцепция, экспорт |
| `physical_ai/evaluate.py` | rollout и доверительные интервалы |
| `physical_ai/scene_tools.py` | просмотр и инспектор параметров |
| `configs/` | исходные конфигурации и формат редактирования |
| `REPORT_TEMPLATE.md` | структура отчёта |
| `submission.example.json` | матрица сдаваемых моделей |

Данные и checkpoints не коммитятся автоматически. Сохраняйте их отдельно с
контрольными суммами. Источники и лицензии: `THIRD_PARTY_NOTICES.md`.
