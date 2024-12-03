## ml_hw1_hse_ai
# EDA
Изначально было проведено EDA, где был изучен датасет.
Были найдены и удалены дубликаты.
Проаналированы основные статистики данных, после чего выяснилось, что аномальных выбросов в числовых данных нет, так как распределения в тестовых и трейн данных схожи.
Числовые признаки, которые были определены как строки из-за указания их размерностей, были преобразованы в числовые.
Далее занялся заполнением пропусков в числовых признаках, заполнение производилось при помощи медианы, так как она более устойчива к выбросам и не смещает распределение выборки.
Была произведена визуализация данных для того, чтобы увидеть какие-то зависимости между признаками и целевой переменной и между признаками. Были построены графики зависимости и хитмапы корреляции. Была реализована корреляция Спирмена вручную и проведено сравнение с библиотечной функцией, результаты оказались схожи. Наиболее зависимый признак с целевой переменной является год выпуск машины. Чем больше год, тем больше цена, при чем можно заметить, что зависимость эта больше похожа на квадратную.
# ML на вещественных признаках
Была построена простейшая модель без регуляризация и масштабирования, модель не переобучилась, но показала слабые результат метрик R^2 и MSE. Было так же реализовано adjusted-R^2, которая работает как R^2, но еще штрафует за количество признаков.
После масштабирования результаты не изменились, так как линейная регрессия не чувствительна к масштабу признаков.
Самым значимым признаком оказался max_power
Далее была использована L1 регуляризация, результаты улучшились не сильно, но занулился признак torque
При помощи GridSearchCV были подобраны гиперпараметры для L1 регуляризации и ElasticNet, обучена новая модель, но почему-то модель оказалась хуже, чем просто для вручную подобранного гиперпараметра alpha для L1.
Была реализована L0 регуляризация, которая штрафует за количество ненулевых весов, результаты оказались примерно такими же, как для L1 регуляризации
# ML с добавлением категориальных признаков
Изначально необходимо было закодировать данные при помощи OneHotEncoding
При помощи GridSearchCV были подобраны гиперпараметры для Ridge регрессии, и получили наилучший результат
# Feature Engineering
Как уже было сказано между year и selling_price, наблюдается квадратная зависимость.
Было расчитано число лошадей на количество обьема
Так же были отлогорифмированы признаки torque и km_driven
После проделанных операций ничего не изменилось после обучения с Ridge регрессией(
# Бизнес часть
Были рассчитаны две бизнес метрики
Нужно было посчитать долю прогнозов, отличающихся от реальных цен на эти авто не более чем на 10%, лучше всего себя показала модель с категориальными признаками и Ridge регрессией, на тесте она не выходит за эти рамки примерно в 30% случаев, но почему-то результат на трейне хуже чем на тесте
Вторая метрика это больший штраф за недопрогноз, тут так же лучше себя показала модель с категориальными признаками и Ridge регрессией, ее и буду брать для сервисной части
# FastAPI
Был реализован сервис, в который можно подать json обьект и получить предсказание цены по нему, так же можно передать csv файл с обьектами машин и получить список предсказанных цен для каждой машины
