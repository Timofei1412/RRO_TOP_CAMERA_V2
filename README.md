# RRO_TOP_CAMERA

Для запуска:

```python .\pipeline.py <путь до фотографии или ссылка на rtsp/rtmp>```

```--speed``` - флаг оптимизации пути по скорости

```--carry``` - запускает алгоритм расчета развоза труб

```-i или --interactive``` - интерактивный режим для проверки построения графа. ***(Флаги carry и interactive взаимоисключающие)***

```-d``` режим отладки, покажет все что происходит в пайплайне

```--animate``` анимация итогового пути. Закрыть можно только по завершении

```--hands``` если углы не найдены, можно будет выбрать их руками. Без этого флага будут использованы предыдущие углы (с предыдущего запуска)

```--port <номер COM порта>``` Порт для передачи данных для esp32 (для меня ```--port COM8```)


# Примеры запуска кода 
```python .\pipeline.py Test_images\NEW1Covered.png --speed --carry ```

```python .\pipeline.py rtmp://192.168.22.68/live/robocamera --carry --port COM8 --speed```

```python .\pipeline.py Test_images\OREN6.png --carry --robot-mode aruco --port COM5```