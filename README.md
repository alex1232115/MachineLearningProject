Сконфигурировать нейронную сеть для решения задачи регрессии (интерполяция значений математического выражения).
Пусть дано выражение некоторого вида с одним глобальным и множеством локальных максимумов (минимумов):
                    Например, функция вида  Z(x,y)=sin⁡(x)+cos⁡(y)-x^2

![image](https://github.com/alex1232115/MachineLearningProject/assets/75955974/cbc98aa1-7102-4639-ada6-4fcb07268406)\


Что требуется исследовать:
1.	Обучить сеть на экспериментальных данных 2 способами 
  •	Выбор x, y проводить случайным образом (метод монте-карло)
  •	Выбор x, y проводить регулярным образом (постепенно уменьшать шаг дробления области в 2 раза)
2.	Выявить зависимость «емкости» обучения и точности работы нейронной сети, построить график.
3.	Выявить зависимость числа нейронов и слоев и точности получаемых результатов (т. е. выявить влияние числа нейронов и слоев на точность)

Каждому из вас требуется работать со своей функцией (по одной из своих координат функция должна иметь периодическую зависимость). Область значений в заданном диапазоне должна содержать не менее 5 и не более 10 периодов. Функция определена на всех точках обучаемой и тестовой области.

Область тренировочных и тестовых значений находятся в одном диапазоне (мин, макс)
