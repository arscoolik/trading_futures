import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import sklearn
import sklearn.neighbors


currency_pair = 'XRPUSD'



class Strategy:
    def __init__(self) -> None:
        self.historical_data = pd.read_excel(f'{currency_pair}.xlsx') 
        self.knn_model = None
        self.historical_data = self.historical_data.dropna(subset=['maxDown'])


    def train_knn_model(self) -> None:
            """Обучаем модель KNN на исторических данных"""
            features = self.historical_data[['min', 'max', 'Relative Standard Deviation',
                                            'big_candles', 'Mean Tick Volume', 'extremas_num', 'maxDown', 'maxUp']]

            # Обучаем модель KNN для поиска ближайшего похожего дня
            self.knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)  # Один ближайший сосед
            self.knn_model.fit(features, np.arange(len(features)))  # Целевые значения - индексы строк

    def find_similar_day(self, yesterday_data):
        """Ищем наиболее похожий день на вчерашний"""
        if self.knn_model is None:
            raise ValueError("KNN model is not trained yet.")

        # Ищем индекс наиболее похожего дня
        similar_day_index = self.knn_model.predict([yesterday_data])
        print(self.historical_data.iloc[similar_day_index[0] + 1])
        return similar_day_index[0]

    def was_next_day_profitable(self, similar_day_index):
        """Проверяем, был ли следующий день прибыльным для стратегии на понижение"""
        if similar_day_index >= len(self.historical_data) - 1:
            return False

        next_day = self.historical_data.iloc[similar_day_index + 1]
        return (next_day['maxDown'] == next_day['maxUp']) and (next_day['maxDown'] < 0)

    def should_enter_trade(self, yesterday_data):
        """Решение о входе в сделку на основе поиска похожего дня и анализа следующего дня"""
        similar_day_index = self.find_similar_day(yesterday_data)
        is_profitable = self.was_next_day_profitable(similar_day_index)
        return is_profitable




new_record = pd.Series({
    'min': 0.5746,
    'max': 0.6,
    'Relative Standard Deviation':  1,
    'big_candles': 10,
    'Mean Tick Volume': 11.69,
    'extremas_num': 55,
    'maxDown': -1.96,
    'maxUp': 1.13
})

s = Strategy()
s.train_knn_model()
print(s.should_enter_trade(new_record))