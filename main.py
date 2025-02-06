import os
import logging
import csv
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from datetime import datetime
from scapy.all import sniff, wrpcap, rdpcap

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def packet_callback(packet):
    """Функция обработки каждого пакета (опционально)"""
    logging.info(f"Captured packet: {packet.summary()}")


def capture_traffic(output_file, duration):
    """
    Захват сетевого трафика.

    Args:
        output_file (str): Путь к файлу для сохранения трафика в формате PCAP.
        duration (int): Время захвата в секундах.
    """
    logging.info(f"Начало захвата трафика на {duration} секунд...")
    packets = sniff(timeout=duration, prn=packet_callback)  # Захват пакетов
    wrpcap(output_file, packets)  # Сохранение пакетов в PCAP-файл
    logging.info(f"Трафик сохранен в файл: {output_file}")


def preprocess_data(input_file, output_file):
    """
    Преобразование .pcap файла в .csv формат.

    Args:
    :param input_file: Путь к .pcap файлу.
    :param output_file: Путь к сохраненному .csv файлу.
    """
    logging.info(f"Preprocessing data from {input_file}...")
    packets = rdpcap(input_file)

    # Открытие CSV-файла для записи
    with open(output_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Заголовки CSV
        headers = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "length",
        ]
        csv_writer.writerow(headers)

        for packet in packets:
            try:
                # Извлечение характеристик пакета
                timestamp = packet.time
                src_ip = packet[1].src if packet.haslayer("IP") else None
                dst_ip = packet[1].dst if packet.haslayer("IP") else None
                src_port = (
                    packet[2].sport
                    if packet.haslayer("TCP") or packet.haslayer("UDP")
                    else 0
                )
                dst_port = (
                    packet[2].dport
                    if packet.haslayer("TCP") or packet.haslayer("UDP")
                    else 0
                )
                protocol = (
                    packet[2].name
                    if packet.haslayer("TCP") or packet.haslayer("UDP")
                    else 0
                )
                length = len(packet)

                # Запись строки в CSV
                csv_writer.writerow(
                    [timestamp, src_ip, dst_ip, src_port, dst_port, protocol, length]
                )
            except IndexError:
                # Игнорируем пакеты, у которых отсутствуют нужные поля
                logging.warning(
                    f"Пропущен пакет из-за отсутствия данных: {packet.summary()}"
                )

    logging.info(f"Данные сохранены в {output_file}")


def train_model(training_data, model_path):
    """
    Обучение модели
    
    Args:
    :param training_data: Путь к CSV-файлу с обучающей выборкой.
    :param model_path: Путь к сохраненной модели K-Means.
    """
    logging.info("Training model...")

    data = pd.read_csv(training_data)
    encoders = {
        'protocol': LabelEncoder(),
        'src_ip': LabelEncoder(),
        'dst_ip': LabelEncoder()
    }

    for col, le in encoders.items():
        data[col] = le.fit_transform(data[col])
        joblib.dump(le, os.path.join(MODEL_DIR, f"le_{col}.pkl"))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    #TODO Рассмотреть возможную реализацию метода локтя или подбор этого гиперпараметра через дендрограмму, GridSearch и оценка коэффициента силуэта, в целом много можно придумать
    # Есть также опция подумать над другим алгоритмом кластеризации: DBSCAN или OPTICS, им не нужно передавать количество кластеров,
    # однако они очень чувствительны к подбору гиперпараметров, также они возвращают out_of_bound итемы, что фактически решает задачу с аномалиями
    #
    # Также есть другая опция: рассмотреть использование одноклассового классификатора:
    # One Class SVM https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    # Пример использования https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_wine.html#sphx-glr-auto-examples-applications-plot-outlier-detection-wine-py
    #
    # IsolationForest(мне его реализация в sklearn не нравится, лучше не использовать) 
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
    # Пример использования https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py
    k = 3
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    joblib.dump(kmeans, model_path)
    
    logging.info(f"Model saved to {model_path}")


def detect_anomalies(data_path, model_path):
    """
    Детекция аномалий 
    
    Args:
    :param data_path: Путь к CSV-файлу с данными
    :param model_path: Путь к обученной модели K-Means
    
    :return: DataFrame с аномалиями
    """
    data = pd.read_csv(data_path)
    artifacts = {
        'protocol': joblib.load(os.path.join(MODEL_DIR, "le_protocol.pkl")),
        'src_ip': joblib.load(os.path.join(MODEL_DIR, "le_src_ip.pkl")),
        'dst_ip': joblib.load(os.path.join(MODEL_DIR, "le_dst_ip.pkl")),
        'scaler': joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        'model': joblib.load(model_path)
    }

    for col in ['protocol', 'src_ip', 'dst_ip']:
        le = artifacts[col]
        data[col] = data[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        
    data_scaled = artifacts['scaler'].transform(data)

    distances = artifacts['model'].transform(data_scaled).min(axis=1)
    threshold = np.percentile(distances, 95)
    anomalies = distances > threshold

    anomalous_data = data[anomalies]
    if anomalous_data.empty:
        logging.info("No anomalies detected.")
    else:
        logging.info(f"{len(anomalous_data)} anomalies detected.")

    logging.info("Anomalies detected:")
    logging.info(anomalous_data.to_string(index=False))

    anomalous_data = anomalous_data.replace(-1, 'UNKNOWN_CATEGORY')
    
    report_file = os.path.join(DATA_DIR, "anomalies_report.txt")
    with open(report_file, "w") as f:
        f.write("Anomalies Detected Report\n")
        f.write("=" * 30 + "\n")
        f.write(anomalous_data.to_string(index=False))
        f.write("\n")
    
    logging.info(f"Anomaly report saved to {report_file}")
    return anomalous_data


def visualize_clusters(data_path, model_path, output_path="cluster_plot.png"):
    """
    Визуализация кластеров на тестовой выборке.
    
    Args:
    :param data_path: Путь к CSV-файлу с тестовыми данными.
    :param model_path: Путь к сохраненной модели K-Means.
    :param scaler_path: Путь к сохраненному объекту StandardScaler.
    :param output_path: Путь для сохранения графика.
    """
    # Загрузка данных
    data = pd.read_csv(data_path)

    # Приведение данных к строковому типу
    data["src_ip"] = data["src_ip"].astype(str)
    data["dst_ip"] = data["dst_ip"].astype(str)
    data["protocol"] = data["protocol"].astype(str)

    artifacts = {
        'protocol': joblib.load(os.path.join(MODEL_DIR, "le_protocol.pkl")),
        'src_ip': joblib.load(os.path.join(MODEL_DIR, "le_src_ip.pkl")),
        'dst_ip': joblib.load(os.path.join(MODEL_DIR, "le_dst_ip.pkl")),
        'scaler': joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        'model': joblib.load(model_path)
    }

    for col in ['protocol', 'src_ip', 'dst_ip']:
        le = artifacts[col]
        data[col] = data[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    # Масштабирование данных
    data_scaled = artifacts['scaler'].transform(data)

    # Понижение размерности до 2D с помощью PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_scaled)

    # Предсказание кластеров
    clusters = artifacts['model'].predict(data_scaled)

    # Создание DataFrame для визуализации
    plot_data = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
    plot_data["Cluster"] = clusters

    # Построение графика
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="PC1", y="PC2", hue="Cluster", palette="tab10", data=plot_data, s=50
    )
    plt.title("K-Means Clustering Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", loc="upper right")
    plt.grid(True)

    # Сохранение графика
    plt.savefig(output_path)
    logging.info(f"Cluster plot saved to {output_path}")
    plt.show()


def visualize_anomalies_2d(data_path, anomalies_path):
    """
    Визуализировать аномалии на 2D-графике.

    Args:
    :param data_path: Путь к CSV файлу с тестовой выборкой.
    :param anomaliea_path: Путь к CSV-файлу с аномалиями.
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    anomalies = pd.read_csv(anomalies_path)
    
    # Разделение данных на нормальные и аномальные
    normal_data = data[~data.index.isin(anomalies.index.tolist())]

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.scatter(normal_data['length'], normal_data['src_port'], c='blue', label='Нормальные пакеты')
    plt.scatter(anomalies['length'], anomalies['src_port'], c='red', label='Аномалии')
    plt.xlabel('Length')
    plt.ylabel('src_port')
    plt.legend()
    plt.title('Визуализация обнаруженных аномалий')
    plt.show()


def main():
    def get_processed_data(input_file: str) -> str:
        """Хелпер функция, которая позволяет либо производить захват трафика, если не передан input_file
            либо обработать уже сохраненный pcap файл

        Args:
            input_file (str): входной файл
        """
        if input_file:
            pcap_file = input_file
        else:
            filename = f"{args.mode}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
            pcap_file = os.path.join(DATA_DIR, filename)
            capture_traffic(pcap_file, args.duration)
        
        processed_file = pcap_file.replace(".pcap", "_features.csv")
        preprocess_data(pcap_file, processed_file)
        
        return processed_file
    
    parser = argparse.ArgumentParser(description="Поиск аномалий в сетевой активности на хосте")
    
    parser.add_argument("--mode", type=str, required=True, choices=["train", "detect"],
                      help="Режим: обучение (train) или обнаружение аномалий (detect)")
    parser.add_argument("--duration", type=int, default=60*60*24, 
                      help="Длительность сбора ОВ в секундах")
    parser.add_argument("--input_file", type=str, 
                      help="Путь к файлу для обработки или анализа")
    parser.add_argument("--visualize", type=str, choices=["data", "anomalies"], 
                      help="Визуализировать данные")

    args = parser.parse_args()
    
    if args.visualize and args.mode != "detect":
        parser.error("Визуализация доступна только в режиме detect")
    
    if args.mode == "train":
        processed_file = get_processed_data(args.input_file)
        model_path = os.path.join(MODEL_DIR, "anomaly_model.pkl")
        train_model(processed_file, model_path)
        
    elif args.mode == "detect":
        processed_file = get_processed_data(args.input_file)
        
        model_path = os.path.join(MODEL_DIR, "anomaly_model.pkl")
        
        anomalies = detect_anomalies(processed_file, model_path)
        anomalies_output = os.path.join(DATA_DIR, "anomalies_detected.csv")
        anomalies.to_csv(anomalies_output, index=False)
        logging.info(f"Аномалии сохранены в {anomalies_output}")

        visualizations = {
            'data': lambda: visualize_clusters(processed_file, model_path),
            'anomalies': lambda: visualize_anomalies_2d(processed_file, anomalies_output)
        }
        
        if args.visualize in visualizations:
            visualizations[args.visualize]()
        

if __name__ == "__main__":
    main()