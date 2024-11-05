import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import mediapipe as mp

# Habilitar mixed precision para treinamento mais rápido em GPUs modernas
from tensorflow.keras import mixed_precision

# Defina a política de mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configuração do diretório base onde as imagens estão organizadas por celebridades
base_dir = '/content/data/'

def add_mask_to_face(image, mask_path='/content/mask.png'):
    """
    Adiciona uma máscara ao rosto detectado em uma imagem.

    Parâmetros:
    - image: Imagem onde a máscara será adicionada.
    - mask_path: Caminho para a imagem da máscara a ser sobreposta.

    Retorna:
    - Imagem com a máscara adicionada ao rosto.
    """
    # Inicializa o modelo FaceMesh do MediaPipe para detecção de pontos chave do rosto
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Carrega a imagem da máscara com canal alpha (transparência)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Converte a imagem de BGR para RGB, pois o MediaPipe espera imagens em RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar os pontos chave do rosto
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Obtém as coordenadas da ponta do nariz e da parte inferior do queixo
            nose_tip = face_landmarks.landmark[1]  # Ponta do nariz
            chin_bottom = face_landmarks.landmark[152]  # Ponto mais baixo do queixo

            # Obtem a altura e largura da imagem
            h, w = image.shape[:2]

            # Converte as coordenadas normalizadas dos pontos chave para coordenadas em pixels
            nose_tip_coords = (int(nose_tip.x * w), int(nose_tip.y * h))
            chin_bottom_coords = (int(chin_bottom.x * w), int(chin_bottom.y * h))

            # Verifica se as coordenadas da máscara são válidas
            if chin_bottom_coords[0] > nose_tip_coords[0] and chin_bottom_coords[1] > nose_tip_coords[1]:
                # Calcula a largura e a altura da máscara com base na distância entre os pontos chave
                mask_width = chin_bottom_coords[0] - nose_tip_coords[0]
                mask_height = chin_bottom_coords[1] - nose_tip_coords[1]

                # Redimensiona a máscara para caber na área do rosto entre o nariz e o queixo
                mask_resized = cv2.resize(mask, (mask_width, mask_height))

                # Determina a posição para sobrepor a máscara na imagem
                mask_pos_y = nose_tip_coords[1]
                mask_pos_x = nose_tip_coords[0] - mask_resized.shape[1] // 2

                # Itera sobre os pixels da máscara e sobrepõe na imagem original
                for i in range(mask_resized.shape[0]):
                    for j in range(mask_resized.shape[1]):
                        # Apenas sobrepõe onde a máscara não é transparente
                        if mask_resized[i, j, 3] > 0:
                            if mask_pos_y + i < h and mask_pos_x + j < w:
                                image[mask_pos_y + i, mask_pos_x + j] = mask_resized[i, j, :3]

    # Libera os recursos do FaceMesh
    face_mesh.close()

    # Retorna a imagem com a máscara sobreposta
    return image

def augment_with_mask(image, mask_path='/content/mask.png'):
    """
    Função de augmentação que adiciona uma máscara a uma imagem.

    Parâmetros:
    - image: Imagem original.
    - mask_path: Caminho para a imagem da máscara.

    Retorna:
    - Imagem augmentada com máscara.
    """
    def add_mask(image):
        # Converte a imagem para uint8 (necessário para operações com OpenCV)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

        # Converte a imagem para um array NumPy
        image_np = image.numpy()

        # Adiciona a máscara ao rosto detectado na imagem
        image_np = add_mask_to_face(image_np, mask_path=mask_path)

        # Retorna a imagem com a máscara aplicada
        return image_np

    # Usa tf.py_function para integrar a função de máscara dentro do fluxo do TensorFlow
    augmented_image = tf.py_function(add_mask, [image], tf.float32)

    # Define a forma da imagem resultante para garantir compatibilidade
    augmented_image.set_shape(image.shape)

    # Normaliza a imagem para o intervalo [0, 1]
    return augmented_image / 255.0

def load_data_with_tfdata(base_dir, batch_size=64, img_size=(224, 224), mask_path='/content/mask.png'):
    """
    Carrega os dados de imagem, aplicando augmentação e convertendo os dados para tensores do TensorFlow.

    Parâmetros:
    - base_dir: Diretório base onde as imagens estão localizadas.
    - batch_size: Tamanho do lote para o carregamento dos dados.
    - img_size: Tamanho das imagens a serem redimensionadas.
    - mask_path: Caminho para a imagem da máscara.

    Retorna:
    - train_ds: Dataset de treinamento com augmentação.
    - val_ds: Dataset de validação.
    - class_names: Lista de nomes das classes.
    """
    AUTOTUNE = tf.data.AUTOTUNE  # Habilita otimizações de carregamento de dados

    # Carrega o dataset de treinamento a partir do diretório, com divisão entre treino e validação
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,  # Define 20% dos dados para validação
        subset="training",  # Define que esta parte é para treinamento
        seed=123,  # Define uma seed para garantir a reprodutibilidade
        image_size=img_size,  # Redimensiona as imagens para o tamanho especificado
        batch_size=batch_size  # Define o tamanho do lote
    )

    # Carrega o dataset de validação a partir do diretório
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        validation_split=0.2,  # Define 20% dos dados para validação
        subset="validation",  # Define que esta parte é para validação
        seed=123,  # Define uma seed para garantir a reprodutibilidade
        image_size=img_size,  # Redimensiona as imagens para o tamanho especificado
        batch_size=batch_size  # Define o tamanho do lote
    )

    # Extrai os nomes das classes (celebridades) do dataset de treinamento
    class_names = train_ds.class_names

    # Função para realizar o one-hot encoding das classes (rótulos)
    def one_hot_encode(image, label):
        label = tf.one_hot(label, depth=len(class_names))
        return image, label

    # Função de augmentação que aplica transformações aleatórias e adiciona máscara
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)  # Aplica flip horizontal aleatório
        image = tf.image.random_brightness(image, max_delta=0.2)  # Ajusta o brilho aleatoriamente
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # Converte a imagem para float32
        image = augment_with_mask(image, mask_path=mask_path)  # Aplica a máscara na imagem
        return image, label

    # Aplica as funções de one-hot encoding e augmentação ao dataset de treinamento
    train_ds = train_ds.map(one_hot_encode).map(augment).cache().prefetch(buffer_size=AUTOTUNE)

    # Aplica o one-hot encoding e a normalização ao dataset de validação
    val_ds = val_ds.map(one_hot_encode).map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(input_shape, num_classes):
    """
    Constrói o modelo de reconhecimento facial baseado na arquitetura ResNet50.

    Parâmetros:
    - input_shape: Dimensão das entradas de imagem.
    - num_classes: Número de classes de saída (número de celebridades).

    Retorna:
    - model: Modelo de rede neural compilado.
    """
    try:
        # Carrega a arquitetura ResNet50 pré-treinada no ImageNet, sem a camada de classificação final
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        # Congela todas as camadas exceto as últimas 20 para permitir treinamento das camadas finais
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Adiciona as camadas personalizadas no topo da ResNet50
        x = Flatten()(base_model.output)  # Achata as saídas da ResNet50
        x = Dense(512, activation='relu')(x)  # Adiciona uma camada densa com 512 unidades e ReLU
        x = BatchNormalization()(x)  # Adiciona BatchNormalization para estabilizar o treinamento
        x = Dropout(0.5)(x)  # Adiciona Dropout para evitar overfitting
        x = Dense(128, activation='relu')(x)  # Adiciona uma camada densa com 128 unidades e ReLU
        output = Dense(num_classes, activation='softmax', dtype='float32')(x)  # Saída softmax para classificação

        # Define o modelo final unindo as camadas personalizadas à base da ResNet50
        model = Model(inputs=base_model.input, outputs=output)

        # Compila o modelo com otimizador Adam e função de perda categorical_crossentropy
        model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Erro ao construir o modelo: {e}")
        return None

def train_model(model, train_ds, val_ds, epochs=20):
    """
    Treina o modelo de rede neural utilizando os datasets de treinamento e validação.

    Parâmetros:
    - model: O modelo de rede neural a ser treinado.
    - train_ds: Dataset de treinamento.
    - val_ds: Dataset de validação.
    - epochs: Número de épocas de treinamento.

    Retorna:
    - history: Histórico de treinamento contendo métricas como perda e acurácia.
    """
    try:
        # Define callbacks para reduzir a taxa de aprendizado e para early stopping
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        # Inicia o treinamento do modelo
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        return None

def save_model(model, model_path='modelo_reconhecimento_facial.keras'):
    """
    Salva o modelo treinado em um arquivo .keras.

    Parâmetros:
    - model: Modelo a ser salvo.
    - model_path: Caminho onde o modelo será salvo.
    """
    try:
        model.save(model_path)
        print(f"Modelo salvo em {model_path}")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")

def extract_and_save_descriptors(model, base_dir, batch_size=64, descriptors_file='descriptors.pkl', labels_file='labels.pkl'):
    """
    Extrai e salva os descritores faciais de todas as imagens do dataset.

    Parâmetros:
    - model: Modelo para extrair os descritores.
    - base_dir: Diretório base onde as imagens estão localizadas.
    - batch_size: Tamanho do lote para extração.
    - descriptors_file: Caminho para salvar os descritores.
    - labels_file: Caminho para salvar os rótulos das imagens.
    """
    try:
        # Modelo para extrair descritores, utilizando a penúltima camada da rede neural
        descriptor_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        descriptors = {}
        labels = {}

        # Carrega todas as imagens do diretório base
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            base_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False
        )

        # Extrai os descritores para cada lote de imagens
        for images, label_batch in dataset:
            descriptors_batch = descriptor_model.predict(images)
            for descriptor, label in zip(descriptors_batch, label_batch.numpy()):
                label_str = dataset.class_names[label]
                if label_str not in descriptors:
                    descriptors[label_str] = []
                    labels[label_str] = []
                descriptors[label_str].append(descriptor)
                labels[label_str].append(f"{label_str}_{len(descriptors[label_str])}.jpg")

        # Salva os descritores e rótulos em arquivos .pkl
        with open(descriptors_file, 'wb') as f:
            pickle.dump(descriptors, f)

        with open(labels_file, 'wb') as f:
            pickle.dump(labels, f)

        print(f"Descritores e rótulos salvos em {descriptors_file} e {labels_file}")
    except Exception as e:
        print(f"Erro ao extrair e salvar descritores: {e}")

def add_person_to_database(model, image_path, person_name, descriptors_file='descriptors.pkl', labels_file='labels.pkl'):
    """
    Adiciona uma nova pessoa ao banco de dados de descritores e rótulos.

    Parâmetros:
    - model: Modelo para extrair os descritores.
    - image_path: Caminho para a imagem da nova pessoa.
    - person_name: Nome da pessoa a ser adicionada.
    - descriptors_file: Caminho para o arquivo .pkl contendo os descritores.
    - labels_file: Caminho para o arquivo .pkl contendo os rótulos.
    """
    try:
        # Modelo para extrair descritores utilizando a penúltima camada
        descriptor_model = Model(inputs=model.input, outputs=model.layers[-2].output)

        # Carrega a imagem da nova pessoa e a pre-processa
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Extrai o descritor da nova pessoa
        new_descriptor = descriptor_model.predict(img_array)

        # Carrega os descritores e rótulos existentes
        with open(descriptors_file, 'rb') as f:
            descriptors = pickle.load(f)

        with open(labels_file, 'rb') as f:
            labels = pickle.load(f)

        # Adiciona o novo descritor e rótulo ao banco de dados
        if person_name not in descriptors:
            descriptors[person_name] = []
            labels[person_name] = []

        descriptors[person_name].append(new_descriptor)
        labels[person_name].append(image_path)

        # Salva os descritores e rótulos atualizados
        with open(descriptors_file, 'wb') as f:
            pickle.dump(descriptors, f)

        with open(labels_file, 'wb') as f:
            pickle.dump(labels, f)

        print(f"{person_name} adicionado(a) ao banco de dados.")
    except Exception as e:
        print(f"Erro ao adicionar a nova pessoa: {e}")

def recognize_person(model, masked_image_path, descriptors_file='descriptors.pkl'):
    """
    Realiza o reconhecimento facial comparando a imagem mascarada com o banco de dados de descritores.

    Parâmetros:
    - model: Modelo para extrair o descritor da imagem mascarada.
    - masked_image_path: Caminho para a imagem mascarada.
    - descriptors_file: Caminho para o arquivo .pkl contendo os descritores.

    Retorna:
    - best_match: Nome da pessoa com a melhor correspondência.
    - similaridade: Valor de similaridade com a melhor correspondência.
    """
    try:
        # Modelo para extrair descritores utilizando a penúltima camada
        descriptor_model = Model(inputs=model.input, outputs=model.layers[-2].output)

        # Carrega e pre-processa a imagem mascarada
        img = tf.keras.preprocessing.image.load_img(masked_image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Extrai o descritor da imagem mascarada
        masked_descriptor = descriptor_model.predict(img_array).reshape(1, -1)

        # Carrega os descritores existentes
        with open(descriptors_file, 'rb') as f:
            descriptors = pickle.load(f)

        # Calcula a similaridade entre o descritor mascarado e os descritores no banco de dados
        similarities = {}
        for label, desc_list in descriptors.items():
            similarities[label] = max([cosine_similarity(masked_descriptor, d.reshape(1, -1))[0][0] for d in desc_list])

        # Identifica a melhor correspondência com base na maior similaridade
        best_match = max(similarities, key=similarities.get)
        print(f"Melhor correspondência: {best_match} com similaridade {similarities[best_match]}")

        # Exibe a imagem mascarada e o nome da pessoa reconhecida
        plt.imshow(img)
        plt.title(f"Reconhecido(a) como: {best_match}")
        plt.show()

        return best_match, similarities[best_match]
    except Exception as e:
        print(f"Erro no reconhecimento facial: {e}")
        return None, None

# Define o caminho para a máscara que será usada na augmentação
mask_path = '/content/mask.png'

# Carregar dados com augmentação de máscaras
train_ds, val_ds, class_names = load_data_with_tfdata(base_dir, mask_path=mask_path)

# Construir e treinar o modelo
model = build_model(input_shape=(224, 224, 3), num_classes=len(class_names))
history = train_model(model, train_ds, val_ds, epochs=20)

# Salvar o modelo treinado
save_model(model)

# Extrair e salvar vetores descritores
extract_and_save_descriptors(model, base_dir)

# Adicionar uma nova pessoa ao banco de dados
new_image_path = '/content/alex_carter.png'
add_person_to_database(model, new_image_path, "Alex Carter")

# Reconhecimento facial
masked_image_path = '/content/alex_carter_masked.png'
recognize_person(model, masked_image_path)