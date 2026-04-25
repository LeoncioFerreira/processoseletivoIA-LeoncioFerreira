import tensorflow as tf
import os

# Variáveis globais declaradas
MODEL_H5 = "model.h5"
MODEL_TFLITE = "model.tflite"
MODEL_TFLITE_FP16 = "model_fp16.tflite"


def file_size(path):
    """
    Retorna o tamanho de um arquivo em bytes.

    Verifica se o arquivo existe antes de acessar, evitando erro
    em caso de caminho inválido.

    Args:
        path (str): Caminho do arquivo.

    Returns:
        int: Tamanho do arquivo em bytes, ou 0 caso não exista.
    """

    return os.path.getsize(path) if os.path.exists(path) else 0


def convert_dynamic_range(model):
    """
    Converte um modelo Keras para TensorFlow Lite aplicando
    Dynamic Range Quantization.

    Essa técnica reduz o tamanho do modelo ao quantizar os pesos,
    sem necessidade de re-treinamento, sendo adequada para execução
    em CPU e ambientes de Edge AI.

    Args:
        model (tf.keras.Model): Modelo treinado em formato Keras.

    Returns:
        bytes: Modelo convertido em formato TFLite.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_float16(model):
    """
    Converte um modelo Keras para TensorFlow Lite aplicando
    quantização em Float16.

    Essa abordagem é utilizada para fins de comparação (benchmark),
    permitindo avaliar trade-offs entre redução de tamanho e possíveis
    impactos na precisão do modelo.

    Args:
        model (tf.keras.Model): Modelo treinado em formato Keras.

    Returns:
        bytes: Modelo convertido em formato TFLite com suporte a float16.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def main():
    """
    Executa o pipeline de conversão e otimização do modelo.

    Etapas:
    1. Verifica a existência do modelo treinado (.h5)
    2. Carrega o modelo Keras
    3. Converte para TensorFlow Lite com quantização dinâmica
    4. Gera uma versão alternativa com quantização Float16 (opcional)
    5. Salva os modelos convertidos em disco
    6. Exibe o tamanho dos arquivos para comparação

    Outputs:
    - Arquivo 'model.tflite' (otimizado para Edge AI)
    - Arquivo 'model_fp16.tflite' (benchmark opcional)
    - Impressão dos tamanhos dos arquivos no terminal
    """

    if not os.path.exists(MODEL_H5):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {MODEL_H5}. Rode train_model.py primeiro.")

    print(f"Carregando {MODEL_H5}...")
    model = tf.keras.models.load_model(MODEL_H5)

    print("Iniciando conversão Dynamic Range Quantization...")
    dynamic_tfilite = convert_dynamic_range(model)

    with open(MODEL_TFLITE, "wb") as f:
        f.write(dynamic_tfilite)

    print(f"Modelo TFLite (dynamic range) salvo em: {MODEL_TFLITE}")

    print("Iniciando conversão Float16 (Bônus)...")

    try:
        fp16_tflite = convert_float16(model)

        with open(MODEL_TFLITE_FP16, "wb") as f:
            f.write(fp16_tflite)

        print(f"Modelo TFLite (float16) salvo em: {MODEL_TFLITE_FP16}")

    except Exception as exc:
        print(f"Float16 opcional não gerado: {exc}")

    print("\n--- RESULTADO DA OTIMIZAÇÃO ---")
    print(f"Tamanho {MODEL_H5}: {file_size(MODEL_H5)} bytes (Original)")
    print(
        f"Tamanho {MODEL_TFLITE}: {file_size(MODEL_TFLITE)} bytes (Otimizado Edge)")
    print(f"Tamanho {MODEL_TFLITE_FP16}: {file_size(MODEL_TFLITE_FP16)} bytes (Otimizado Float16 para fim de benchmark)")


if __name__ == "__main__":
    main()
