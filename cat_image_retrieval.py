"""
전이학습 + 고양이 이미지 검색 프로젝트
=====================================
- ImageNet 사전학습 ResNet50을 사용한 고양이 품종 분류기 전이학습
- 임베딩 기반 유사 이미지 검색 시스템

Author: Deep Learning Project
Framework: TensorFlow 2.x + Keras
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam


# =============================================================================
# (1) 설정/Config 클래스
# =============================================================================
class Config:
    """하이퍼파라미터 및 경로 설정을 관리하는 클래스"""
    
    # ----- 데이터 경로 -----
    data_dir = "data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    gallery_dir = os.path.join(data_dir, "gallery")
    query_dir = "query_images"
    
    # ----- 모델 저장 경로 -----
    model_save_path = "cat_breed_classifier_tf.h5"
    class_to_idx_path = "class_to_idx.json"
    gallery_features_path = "gallery_features.npy"
    gallery_meta_path = "gallery_meta.json"
    
    # ----- 이미지 설정 -----
    image_size = (224, 224)
    input_shape = (224, 224, 3)
    
    # ----- 학습 설정 -----
    batch_size = 32
    
    # Stage 1: Head만 학습 (백본 freeze)
    epochs_stage1 = 5
    lr_head = 1e-3
    
    # Stage 2: Backbone fine-tuning
    epochs_stage2 = 5
    lr_backbone = 1e-5
    
    # ----- 모델 설정 -----
    dropout_rate = 0.3
    
    # ----- 검색 설정 -----
    top_k = 5


# =============================================================================
# (2) 데이터 로딩 함수
# =============================================================================
def create_dataset(directory, shuffle=True, augment=False):
    """
    지정된 디렉토리에서 이미지 데이터셋을 생성합니다.
    
    Args:
        directory: 이미지가 있는 디렉토리 경로
        shuffle: 데이터 셔플 여부
        augment: 데이터 증강 여부 (현재 미구현, 필요시 확장 가능)
    
    Returns:
        tf.data.Dataset 객체, class_names 리스트
    """
    print(f"[데이터 로딩] {directory} 에서 데이터셋 생성 중...")
    
    # image_dataset_from_directory로 데이터셋 생성
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode="int",  # 정수 레이블 사용
        image_size=Config.image_size,
        batch_size=Config.batch_size,
        shuffle=shuffle
    )
    
    # 클래스 이름 저장
    class_names = dataset.class_names
    print(f"  -> 클래스 수: {len(class_names)}, 클래스: {class_names}")
    
    # ResNet50용 전처리 함수 적용
    def preprocess(images, labels):
        """ResNet50 전처리 함수를 적용"""
        return preprocess_input(images), labels
    
    # 전처리 및 성능 최적화
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"  -> 데이터셋 생성 완료!")
    return dataset, class_names


def create_gallery_dataset_with_paths(directory):
    """
    갤러리 데이터셋을 생성하고, 파일 경로 정보도 함께 반환합니다.
    
    Args:
        directory: 갤러리 이미지가 있는 디렉토리 경로
    
    Returns:
        tf.data.Dataset 객체, class_names, file_paths 리스트
    """
    print(f"[갤러리 데이터 로딩] {directory} 에서 데이터셋 생성 중...")
    
    # 파일 경로와 레이블을 수동으로 수집
    file_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(directory) 
                         if os.path.isdir(os.path.join(directory, d))])
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                file_paths.append(os.path.join(class_dir, filename))
                labels.append(class_to_idx[class_name])
    
    print(f"  -> 총 {len(file_paths)}개의 이미지 발견")
    print(f"  -> 클래스 수: {len(class_names)}, 클래스: {class_names}")
    
    # 기본 데이터셋 생성 (전처리 및 배치 처리를 위해)
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode="int",
        image_size=Config.image_size,
        batch_size=Config.batch_size,
        shuffle=False  # 갤러리는 순서 유지 필요
    )
    
    # 전처리 함수 적용
    def preprocess(images, labels):
        return preprocess_input(images), labels
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"  -> 갤러리 데이터셋 생성 완료!")
    return dataset, class_names, file_paths, labels


# =============================================================================
# (3) 분류 모델 구축 (전이학습)
# =============================================================================
def build_classifier(num_classes):
    """
    ResNet50 기반 분류 모델을 구축합니다.
    
    Args:
        num_classes: 분류할 클래스 수
    
    Returns:
        분류 모델, base_model (백본)
    """
    print("[모델 구축] ResNet50 기반 분류 모델 생성 중...")
    
    # ImageNet 사전학습 가중치를 가진 ResNet50 로드 (top layer 제외)
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=Config.input_shape
    )
    
    # 분류 헤드 추가
    inputs = keras.Input(shape=Config.input_shape)
    x = base_model(inputs, training=False)  # 초기에는 training=False
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(Config.dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    
    model = Model(inputs, outputs, name="cat_breed_classifier")
    
    print(f"  -> 모델 구축 완료! 총 파라미터: {model.count_params():,}")
    return model, base_model


def train_classifier():
    """
    분류 모델을 전이학습으로 학습합니다.
    Stage 1: Head만 학습 (백본 freeze)
    Stage 2: Backbone fine-tuning
    """
    print("\n" + "="*60)
    print("분류 모델 학습 시작")
    print("="*60)
    
    # ----- 데이터 로딩 -----
    train_ds, train_class_names = create_dataset(Config.train_dir, shuffle=True)
    val_ds, val_class_names = create_dataset(Config.val_dir, shuffle=False)
    
    # 클래스 이름 확인 (train과 val이 동일해야 함)
    assert train_class_names == val_class_names, "train과 val의 클래스가 다릅니다!"
    class_names = train_class_names
    num_classes = len(class_names)
    
    # class_to_idx 딕셔너리 생성
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    print(f"\n[클래스 정보] 총 {num_classes}개 클래스")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")
    
    # ----- 모델 구축 -----
    model, base_model = build_classifier(num_classes)
    
    # ----- Stage 1: Head만 학습 -----
    print("\n" + "-"*60)
    print(f"Stage 1: Head만 학습 (백본 freeze) - {Config.epochs_stage1} epochs")
    print("-"*60)
    
    # 백본 freeze
    base_model.trainable = False
    
    # 컴파일
    model.compile(
        optimizer=Adam(learning_rate=Config.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # 학습
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.epochs_stage1,
        verbose=1
    )
    
    print(f"\n  -> Stage 1 완료! Val Accuracy: {history_stage1.history['val_accuracy'][-1]:.4f}")
    
    # ----- Stage 2: Backbone fine-tuning -----
    print("\n" + "-"*60)
    print(f"Stage 2: Backbone fine-tuning - {Config.epochs_stage2} epochs")
    print("-"*60)
    
    # 백본 unfreeze
    base_model.trainable = True
    
    # 더 낮은 learning rate로 재컴파일
    model.compile(
        optimizer=Adam(learning_rate=Config.lr_backbone),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 학습
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.epochs_stage2,
        verbose=1
    )
    
    print(f"\n  -> Stage 2 완료! Val Accuracy: {history_stage2.history['val_accuracy'][-1]:.4f}")
    
    # ----- 모델 저장 -----
    print("\n[모델 저장]")
    model.save(Config.model_save_path)
    print(f"  -> 모델 저장 완료: {Config.model_save_path}")
    
    # class_to_idx 저장
    with open(Config.class_to_idx_path, 'w', encoding='utf-8') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }, f, ensure_ascii=False, indent=2)
    print(f"  -> 클래스 정보 저장 완료: {Config.class_to_idx_path}")
    
    print("\n" + "="*60)
    print("분류 모델 학습 완료!")
    print("="*60)
    
    return model, class_to_idx, idx_to_class


# =============================================================================
# (4) 임베딩 모델 구축
# =============================================================================
def build_embedding_model(classifier_model_path=None):
    """
    분류 모델에서 마지막 Dense 레이어 직전의 feature를 추출하는 임베딩 모델을 구축합니다.
    
    Args:
        classifier_model_path: 저장된 분류 모델 경로 (None이면 Config에서 가져옴)
    
    Returns:
        임베딩 모델 (출력: L2 정규화된 feature vector)
    """
    if classifier_model_path is None:
        classifier_model_path = Config.model_save_path
    
    print(f"\n[임베딩 모델 구축] {classifier_model_path} 로드 중...")
    
    # 저장된 분류 모델 로드
    classifier_model = keras.models.load_model(classifier_model_path)
    
    # GlobalAveragePooling2D 레이어의 출력을 feature로 사용
    # 모델 구조: Input -> ResNet50 -> GlobalAveragePooling2D -> Dropout -> Dense(softmax)
    # layers[-3]이 GlobalAveragePooling2D 출력
    
    # 레이어 이름으로 찾기 (더 안정적인 방법)
    feature_layer = None
    for layer in classifier_model.layers:
        if 'global_avg_pool' in layer.name or isinstance(layer, layers.GlobalAveragePooling2D):
            feature_layer = layer
            break
    
    if feature_layer is None:
        # 이름으로 못 찾으면 인덱스로
        feature_layer = classifier_model.layers[-3]
    
    print(f"  -> Feature 레이어: {feature_layer.name}")
    
    # 임베딩 모델 생성
    embedding_model = Model(
        inputs=classifier_model.input,
        outputs=feature_layer.output,
        name="embedding_model"
    )
    
    print(f"  -> 임베딩 차원: {embedding_model.output_shape[-1]}")
    print(f"  -> 임베딩 모델 구축 완료!")
    
    return embedding_model


def extract_embedding(embedding_model, images):
    """
    이미지 배치에서 L2 정규화된 임베딩을 추출합니다.
    
    Args:
        embedding_model: 임베딩 모델
        images: 전처리된 이미지 배치 [batch_size, H, W, C]
    
    Returns:
        L2 정규화된 임베딩 [batch_size, feature_dim]
    """
    features = embedding_model(images, training=False)
    # L2 정규화 적용 (길이가 1인 벡터로 만듦)
    normalized_features = tf.nn.l2_normalize(features, axis=1)
    return normalized_features


# =============================================================================
# (5) 갤러리 임베딩 계산 및 저장
# =============================================================================
def build_gallery_embeddings():
    """
    갤러리 이미지들의 임베딩을 계산하고 저장합니다.
    """
    print("\n" + "="*60)
    print("갤러리 임베딩 생성 시작")
    print("="*60)
    
    # 임베딩 모델 로드
    embedding_model = build_embedding_model()
    
    # 갤러리 데이터셋 로드
    gallery_ds, gallery_class_names, file_paths, labels = create_gallery_dataset_with_paths(
        Config.gallery_dir
    )
    
    # class_to_idx 로드 (학습 시 저장한 것)
    print(f"\n[클래스 매핑 로드] {Config.class_to_idx_path}")
    with open(Config.class_to_idx_path, 'r', encoding='utf-8') as f:
        class_info = json.load(f)
    
    train_class_to_idx = class_info['class_to_idx']
    train_idx_to_class = class_info['idx_to_class']
    
    # 갤러리 클래스와 학습 클래스 매핑 확인
    gallery_class_to_idx = {name: idx for idx, name in enumerate(gallery_class_names)}
    
    print(f"  -> 학습 클래스: {list(train_class_to_idx.keys())}")
    print(f"  -> 갤러리 클래스: {gallery_class_names}")
    
    # ----- 배치 단위로 임베딩 추출 -----
    print("\n[임베딩 추출 중...]")
    all_features = []
    all_labels = []
    
    batch_count = 0
    for images, batch_labels in gallery_ds:
        batch_count += 1
        
        # 임베딩 추출
        features = extract_embedding(embedding_model, images)
        all_features.append(features.numpy())
        all_labels.extend(batch_labels.numpy().tolist())
        
        if batch_count % 10 == 0:
            print(f"  -> {batch_count} 배치 처리 완료...")
    
    # numpy 배열로 결합
    gallery_features = np.vstack(all_features)  # [N, D]
    
    print(f"\n  -> 총 {gallery_features.shape[0]}개 이미지의 임베딩 추출 완료")
    print(f"  -> 임베딩 shape: {gallery_features.shape}")
    
    # ----- 저장 -----
    # Feature 저장
    np.save(Config.gallery_features_path, gallery_features)
    print(f"\n[저장] 갤러리 feature: {Config.gallery_features_path}")
    
    # 메타 정보 저장
    # 갤러리의 레이블을 학습 시 사용한 클래스 이름으로 매핑
    gallery_idx_to_class = {idx: name for name, idx in gallery_class_to_idx.items()}
    
    meta_data = {
        'num_images': len(all_labels),
        'feature_dim': gallery_features.shape[1],
        'labels': all_labels,  # 각 이미지의 레이블 (정수)
        'file_paths': file_paths,  # 각 이미지의 파일 경로
        'gallery_class_to_idx': gallery_class_to_idx,
        'gallery_idx_to_class': gallery_idx_to_class,
        'train_class_to_idx': train_class_to_idx,
        'train_idx_to_class': train_idx_to_class
    }
    
    with open(Config.gallery_meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    print(f"[저장] 갤러리 메타 정보: {Config.gallery_meta_path}")
    
    print("\n" + "="*60)
    print("갤러리 임베딩 생성 완료!")
    print("="*60)
    
    return gallery_features, meta_data


# =============================================================================
# (6) 쿼리 이미지 임베딩 및 검색
# =============================================================================
def load_and_preprocess_query_image(image_path):
    """
    쿼리 이미지를 로드하고 전처리합니다.
    
    Args:
        image_path: 쿼리 이미지 경로
    
    Returns:
        전처리된 이미지 [1, 224, 224, 3]
    """
    print(f"\n[쿼리 이미지 로드] {image_path}")
    
    # 이미지 로드 및 리사이즈
    img = keras.utils.load_img(
        image_path,
        target_size=Config.image_size
    )
    
    # numpy 배열로 변환
    img_array = keras.utils.img_to_array(img)
    
    # 배치 차원 추가 [1, H, W, C]
    img_array = np.expand_dims(img_array, axis=0)
    
    # ResNet50 전처리 적용
    img_array = preprocess_input(img_array)
    
    print(f"  -> 이미지 shape: {img_array.shape}")
    return img_array


def retrieve_topk(query_img_path, k=None):
    """
    쿼리 이미지와 가장 유사한 갤러리 이미지 Top-k를 검색합니다.
    
    Args:
        query_img_path: 쿼리 이미지 경로
        k: 반환할 상위 결과 수 (None이면 Config.top_k 사용)
    
    Returns:
        검색 결과 리스트 [(순위, 유사도, 클래스 이름, 파일 경로), ...]
    """
    if k is None:
        k = Config.top_k
    
    print("\n" + "="*60)
    print("이미지 검색 시작")
    print("="*60)
    
    # ----- 임베딩 모델 로드 -----
    embedding_model = build_embedding_model()
    
    # ----- 쿼리 이미지 임베딩 추출 -----
    query_img = load_and_preprocess_query_image(query_img_path)
    query_embedding = extract_embedding(embedding_model, query_img)
    query_vec = query_embedding.numpy().flatten()  # [D]
    
    print(f"  -> 쿼리 임베딩 shape: {query_vec.shape}")
    
    # ----- 갤러리 데이터 로드 -----
    print(f"\n[갤러리 데이터 로드]")
    gallery_features = np.load(Config.gallery_features_path)  # [N, D]
    
    with open(Config.gallery_meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    print(f"  -> 갤러리 이미지 수: {gallery_features.shape[0]}")
    print(f"  -> 임베딩 차원: {gallery_features.shape[1]}")
    
    # ----- 코사인 유사도 계산 -----
    # 이미 L2 정규화되어 있으므로 내적 = 코사인 유사도
    print(f"\n[코사인 유사도 계산 중...]")
    similarities = gallery_features @ query_vec  # [N]
    
    # ----- Top-k 검색 -----
    # 유사도가 큰 순서대로 정렬
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # 결과 추출
    labels = meta_data['labels']
    file_paths = meta_data['file_paths']
    idx_to_class = meta_data['gallery_idx_to_class']
    
    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        similarity = similarities[idx]
        label = labels[idx]
        class_name = idx_to_class[str(label)]
        file_path = file_paths[idx]
        results.append((rank, similarity, class_name, file_path))
    
    # ----- 결과 출력 -----
    print("\n" + "="*60)
    print(f"검색 결과: Top-{k}")
    print("="*60)
    print(f"쿼리 이미지: {query_img_path}")
    print("-"*60)
    print(f"{'순위':<6}{'유사도':<12}{'클래스 이름':<20}{'파일 경로'}")
    print("-"*60)
    
    for rank, similarity, class_name, file_path in results:
        print(f"{rank:<6}{similarity:<12.4f}{class_name:<20}{file_path}")
    
    print("="*60)
    
    return results


# =============================================================================
# (7) main() 함수
# =============================================================================
def main():
    """
    메인 실행 함수
    
    사용법:
        1) 처음 실행 시: 분류 모델 학습 + 갤러리 임베딩 생성
        2) 이후: 검색만 실행
    """
    print("\n" + "="*70)
    print("전이학습 + 고양이 이미지 검색 프로젝트")
    print("="*70)
    
    # GPU 확인
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n[GPU 감지] {len(gpus)}개의 GPU 사용 가능")
        for gpu in gpus:
            print(f"  -> {gpu}")
        # GPU 메모리 증가 허용 설정
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("  -> GPU 메모리 증가 허용 설정 완료")
        except RuntimeError as e:
            print(f"  -> GPU 설정 오류: {e}")
    else:
        print("\n[CPU 모드] GPU를 찾을 수 없습니다. CPU로 실행합니다.")
    
    print(f"\n[설정 정보]")
    print(f"  - 데이터 경로: {Config.data_dir}")
    print(f"  - 이미지 크기: {Config.image_size}")
    print(f"  - 배치 사이즈: {Config.batch_size}")
    print(f"  - Stage 1 epochs: {Config.epochs_stage1}")
    print(f"  - Stage 2 epochs: {Config.epochs_stage2}")
    
    # =========================================================================
    # 실행 모드 선택
    # =========================================================================
    # 아래 주석을 해제하여 원하는 작업을 실행하세요.
    
    # ----- 1) 분류 모델 학습 -----
    # 처음 실행하거나 모델을 다시 학습할 때 실행
    train_classifier()
    
    # ----- 2) 갤러리 임베딩 생성 -----
    # 학습 후 또는 갤러리가 변경되었을 때 실행
    build_gallery_embeddings()
    
    # ----- 3) 쿼리 이미지로 검색 -----
    # 검색만 실행할 때 (모델과 갤러리 임베딩이 이미 있을 때)
    query_image_path = os.path.join(Config.query_dir, "example_cat.jpg")
    
    # 쿼리 이미지가 존재하는지 확인
    if os.path.exists(query_image_path):
        results = retrieve_topk(query_image_path, k=Config.top_k)
    else:
        print(f"\n[경고] 쿼리 이미지를 찾을 수 없습니다: {query_image_path}")
        print("검색을 건너뜁니다. query_images/ 폴더에 example_cat.jpg를 추가해주세요.")


def search_only(query_image_path, k=5):
    """
    학습과 갤러리 생성 없이 검색만 수행하는 헬퍼 함수
    
    사용 예시:
        search_only("query_images/my_cat.jpg", k=10)
    """
    if not os.path.exists(Config.model_save_path):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {Config.model_save_path}")
        print("먼저 train_classifier()를 실행해주세요.")
        return None
    
    if not os.path.exists(Config.gallery_features_path):
        print(f"[오류] 갤러리 임베딩을 찾을 수 없습니다: {Config.gallery_features_path}")
        print("먼저 build_gallery_embeddings()를 실행해주세요.")
        return None
    
    return retrieve_topk(query_image_path, k=k)


# =============================================================================
# 실행
# =============================================================================
if __name__ == "__main__":
    main()

