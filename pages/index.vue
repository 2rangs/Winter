<template>
  <div class="min-h-screen flex flex-col items-center justify-center bg-gray-200 p-4">
    <div v-if="isLoadingModel" class="text-lg font-bold">Loading Model ...</div>
    <div v-else>
      <label for="avatar" class="block mb-2 text-lg font-medium">Choose a picture:</label>
      <input
          @change="handleImageChange"
          type="file"
          id="avatar"
          name="avatar"
          accept="image/*"
          class="mb-4"
      />
    </div>

    <div class="relative">
      <img id="img_to_detect" class="hidden" />
      <canvas id="detect_result" class="border rounded shadow-md"></canvas>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';

// 모델 경로
const MODEL_URL = '/license_plate/model.json'; // 사용자 모델 경로

// 상태 관리
const isLoadingModel = ref(false);
let customModel: tf.GraphModel | null = null;

// 모델 로드
const loadModel = async () => {
  try {
    isLoadingModel.value = true;
    customModel = await loadGraphModel(MODEL_URL);
    console.log('Model loaded successfully!');
  } catch (error) {
    console.error('Error loading model:', error);
    alert('Failed to load the model. Please check the model path.');
  } finally {
    isLoadingModel.value = false;
  }
};

// 이미지 파일 선택
const handleImageChange = (event: Event) => {
  const file = (event.target as HTMLInputElement)?.files?.[0];
  if (!file) {
    console.warn('No file selected');
    return;
  }

  const img = document.getElementById('img_to_detect') as HTMLImageElement;
  const url = URL.createObjectURL(file);

  img.src = url;
  img.onload = async () => {
    await runObjectDetection(img);
  };
};

// 객체 탐지 실행
const runObjectDetection = async (img: HTMLImageElement) => {
  if (!customModel) {
    console.error('Model not loaded');
    return;
  }

  const canvas = document.getElementById('detect_result') as HTMLCanvasElement;
  const context = canvas.getContext('2d');

  canvas.width = img.width;
  canvas.height = img.height;

  // 이미지 -> 텐서 변환
  const tensor = tf.browser
      .fromPixels(img)
      .resizeNearestNeighbor([640, 640]) // 모델 입력 크기에 맞게 조정
      .toFloat()
      .div(255.0)
      .expandDims(0);

  // 모델 추론
  const predictions = await customModel.executeAsync(tensor);

  // 바운딩 박스 좌표 및 신뢰도 텐서
  const boxesTensor = predictions[0]; // 바운딩 박스 좌표
  const scoresTensor = predictions[1]; // 신뢰도

  const boxes = await boxesTensor.data(); // 1D 배열로 가져오기
  const scores = await scoresTensor.data();

  const threshold = 0.9; // 신뢰도 임계값
  const filteredBoxes = [];
  const filteredScores = [];

  // 필터링
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > threshold) {
      const boxIndex = i * 4; // 4개씩 묶임 (xMin, yMin, xMax, yMax)
      filteredBoxes.push([
        boxes[boxIndex],
        boxes[boxIndex + 1],
        boxes[boxIndex + 2],
        boxes[boxIndex + 3],
      ]);
      filteredScores.push(scores[i]);
    }
  }

  console.log('Filtered Boxes:', filteredBoxes);
  console.log('Filtered Scores:', filteredScores);

  // 캔버스에 바운딩 박스 그리기
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(img, 0, 0, canvas.width, canvas.height);

  context.font = '16px Arial';
  context.strokeStyle = 'red';
  context.fillStyle = 'red';

  filteredBoxes.forEach(([xMin, yMin, xMax, yMax], index) => {
    const x = xMin * canvas.width;
    const y = yMin * canvas.height;
    const width = (xMax - xMin) * canvas.width;
    const height = (yMax - yMin) * canvas.height;

    // Draw bounding box
    context.beginPath();
    context.rect(x, y, width, height);
    context.lineWidth = 2;
    context.stroke();

    // Draw label and score
    context.fillText(`Score: ${(filteredScores[index] * 100).toFixed(1)}%`, x, y - 5);
  });

  tensor.dispose();
  boxesTensor.dispose();
  scoresTensor.dispose();
};

onMounted(loadModel);
</script>

<style scoped>
canvas {
  width: 100%;
  max-width: 500px;
  height: auto;
}
</style>
