<template>
  <div class="min-h-screen flex flex-col items-center justify-center bg-gray-200 p-6">
    <!-- 이미지 업로드 -->
    <div class="w-full max-w-2xl">
      <label for="upload" class="block mb-4 text-lg font-medium text-center">
        Upload an image:
      </label>
      <input
          type="file"
          id="upload"
          accept="image/*"
          class="block w-full text-sm text-gray-500
               file:mr-4 file:py-2 file:px-4
               file:rounded-lg file:border-0
               file:text-sm file:font-semibold
               file:bg-blue-500 file:text-white
               hover:file:bg-blue-600"
          @change="handleImageUpload"
      />
    </div>

    <!-- 이미지 & 캔버스 -->
    <div class="relative mt-6 max-w-5xl w-full">
      <img
          v-if="imageUrl"
          :src="imageUrl"
          ref="imageRef"
          class="w-full border rounded shadow-md"
          @load="updateCanvasSize"
      />
      <canvas ref="canvasRef" class="absolute top-0 left-0"></canvas>
    </div>

    <!-- 탐지 결과 -->
    <div class="mt-6 w-full max-w-2xl text-center">
      <h2 class="text-lg font-bold">Detection Results:</h2>
      <p v-if="loading" class="text-gray-600">Processing...</p>
      <p v-else class="bg-gray-100 p-4 rounded shadow-md text-lg w-full text-center">
        <span class="">Processing Time: {{ processTime }} ms ({{ (processTime / 1000).toFixed(1) }} s)</span>
      </p>
    </div>

    <!-- 버튼 컨트롤 -->
    <div class="mt-6 flex space-x-4">
      <button
          @click="toggleBlur"
          :class="{'bg-blue-500': isBlurActive, 'bg-gray-500': !isBlurActive}"
          class="px-4 py-2 text-white rounded-lg"
      >
        {{ isBlurActive ? 'Disable Blur' : 'Enable Blur' }}
      </button>
      <button
          @click="toggleBoundingBox"
          :class="{'bg-blue-500': isBoundingBoxActive, 'bg-gray-500': !isBoundingBoxActive}"
          class="px-4 py-2 text-white rounded-lg"
      >
        {{ isBoundingBoxActive ? 'Hide Bounding Box' : 'Show Bounding Box' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'
import * as tf from '@tensorflow/tfjs'

// 상태 변수
const imageUrl = ref<string | null>(null)
const licensePredictions = ref<any[]>([])
const facePredictions = ref<any[]>([])
const processTime = ref<number>(0) // ⏳ 처리 시간 표시
const loading = ref(false)
const imageRef = ref<HTMLImageElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)
const isBlurActive = ref(false)
const isBoundingBoxActive = ref(false)

// 모델 설정
const LICENSE_MODEL_PATH = '/license_plate/model.json' // ✅ 번호판 검출 모델
const FACE_MODEL_PATH = '/face/model.json' // ✅ 얼굴 검출 모델
const CONFIDENCE_THRESHOLD = 0.1 // ✅ 신뢰도 0.3 이상 필터링

// **📌 모델 로드 함수**
const loadModels = async () => {
  console.log("🔄 Loading models...")
  const [licenseModel, faceModel] = await Promise.all([
    tf.loadGraphModel(LICENSE_MODEL_PATH),
    tf.loadGraphModel(FACE_MODEL_PATH)
  ])
  console.log("✅ Models loaded successfully!")
  return { licenseModel, faceModel }
}

// **📌 이미지 업로드 처리 함수**
const handleImageUpload = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return

  imageUrl.value = URL.createObjectURL(file)
  loading.value = true

  await nextTick() // 이미지 렌더링 완료 후 실행

  const startTime = performance.now() // ⏳ 처리 시간 측정 시작
  const { licenseModel, faceModel } = await loadModels()

  const [licenseResults, faceResults] = await Promise.all([
    predict(licenseModel, imageRef.value!),
    predict(faceModel, imageRef.value!)
  ])

  // **✅ 신뢰도 필터링 적용**
  licensePredictions.value = filterPredictions(licenseResults)
  facePredictions.value = filterPredictions(faceResults)

  console.log('🔍 License Predictions:', licensePredictions.value)
  console.log('🔍 Face Predictions:', facePredictions.value)

  await nextTick() // 캔버스 업데이트
  drawCanvas(imageRef.value!, canvasRef.value!, licensePredictions.value, facePredictions.value)
  loading.value = false

  processTime.value = Math.round(performance.now() - startTime) // ⏳ 처리 시간 저장
}

// **📌 모델 추론 수행 함수**
const predict = async (model: tf.GraphModel, imageElement: HTMLImageElement) => {
  const tensor = tf.browser
      .fromPixels(imageElement)
      .resizeNearestNeighbor([640, 640])
      .toFloat()
      .expandDims(0)
      .div(tf.scalar(255.0))

  const predictions = await model.executeAsync(tensor) as tf.Tensor
  const reshapedPredictions = predictions.reshape([5, 8400])
  const boxesTensor = reshapedPredictions.slice([0, 0], [4, 8400])
  const scoresTensor = reshapedPredictions.slice([4, 0], [1, 8400]).squeeze()
  const classesTensor = tf.fill([8400], 0) // 클래스 인덱스

  const boxes = boxesTensor.arraySync()
  const scores = scoresTensor.arraySync()
  const classes = classesTensor.arraySync()

  // ✅ 메모리 해제
  tensor.dispose()
  predictions.dispose()
  boxesTensor.dispose()
  scoresTensor.dispose()
  classesTensor.dispose()

  return { boxes, scores, classes }
}

// **📌 신뢰도 필터링 함수**
const filterPredictions = (results: any) => {
  const { boxes, scores, classes } = results
  return scores
      .map((score, index) => score > CONFIDENCE_THRESHOLD ? index : -1)
      .filter(i => i !== -1)
      .map(i => ({
        bbox: boxes.map(box => box[i]), // 4개의 좌표 값 가져오기
        score: scores[i],
        class: classes[i]
      }))
}

// **📌 캔버스에 바운딩 박스 및 블러 처리**
const drawCanvas = (imageElement: HTMLImageElement, canvas: HTMLCanvasElement, plates: any[], faces: any[]) => {
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  canvas.width = imageElement.clientWidth
  canvas.height = imageElement.clientHeight

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height)

  const boxes = [...plates, ...faces] // ✅ 번호판 + 얼굴 바운딩 박스 합침

  boxes.forEach(({ bbox }) => {
    const [x_center, y_center, width, height] = bbox
    const x = (x_center - width / 2) * canvas.width / 640
    const y = (y_center - height / 2) * canvas.height / 640
    const w = (width * canvas.width) / 640
    const h = (height * canvas.height) / 640

    if (isBlurActive.value) {
      const tempCanvas = document.createElement("canvas")
      tempCanvas.width = w
      tempCanvas.height = h
      const tempCtx = tempCanvas.getContext("2d")
      if (!tempCtx) return

      tempCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h)
      tempCtx.filter = "blur(10px)"
      tempCtx.drawImage(tempCanvas, 0, 0, w, h)
      ctx.drawImage(tempCanvas, 0, 0, w, h, x, y, w, h)
    }

    if (isBoundingBoxActive.value) {
      ctx.strokeStyle = 'red'
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
    }
  })
}

// **📌 블러 토글 함수**
const toggleBlur = () => {
  isBlurActive.value = !isBlurActive.value
  if (imageRef.value && canvasRef.value) {
    drawCanvas(imageRef.value, canvasRef.value, licensePredictions.value, facePredictions.value)
  }
}

// **📌 바운딩 박스 토글 함수**
const toggleBoundingBox = () => {
  isBoundingBoxActive.value = !isBoundingBoxActive.value
  if (imageRef.value && canvasRef.value) {
    drawCanvas(imageRef.value, canvasRef.value, licensePredictions.value, facePredictions.value)
  }
}
</script>

<style scoped>
img {
  display: block;
  margin: auto;
}
canvas {
  display: block;
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
}
</style>