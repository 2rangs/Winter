import { defineEventHandler, readMultipartFormData } from 'h3'
import * as tf from '@tensorflow/tfjs-node'

const MODEL_PATH = './static/license_plate/model.json'
let model: tf.GraphModel | null = null

/** 모델 로드 함수 */
const initModel = async () => {
    if (!model) {
        console.log('🔄 Loading model...')
        model = await tf.loadGraphModel(`file://${MODEL_PATH}`)
        console.log('✅ Model loaded successfully')
    }
}

/** 모델 로드 상태 확인 함수 */
const isModelLoaded = () => model !== null

export default defineEventHandler(async (event) => {
    try {
        await initModel()
        if (!isModelLoaded()) throw new Error('Model is not loaded properly')

        // 이미지 업로드 처리
        const body = await readMultipartFormData(event)
        const file = body?.find((f) => f.name === 'image')

        if (!file) {
            console.error('❌ No image provided')
            return { error: 'No image provided' }
        }

        console.log('✅ Image received')

        // 이미지 → Tensor 변환 (모델 입력 크기 맞춤)
        let tensor = tf.node.decodeImage(file.data)
            .resizeBilinear([640, 640])
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255)) // 정규화 (0~1 범위)

        console.log('🖼️ Tensor Shape:', tensor.shape)

        // 모델 추론 실행
        const predictions = await model?.executeAsync([tensor]) as any;
        //  predictions shape is [1, 5, 8400]
        const reshapedPredictions = predictions.reshape([5, 8400]);
        const boxesTensor = reshapedPredictions.slice([0, 0], [4, 8400]);
        const scoresTensor = reshapedPredictions
            .slice([4, 0], [1, 8400])
            .squeeze();

        // Assuming that the category label of each box is fixed (you can adjust it as needed)
        const classesTensor = tf.fill([8400], 0);

        // 🚀 텐서를 배열로 변환
        const boxes = boxesTensor.arraySync();
        const scores = scoresTensor.arraySync();
        const classes = classesTensor.arraySync();

        console.log('✅ Processed Predictions:', { boxes, scores, classes });

        // 반환
        return { boxes, scores, classes };
    } catch (error: any) {
        console.error('❌ Error:', error.message)
        return { error: error.message || 'Unknown error occurred' }
    } finally {
        // 메모리 정리
        tf.disposeVariables()
    }
})
